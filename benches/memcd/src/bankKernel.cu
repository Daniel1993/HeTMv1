#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <curand_kernel.h>
#include "helper_cuda.h"
#include "helper_timer.h"
#include <time.h>
#include <math.h>

#include "hetm-log.h"
#include "bankKernel.cuh"
#include "bitmap.hpp"

#include "pr-stm-wrapper.cuh" // enables the granularity
#include "setupKernels.cuh"

#define INTEREST_RATE 0.5 // bank readIntensive transaction
#define FULL_MASK 0xffffffff

/****************************************************************************
 *	GLOBALS
 ****************************************************************************/

__constant__ __device__ long  BANK_SIZE;
// __constant__ __device__ const int BANK_NB_TRANSFERS;
__constant__ __device__ int   HASH_NUM;
__constant__ __device__ int   num_ways;
__constant__ __device__ int   num_sets;
__constant__ __device__ int   txsPerGPUThread;
__constant__ __device__ int   txsInKernel;
__constant__ __device__ int   hprob;
__constant__ __device__ int   prec_read_intensive;
__constant__ __device__ float hmult;
__constant__ __device__ int   read_intensive_size;

__constant__ __device__ thread_data_t devParsedData;

__device__ static unsigned memcached_global_clock = 0;

__constant__ __device__ int PR_maxNbRetries = 16;

/****************************************************************************
 *	KERNELS
 ****************************************************************************/

//   Memory access layout
// +------------+--------------------+
// | NOT ACCESS |      CPU_PART      |
// +------------+--------------------+
// +------------------+--------------+
// |     GPU_PART     |  NOT ACCESS  |
// +------------------+--------------+
//
// random Function random several different numbers and store them into idx(Local array which stores idx of every source in Global memory).
__device__ void random_Kernel(PR_txCallDefArgs, int *idx, int size, int is_intersection)
{
	int i = 0;
	int devId = args.devId;

#if BANK_PART == 1
	// break the dataset in CPU/GPU
	int is_intersect = is_intersection; //IS_INTERSECT_HIT(PR_rand(100000));
#endif

	int randVal = PR_rand(INT_MAX);
#if BANK_PART == 1
	if (is_intersect) {
		idx[i] = INTERSECT_ACCESS_GPU(devId, randVal, (size-BANK_NB_TRANSFERS-1));
	} else {
		idx[i] = GPU_ACCESS(devId, randVal, (size-BANK_NB_TRANSFERS-1));
	}
#elif BANK_PART == 2
	int is_hot = IS_ACCESS_H(devId, randVal, hprob);
	randVal = PR_rand(INT_MAX);
	if (is_hot) {
		idx[i] = GPU_ACCESS_H(devId, randVal, hmult, size);
	} else {
		idx[i] = GPU_ACCESS_M(devId, randVal, hmult, size);
	}
#else
	idx[i] = INTERSECT_ACCESS_GPU(devId, randVal, (size-BANK_NB_TRANSFERS-1));
#endif

	// TODO: accounts are consecutive
	// generates the target accounts for the transaction
	for (i = 1; i < BANK_NB_TRANSFERS; i++) {
		idx[i] = (idx[i-1] + 1) % GPU_TOP_IDX(devId, size);
	}
}

__device__ void random_KernelReadIntensive(PR_txCallDefArgs, int *idx, int size, int is_intersection)
{
	int i = 0;
	int devId = args.devId;

#if BANK_PART == 1
	// break the dataset in CPU/GPU
	int is_intersect = is_intersection; //IS_INTERSECT_HIT(PR_rand(100000));
#endif

	int randVal = PR_rand(INT_MAX);
#if BANK_PART == 1
	if (is_intersect) {
		idx[i] = INTERSECT_ACCESS_GPU(devId, randVal, (size-10*BANK_NB_TRANSFERS-1));
	} else {
		idx[i] = GPU_ACCESS(devId, randVal, (size-10*BANK_NB_TRANSFERS-1));
	}
#elif BANK_PART == 2
	int is_hot = IS_ACCESS_H(randVal, hprob);
	randVal = PR_rand(INT_MAX);
	if (is_hot) {
		idx[i] = GPU_ACCESS_H(devId, randVal, hmult, size);
	} else {
		idx[i] = GPU_ACCESS_M(devId, randVal, hmult, size);
	}
#else
	idx[i] = INTERSECT_ACCESS_GPU(devId, randVal, (size-10*BANK_NB_TRANSFERS-1));
#endif

	for (i = 1; i < BANK_NB_TRANSFERS*10; i++) {
		idx[i] = (idx[i-1] + 1) % GPU_TOP_IDX(devId, size);
	}
}

#define COMPUTE_TRANSFER(val) \
	val // TODO: do math that does not kill the final result

__device__ void readIntensive_tx(PR_txCallDefArgs, int txCount)
{
	const int max_size_of_reads = 20;

	int id = threadIdx.x+blockDim.x*blockIdx.x;
	int i = 0, j;	//how many transactions one thread need to commit
	int idx[max_size_of_reads];
	HeTM_bankTx_input_s *input = (HeTM_bankTx_input_s*)args.inBuf;
	PR_GRANULE_T *accounts = (PR_GRANULE_T*)input->accounts;
	size_t nbAccounts = input->nbAccounts;
	int *input_buffer = input->input_buffer;
	int *output_buffer = input->output_buffer;
	int option = PR_rand(INT_MAX);
	int tot_nb_threads = blockDim.x*gridDim.x;
	// int need_to_extra_read = 1;

	// TODO:
	// random_KernelReadIntensive(PR_txCallArgs, idx, nbAccounts, is_intersection);

	__syncthreads();
	idx[0] = input_buffer[id+txCount*tot_nb_threads];
	// need_to_extra_read = (need_to_extra_read && idx[0] == id) ? 0 : 1;
	#pragma unroll
	for (i = 1; i < read_intensive_size; i++) {
		idx[i] = idx[i-1] + 2; // access even accounts
		// need_to_extra_read = (need_to_extra_read && idx[i] == (id % 4096)) ? 0 : 1;
	}

	// TODO:
	output_buffer[id] = 1;

#ifndef BANK_DISABLE_PRSTM
	int nbRetries = 0;
	PR_txBegin();
	if (nbRetries++ > PR_maxNbRetries) break;
#endif /* BANK_DISABLE_PRSTM */

	// reads the accounts first, then mutates the locations
	float resF = 0;
	int resI;
	for (j = 0; j < read_intensive_size; ++j)	{ // saves 1 slot for the write
		if (idx[j] < 0 || idx[j] >= nbAccounts) {
			break;
		}
#ifndef BANK_DISABLE_PRSTM

		resF += (float)PR_read(accounts + idx[j]) * INTEREST_RATE;
		// resF = cos(resF);
#else /* PR-STM disabled */
		resF += accounts[idx[j]];
#endif
	}
	resF += 1.0; // adds at least 1
	resI = (int)resF;

	__syncthreads();
#ifndef BANK_DISABLE_PRSTM
	// TODO: writes in the beginning (less transfers)
	PR_write(accounts + idx[0], resI); //write changes to write set

	// if (need_to_extra_read) {
	// 	PR_read(accounts + (id%4096)); // read-before-write
	// }
	// int target = (id * 2) % nbAccounts;
	// PR_write(accounts + target, resI); //write changes to write set

#else /* PR-STM disabled */
	accounts[idx[0]] = resI; //write changes to write set
#endif

#ifndef BANK_DISABLE_PRSTM
	PR_txCommit();
#else
	if (args.nbCommits != NULL) args.nbCommits[PR_THREAD_IDX]++;
#endif
}

__device__ void readOnly_tx(PR_txCallDefArgs, int txCount)
{
	const int max_size_of_reads = 20;

	int id = threadIdx.x+blockDim.x*blockIdx.x;
	int i = 0, j;	//how many transactions one thread need to commit
	int idx[max_size_of_reads];
	HeTM_bankTx_input_s *input = (HeTM_bankTx_input_s*)args.inBuf;
	PR_GRANULE_T *accounts = (PR_GRANULE_T*)input->accounts;
	size_t nbAccounts = input->nbAccounts;
	int *input_buffer = input->input_buffer;
	int *output_buffer = input->output_buffer;
	int option = PR_rand(INT_MAX);
	int tot_nb_threads = blockDim.x*gridDim.x;

	__syncthreads();
	idx[0] = (input_buffer[id+txCount*tot_nb_threads]) % nbAccounts;
	#pragma unroll
	for (i = 1; i < read_intensive_size; i++) {
		idx[i] = (idx[i-1] + 2) % nbAccounts; // access even accounts
	}

	// TODO:
	output_buffer[id] = 1;

#ifndef BANK_DISABLE_PRSTM
	int nbRetries = 0;
	PR_txBegin();
	if (nbRetries++ > PR_maxNbRetries) break;
#endif /* BANK_DISABLE_PRSTM */

	// reads the accounts first, then mutates the locations
	float resF = 0;
	// int resI;
	for (j = 0; j < read_intensive_size; ++j)	{ // saves 1 slot for the write
		if (idx[j] < 0 || idx[j] >= nbAccounts) {
			break;
		}
#ifndef BANK_DISABLE_PRSTM

		// int targetAccount = idx[j] % (nbAccounts);
		resF += (float)PR_read(accounts + idx[j]) * INTEREST_RATE;
		// resF = cos(resF);

#else /* PR-STM disabled */
		resF += accounts[idx[j]];
#endif
	}
	resF += 1.0; // adds at least 1
	// resI = (int)resF;

	__syncthreads();
// #ifndef BANK_DISABLE_PRSTM
// 	// TODO: writes in the beginning (less transfers)
// 	PR_write(accounts + idx[0], resI); //write changes to write set
// #else /* PR-STM disabled */
// 	accounts[idx[0]] = resI; //write changes to write set
// #endif

#ifndef BANK_DISABLE_PRSTM
	PR_txCommit();
#else /* BANK_DISABLE_PRSTM */
	if (args.nbCommits != NULL) args.nbCommits[PR_THREAD_IDX]++;
#endif /* BANK_DISABLE_PRSTM */
}

__device__ void update_tx(PR_txCallDefArgs, int txCount)
{
	const int max_size_of_reads = 20;

	int id = threadIdx.x+blockDim.x*blockIdx.x;
	int i = 0, j; //how many transactions one thread need to commit
	int target;
	PR_GRANULE_T nval;
	int idx[max_size_of_reads];
	double count_amount = 0;
	HeTM_bankTx_input_s *input = (HeTM_bankTx_input_s*)args.inBuf;
	PR_GRANULE_T *accounts = (PR_GRANULE_T*)input->accounts;
	// int is_intersection = input->is_intersection;
	size_t nbAccounts = input->nbAccounts;
	int *input_buffer = input->input_buffer;
	int *output_buffer = input->output_buffer;
	int option = PR_rand(INT_MAX);
	int tot_nb_threads = blockDim.x*gridDim.x;

	// before the transaction pick the accounts (needs PR-STM stuff)
	// random_Kernel(PR_txCallArgs, idx, nbAccounts, is_intersection); //get random index
	__syncthreads();
	idx[0] = input_buffer[id+txCount*tot_nb_threads];
	#pragma unroll
	#pragma unroll
	for (i = 1; i < read_intensive_size; i++) {
		idx[i] = (idx[i-1] + 2) % nbAccounts; // access even accounts
	}
	output_buffer[id] = 0;

#ifndef BANK_DISABLE_PRSTM
	int nbRetries = 0;
	PR_txBegin();
	if (nbRetries++ > PR_maxNbRetries) break;
#endif

	// reads the accounts first, then mutates the locations
	#pragma unroll
	for (j = 0; j < read_intensive_size; j++)	{
		if (idx[j] < 0 || idx[j] >= nbAccounts) {
			break;
		}
#ifndef BANK_DISABLE_PRSTM
		count_amount += PR_read(accounts + idx[j]);
#else /* PR-STM disabled */
		count_amount += accounts[idx[j]];
#endif
	}

	// TODO: make the GPU slower
	target = 0;
	if (idx[target] < 0 || idx[target] >= nbAccounts) {
		break;
	}
	// nval = COMPUTE_TRANSFER(reads[target] - 1); // -money
	nval = COMPUTE_TRANSFER(count_amount - 1); // -money
	if (idx[target] < nbAccounts) {
#ifndef BANK_DISABLE_PRSTM
		PR_write(accounts + idx[target], nval); //write changes to write set
#else /* PR-STM disabled */
		accounts[idx[target]] = nval; //write changes to write set
#endif
	}
	// __syncthreads();
// TODO: only 1 write now
// 		target = j*2+1;
// 		__syncthreads();
// 		nval = COMPUTE_TRANSFER(reads[target] + 1); // +money
// #ifndef BANK_DISABLE_PRSTM
// 		PR_write(accounts + idx[target], nval); //write changes to write set
// #else /* PR-STM disabled */
// 		accounts[idx[target]] = nval; //write changes to write set
// #endif
#ifndef BANK_DISABLE_PRSTM
	PR_txCommit();
#else
	if (args.nbCommits != NULL) args.nbCommits[PR_THREAD_IDX]++;
#endif
}

__device__ void updateReadOnly_tx(PR_txCallDefArgs, int txCount)
{
	int id = threadIdx.x+blockDim.x*blockIdx.x;
	int i = 0, j; //how many transactions one thread need to commit
	// int target;
	// PR_GRANULE_T nval;
	int idx[BANK_NB_TRANSFERS];
	PR_GRANULE_T reads[BANK_NB_TRANSFERS];
	HeTM_bankTx_input_s *input = (HeTM_bankTx_input_s*)args.inBuf;
	PR_GRANULE_T *accounts = (PR_GRANULE_T*)input->accounts;
	// int is_intersection = input->is_intersection;
	size_t nbAccounts = input->nbAccounts;
	int *input_buffer = input->input_buffer;
	int *output_buffer = input->output_buffer;
	int option = PR_rand(INT_MAX);
	int tot_nb_threads = blockDim.x*gridDim.x;

	// before the transaction pick the accounts (needs PR-STM stuff)
	// random_Kernel(PR_txCallArgs, idx, nbAccounts, is_intersection); //get random index
	// __syncthreads();
	idx[0] = input_buffer[id+txCount*tot_nb_threads];
	#pragma unroll
	for (i = 1; i < BANK_NB_TRANSFERS; i++) {
		idx[i] = idx[i-1] + devParsedData.access_offset;
	}
	output_buffer[id] = 0;

#ifndef BANK_DISABLE_PRSTM
	int nbRetries = 0;
	PR_txBegin();
	if (nbRetries++ > PR_maxNbRetries) break;
#endif

	// reads the accounts first, then mutates the locations
	for (j = 0; j < BANK_NB_TRANSFERS; j++)	{
		if (idx[j] < 0 || idx[j] >= nbAccounts) {
			break;
		}
#ifndef BANK_DISABLE_PRSTM
		reads[j] = PR_read(accounts + idx[j]);
#else /* PR-STM disabled */
		reads[j] = accounts[idx[j]];
#endif
	}
#ifndef BANK_DISABLE_PRSTM
	PR_txCommit();
#else
	if (args.nbCommits != NULL) args.nbCommits[PR_THREAD_IDX]++;
#endif

	output_buffer[id] = reads[0] + reads[1];
}

// TODO: MemcachedGPU part
// TODO: R/W-set size is NUMBER_WAYS here
// ----------------------

__global__ void memcd_check(PR_GRANULE_T* keys, size_t size)
{
	int id = threadIdx.x+blockDim.x*blockIdx.x;
	PR_GRANULE_T* stateBegin = &keys[size*3];

	if (id < size)
	printf("%i : KEY=%i STATE=%i (valid=%i)\n", id, keys[id], stateBegin[id], stateBegin[id] & MEMCD_VALID);
}

__global__
void emptyKernelTx(PR_globalKernelArgs) { }

/*********************************
 *	readKernelTransaction()
 *
 *  Main PR-STM transaction kernel
 **********************************/
 // PR_MAX_RWSET_SIZE = 4
__global__
// __launch_bounds__(1024, 1) // TODO: what is this for?
void memcdReadTx(PR_globalKernelArgs)
{
	int tid = PR_THREAD_IDX;

	PR_enterKernel(tid);

	int id = threadIdx.x+blockDim.x*blockIdx.x;
	int wayId = id % (num_ways /*+ devParsedData.trans*/);
	int targetKeyIdx = id / (num_ways + devParsedData.trans); // id of the key that each thread will take

	// num_ways threads will colaborate for the same input
	// REQUIREMENT: 1 block >= num_ways

	HeTM_memcdTx_input_s *input = (HeTM_memcdTx_input_s*)args.inBuf;
	PR_GRANULE_T       *keys = (PR_GRANULE_T*)input->key;
	// PR_GRANULE_T  *extraKeys = (PR_GRANULE_T*)input->extraKey;
	PR_GRANULE_T     *values = (PR_GRANULE_T*)input->val;
	PR_GRANULE_T  *extraVals = (PR_GRANULE_T*)input->extraVal;
	PR_GRANULE_T     *ts_CPU = (PR_GRANULE_T*)input->ts_CPU;
	PR_GRANULE_T     *ts_GPU = (PR_GRANULE_T*)input->ts_GPU;
	PR_GRANULE_T      *state = (PR_GRANULE_T*)input->state;
	// PR_GRANULE_T   *setUsage = (PR_GRANULE_T*)input->setUsage;

	// TODO: out is NULL
	memcd_get_output_t  *out = (memcd_get_output_t*)input->output;
	// int           curr_clock = *((int*)input->curr_clock);
	int          *input_keys = (int*)input->input_keys;
	int               nbSets = input->nbSets;
	int               nbWays = input->nbWays;
	int            sizeCache = nbSets * nbWays;

	// __shared__ int foundKey[1024];
	// foundKey[threadIdx.x] = 0;
	// __syncthreads();

	HeTM_GPU_log_s *GPU_log = (HeTM_GPU_log_s*)args.pr_args_ext;
	volatile long *GPU_log_state = GPU_log->state;

	// TODO: new parameter for the spins
	for (volatile int i = 0; i < devParsedData.GPU_backoff; i++) {
		GPU_log_state[id] += id + 1234 * i;
	}

	for (int i = 0; i < devParsedData.trans; ++i) { // num_ways keys
		out[threadIdx.x+blockDim.x*blockIdx.x*devParsedData.trans + i].isFound = 0;
	}

	for (int i = 0; i < (nbWays + devParsedData.trans); ++i) { // num_ways keys
		// TODO: for some reason input_key == 0 does not work --> PR-STM loops forever
		int alreadyTakenClock = 0;
		unsigned memcd_clock_val;
		int takeClockRetries = 0;
		PR_txBegin();
		int input_key = input_keys[targetKeyIdx + i]; // input size is well defined

		// int target_set = input_key % nbSets;
		// int thread_pos = target_set*nbWays + wayId;
// #if BANK_PART == 1 /* use MOD 3 */
// 		int mod_key = input_key % nbSets;
// 		int target_set = (mod_key / 3 + (mod_key % 3) * (nbSets / 3)) % nbSets;
// #else /* use MOD 2 */
// 		int mod_key = input_key % nbSets;
// 		int target_set = (mod_key / 2 + (mod_key % 2) * (nbSets / 2)) % nbSets;
// #endif
		int mod_key = input_key % nbSets;
		// int mod_key = (input_key>>4) % nbSets;
		int target_set = mod_key; // / 3 + (mod_key % 3) * (nbSets / 3);

		int thread_pos = target_set*nbWays + wayId;

		int thread_is_found;
		volatile PR_GRANULE_T thread_key;
		// volatile PR_GRANULE_T thread_key_check;
		PR_GRANULE_T thread_val;
		PR_GRANULE_T thread_val1;
		PR_GRANULE_T thread_val2;
		PR_GRANULE_T thread_val3;
		PR_GRANULE_T thread_val4;
		PR_GRANULE_T thread_val5;
		PR_GRANULE_T thread_val6;
		PR_GRANULE_T thread_val7;
		PR_GRANULE_T thread_state;

		// PR_write(&timestamps[thread_pos], curr_clock);
		thread_key = keys[thread_pos];
		thread_state = state[thread_pos];

		// __syncthreads(); // each num_ways thread helps on processing the targetKey

		// TODO: divergency here
		thread_is_found = (thread_key == input_key && ((thread_state & MEMCD_VALID) != 0));

		if (thread_is_found) {
			// int nbRetries = 0;
			int ts_val_CPU, ts_val_GPU;
			unsigned ts;
			if (!alreadyTakenClock && takeClockRetries > 2) {
				memcd_clock_val = atomicAdd(&memcached_global_clock, 1);//memcached_global_clock+1;//
				alreadyTakenClock = 1;
			}
			if (!alreadyTakenClock) {
				unsigned memcd_clock_val = PR_read(&memcached_global_clock);//memcached_global_clock+1;//
				PR_write(&memcached_global_clock, memcd_clock_val + 1);
				takeClockRetries++;
			}
			// if (nbRetries > 32) {
			// 	aborted[threadIdx.x] = 1;
			// 	i--; // TODO: should test the key again
			// 	// aborted[targetKeyIdx] = 1;
			// 	// printf("Thread%i blocked thread_key=%i thread_pos=%i ts=%i curr_clock=%i\n", id, thread_key, thread_pos, ts, curr_clock);
			// 	nbRetries = 0;
			// 	break; // retry, key changed
			// }
			// nbRetries++;

			// // TODO: some BUG on verifying the key (blocks PR-STM)
			// /*thread_key_check = */PR_read(&keys[thread_pos]);
			// PR_read(&extraKeys[thread_pos]);
			// PR_read(&extraKeys[thread_pos+sizeCache]);
			// PR_read(&extraKeys[thread_pos+2*sizeCache]);
			//
			// // TODO:
			// if (thread_key != thread_key_check) {
			// 	i--; // retry, key changed
			// 	nbRetries = 0;
			// 	break; // breaks the PR_txBegin() loop
			// }

			ts_val_CPU = ts_CPU[thread_pos]; // hack here
			ts_val_GPU = PR_read(&ts_GPU[thread_pos]);
			ts = max((unsigned)ts_val_CPU, (unsigned)ts_val_GPU);

			thread_val = PR_read(&values[thread_pos]);
			thread_val1 = PR_read(&extraVals[thread_pos]);
			thread_val2 = PR_read(&extraVals[thread_pos+sizeCache]);
			thread_val3 = PR_read(&extraVals[thread_pos+2*sizeCache]);
			thread_val4 = PR_read(&extraVals[thread_pos+3*sizeCache]);
			thread_val5 = PR_read(&extraVals[thread_pos+4*sizeCache]);
			thread_val6 = PR_read(&extraVals[thread_pos+5*sizeCache]);
			thread_val7 = PR_read(&extraVals[thread_pos+6*sizeCache]);

			if (ts < memcd_clock_val) {
				PR_write(&ts_GPU[thread_pos], memcd_clock_val);
				// PR_read(&setUsage[target_set]); // "locks" the set --> only needed for SETs
			}

			out[targetKeyIdx + i].isFound = 1;
			out[targetKeyIdx + i].value = thread_val;
			out[targetKeyIdx + i].val2 = thread_val1;
			out[targetKeyIdx + i].val3 = thread_val2;
			out[targetKeyIdx + i].val4 = thread_val3;
			out[targetKeyIdx + i].val5 = thread_val4;
			out[targetKeyIdx + i].val6 = thread_val5;
			out[targetKeyIdx + i].val7 = thread_val6;
			out[targetKeyIdx + i].val8 = thread_val7;
			// for (int j = 0; j < nbWays; ++j) {
			// 	foundKey[threadIdx.x - wayId + j] = 1;
			// }
			// printf("Found key %i \n", input_key);
		}

		PR_txCommit(); // TODO: do we want tansactions this long?

		// __syncthreads();
		// if (!foundKey[threadIdx.x]) {
		// 	printf("Key %i not found \n", input_key);
		// }
	}

	PR_exitKernel();
}

// TODO: IMPORTANT ---> set PR_MAX_RWSET_SIZE to number of ways

/*********************************
 *	writeKernelTransaction()
 *
 *  Main PR-STM transaction kernel
 **********************************/
__global__ void memcdWriteTx(PR_globalKernelArgs)
{
	int tid = PR_THREAD_IDX;

	PR_enterKernel(tid);

	// TODO: blockDim.x must be multiple of num_ways --> else this does not work
	int id = threadIdx.x+blockDim.x*blockIdx.x;
	// TODO: too much memory (this should be blockDim.x / num_ways)
	// TODO: 32 --> min num_ways == 8 for 256 block size
	// TODO: I'm using warps --> max num_ways is 32 (CAN BE EXTENDED!)
	const int maxWarpSlices = 32; // 32*32 == 1024
	int warpSliceID = threadIdx.x / num_ways;
	int wayId = id % (num_ways /*+ devParsedData.trans*/);
	int reductionID = wayId / 32;
	int reductionSize = max(num_ways / 32, 1);
	int targetKey = id / (num_ways + devParsedData.trans); // id of the key that each group of num_ways thread will take

	__shared__ int reduction_is_found[maxWarpSlices]; // TODO: use shuffle instead
	__shared__ int reduction_is_empty[maxWarpSlices]; // TODO: use shuffle instead
	__shared__ int reduction_empty_min_id[maxWarpSlices]; // TODO: use shuffle instead
	__shared__ int reduction_min_ts[maxWarpSlices]; // TODO: use shuffle instead

	// num_ways threads will colaborate for the same input
	// REQUIREMENT: 1 block >= num_ways

	__shared__ int failed_to_insert[256]; // TODO
	if (wayId == 0) failed_to_insert[warpSliceID] = 0;

	HeTM_GPU_log_s *GPU_log = (HeTM_GPU_log_s*)args.pr_args_ext;
	volatile long *GPU_log_state = GPU_log->state;

	// TODO: new parameter for the spins
	for (volatile int i = 0; i < devParsedData.GPU_backoff; i++) {
		GPU_log_state[id] += id + 1234 * i;
	}

	HeTM_memcdTx_input_s *input = (HeTM_memcdTx_input_s*)args.inBuf;
	memcd_get_output_t  *out = (memcd_get_output_t*)input->output;
	PR_GRANULE_T       *keys = (PR_GRANULE_T*)input->key;
	PR_GRANULE_T  *extraKeys = (PR_GRANULE_T*)input->extraKey;
	PR_GRANULE_T     *values = (PR_GRANULE_T*)input->val;
	PR_GRANULE_T  *extraVals = (PR_GRANULE_T*)input->extraVal;
	PR_GRANULE_T     *ts_CPU = (PR_GRANULE_T*)input->ts_CPU;
	PR_GRANULE_T     *ts_GPU = (PR_GRANULE_T*)input->ts_GPU;
	PR_GRANULE_T      *state = (PR_GRANULE_T*)input->state;
	PR_GRANULE_T   *setUsage = (PR_GRANULE_T*)input->setUsage;
	int           curr_clock = *((int*)input->curr_clock);
	int          *input_keys = (int*)input->input_keys;
	int          *input_vals = (int*)input->input_vals;
	int               nbSets = (int)input->nbSets;
	int               nbWays = (int)input->nbWays;
	int            sizeCache = nbSets*nbWays;

	int thread_is_found; // TODO: use shuffle instead
	int thread_is_empty; // TODO: use shuffle instead
	// int thread_is_older; // TODO: use shuffle instead
	PR_GRANULE_T thread_key;
	// PR_GRANULE_T thread_val;
	PR_GRANULE_T thread_ts;
	PR_GRANULE_T thread_state;

	int checkKey;
	// int checkKey1;
	// int checkKey2;
	// int checkKey3;
	int maxRetries = 0;

	// TODO: write kernel is too slow
	for (int i = 0; i < nbWays + devParsedData.trans; ++i) {

		__syncthreads(); // TODO: check with and without this
		// TODO
		if (failed_to_insert[warpSliceID] && maxRetries < 64) { // TODO: blocks
			maxRetries++;
			i--;
		}
		__syncthreads(); // TODO: check with and without this
		if (wayId == 0) failed_to_insert[warpSliceID] = 0;

		// TODO: problem with the GET
		int input_key = input_keys[targetKey + i]; // input size is well defined
		int input_val = input_vals[targetKey + i]; // input size is well defined
		// int target_set = input_key % nbSets;
		// int thread_pos = target_set*nbWays + wayId;

// #if BANK_PART == 1 /* use MOD 3 */
// 		int mod_key = input_key % nbSets;
// 		int target_set = (mod_key / 3 + (mod_key % 3) * (nbSets / 3)) % nbSets;
// #else /* use MOD 2 */
// 		int mod_key = input_key % nbSets;
// 		int target_set = (mod_key / 2 + (mod_key % 2) * (nbSets / 2)) % nbSets;
// #endif
		int mod_key = input_key % nbSets;
		// int mod_key = (input_key>>4) % nbSets;
		// int target_set = (mod_key / 2 + (mod_key % 2) * (nbSets / 2)) % nbSets;
		int target_set = mod_key;

		int thread_pos = target_set*nbWays + wayId;

		thread_key = keys[thread_pos];
		// thread_val = values[thread_pos]; // assume read-before-write
		thread_state = state[thread_pos];
		thread_ts = max(ts_GPU[thread_pos], ts_CPU[thread_pos]); // TODO: only needed for evict

		// TODO: divergency here
		thread_is_found = (thread_key == input_key && (thread_state & MEMCD_VALID));
		thread_is_empty = !(thread_state & MEMCD_VALID);
		int empty_min_id = thread_is_empty ? id : id + 32; // warpSize == 32
		int min_ts = thread_ts;

		int warp_is_found = thread_is_found; // 1 someone has found; 0 no one found
		int warp_is_empty = thread_is_empty; // 1 someone has empty; 0 no empties
		int mask = nbWays > 32 ? FULL_MASK : ((1 << nbWays) - 1) << (warpSliceID*nbWays);

		for (int offset = max(nbWays, 32)/2; offset > 0; offset /= 2) {
			warp_is_found = max(warp_is_found, __shfl_xor_sync(mask, warp_is_found, offset));
			warp_is_empty = max(warp_is_empty, __shfl_xor_sync(mask, warp_is_empty, offset));
			empty_min_id = min(empty_min_id, __shfl_xor_sync(mask, empty_min_id, offset));
			min_ts = min(min_ts, __shfl_xor_sync(mask, min_ts, offset));
		}

		reduction_is_found[reductionID] = warp_is_found;
		reduction_is_empty[reductionID] = warp_is_empty;
		reduction_empty_min_id[reductionID] = empty_min_id;
		reduction_min_ts[reductionID] = min_ts;

		// STEP: for n-way > 32 go to shared memory and try again
		warp_is_found = reduction_is_found[wayId % reductionSize];
		warp_is_empty = reduction_is_empty[wayId % reductionSize];
		empty_min_id = reduction_empty_min_id[wayId % reductionSize];
		min_ts = reduction_min_ts[wayId % reductionSize];

		for (int offset = reductionSize/2; offset > 0; offset /= 2) {
			warp_is_found = max(warp_is_found, __shfl_xor_sync(mask, warp_is_found, offset));
			warp_is_empty = max(warp_is_empty, __shfl_xor_sync(mask, warp_is_empty, offset));
			empty_min_id = min(empty_min_id, __shfl_xor_sync(mask, empty_min_id, offset));
			min_ts = min(min_ts, __shfl_xor_sync(mask, min_ts, offset));
		}

				// if (maxRetries == 8191) {
				// 	 printf("thr%i retry 8191 times for key%i thread_pos=%i check_key=%i \n",
				// 		id, input_key, thread_pos, checkKey);
				// }

		if (thread_is_found) {
			int nbRetries = 0; //TODO: on fail should repeat the search
			PR_txBegin(); // TODO: I think this may not work
			if (nbRetries > PR_maxNbRetries) {
				// TODO: is ignoring the input
				// someone got it; need to find a new spot for the key
				failed_to_insert[warpSliceID] = 1;
				break;
			}
			nbRetries++;

			checkKey = PR_read(&keys[thread_pos]); // read-before-write
			/*checkKey1 = */PR_read(&keys[thread_pos]); // read-before-write
			/*checkKey2 = */PR_read(&keys[thread_pos+sizeCache]); // read-before-write
			/*checkKey3 = */PR_read(&keys[thread_pos+2*sizeCache]); // read-before-write
			// TODO: does not work
			// if (checkKey != input_key || checkKey1 != input_key
			// 		|| checkKey2 != input_key || checkKey3 != input_key) { // we are late
			// 	failed_to_insert[warpSliceID] = 1;
			// 	break;
			// }
			PR_read(&values[thread_pos]); // read-before-write
			PR_read(&ts_GPU[thread_pos]); // read-before-write
			// TODO: check if values changed: if yes abort
			PR_write(&keys[thread_pos], input_key);
			PR_write(&extraKeys[thread_pos], input_key);
			PR_write(&extraKeys[thread_pos+sizeCache], input_key);
			PR_write(&extraKeys[thread_pos+2*sizeCache], input_key);
			PR_write(&values[thread_pos], input_val);
			PR_write(&extraVals[thread_pos], input_val);
			PR_write(&extraVals[thread_pos+sizeCache], input_val);
			PR_write(&extraVals[thread_pos+2*sizeCache], input_val);
			PR_write(&extraVals[thread_pos+3*sizeCache], input_val);
			PR_write(&extraVals[thread_pos+4*sizeCache], input_val);
			PR_write(&extraVals[thread_pos+5*sizeCache], input_val);
			PR_write(&extraVals[thread_pos+6*sizeCache], input_val);
			PR_write(&ts_GPU[thread_pos], curr_clock);
			PR_read(&setUsage[target_set]); // locks this set (if the CPU tries to modify)
			// TODO: it seems not to reach this if but the nbRetries is needed
						// if (nbRetries == 8191) printf("thr%i aborted 8191 times for key%i thread_pos=%i rsetSize=%lu, wsetSize=%lu\n",
						// 	id, input_key, thread_pos, pr_args.rset.size, pr_args.wset.size);
			PR_txCommit();
			out[targetKey + i].isFound = 1;
			out[targetKey + i].value = checkKey;
		}

		// if(id == 0) printf("is found=%i\n", thread_is_found);

		// TODO: if num_ways > 32 this does not work very well... (must use shared memory)
		//       using shared memory --> each warp compute the min then: min(ResW1, ResW2)
		//       ResW1 and ResW2 are shared
		// was it found?
		if (!warp_is_found && thread_is_empty && empty_min_id == id) {
			// the low id thread must be the one that writes
			int nbRetries = 0;  //TODO: on fail should repeat the search
			PR_txBegin(); // TODO: I think this may not work
			if (nbRetries > PR_maxNbRetries) {
				// someone got it; need to find a new spot for the key
				failed_to_insert[warpSliceID] = 1;
				break;
			}
			nbRetries++;

			checkKey = PR_read(&keys[thread_pos]); // read-before-write
			/*checkKey1 = */PR_read(&keys[thread_pos]); // read-before-write
			/*checkKey2 = */PR_read(&keys[thread_pos+sizeCache]); // read-before-write
			/*checkKey3 = */PR_read(&keys[thread_pos+2*sizeCache]); // read-before-write
			// TODO: does not work
			// if (checkKey != input_key || checkKey1 != input_key
			// 		|| checkKey2 != input_key || checkKey3 != input_key) { // we are late
			// 	failed_to_insert[warpSliceID] = 1;
			// 	break;
			// }
			PR_read(&values[thread_pos]); // read-before-write
			PR_read(&ts_GPU[thread_pos]); // read-before-write
			PR_read(&state[thread_pos]); // read-before-write
			// TODO: check if values changed: if yes abort
			PR_write(&keys[thread_pos], input_key);
			PR_write(&extraKeys[thread_pos], input_key);
			PR_write(&extraKeys[thread_pos+sizeCache], input_key);
			PR_write(&extraKeys[thread_pos+2*sizeCache], input_key);
			PR_write(&values[thread_pos], input_val);
			PR_write(&extraVals[thread_pos], input_val);
			PR_write(&extraVals[thread_pos+sizeCache], input_val);
			PR_write(&extraVals[thread_pos+2*sizeCache], input_val);
			PR_write(&extraVals[thread_pos+3*sizeCache], input_val);
			PR_write(&extraVals[thread_pos+4*sizeCache], input_val);
			PR_write(&extraVals[thread_pos+5*sizeCache], input_val);
			PR_write(&extraVals[thread_pos+6*sizeCache], input_val);
			PR_write(&ts_GPU[thread_pos], curr_clock);
			int newState = MEMCD_VALID|MEMCD_WRITTEN;
			PR_write(&state[thread_pos], newState);
			PR_txCommit();
			out[targetKey + i].isFound = 0;
			out[targetKey + i].value = checkKey;
		}

		// not found, none empty --> evict the oldest
		if (!warp_is_found && !warp_is_empty && min_ts == thread_ts) {
			int nbRetries = 0; //TODO: on fail should repeat the search
			PR_txBegin(); // TODO: I think this may not work
			if (nbRetries > PR_maxNbRetries) {
		 		// someone got it; need to find a new spot for the key
				failed_to_insert[warpSliceID] = 1;
				break;
			}
			nbRetries++;

			checkKey = PR_read(&keys[thread_pos]); // read-before-write
			// TODO: does not work
			// if (checkKey != input_key) { // we are late
			// 	failed_to_insert[warpSliceID] = 1;
			// 	break;
			// }
			PR_read(&values[thread_pos]); // read-before-write
			PR_read(&ts_GPU[thread_pos]); // read-before-write
			// TODO: check if values changed: if yes abort
			PR_write(&keys[thread_pos], input_key);
			PR_write(&extraKeys[thread_pos], input_key);
			PR_write(&extraKeys[thread_pos+sizeCache], input_key);
			PR_write(&extraKeys[thread_pos+2*sizeCache], input_key);
			PR_write(&values[thread_pos], input_val);
			PR_write(&extraVals[thread_pos], input_val);
			PR_write(&extraVals[thread_pos+sizeCache], input_val);
			PR_write(&extraVals[thread_pos+2*sizeCache], input_val);
			PR_write(&extraVals[thread_pos+3*sizeCache], input_val);
			PR_write(&extraVals[thread_pos+4*sizeCache], input_val);
			PR_write(&extraVals[thread_pos+5*sizeCache], input_val);
			PR_write(&extraVals[thread_pos+6*sizeCache], input_val);
			PR_write(&ts_GPU[thread_pos], curr_clock);
			// if (nbRetries == 8191) printf("thr%i aborted 8191 times for key%i thread_pos=%i rsetSize=%lu, wsetSize=%lu\n",
			// 	id, input_key, thread_pos, pr_args.rset.size, pr_args.wset.size);
			PR_txCommit();
			out[targetKey + i].isFound = 0;
			out[targetKey + i].value = checkKey;
		}
	}

	PR_exitKernel();
}
// ----------------------

/****************************************************************************
 *	FUNCTIONS
/****************************************************************************/

extern "C"
cuda_config cuda_configInit(long size, int trans, int hash, int tx, int bl, int hprob, float hmult)
{
	cuda_config c;

	c.size = size;
	c.TransEachThread = trans > 0 ? trans : ( DEFAULT_TransEachThread << 1) / BANK_NB_TRANSFERS;
	c.hashNum = hash > 0 ? hash : DEFAULT_hashNum;
	c.threadNum = tx > 0 ? tx : DEFAULT_threadNum;
	c.blockNum = bl > 0 ? bl : DEFAULT_blockNum;
	c.hprob = hprob > 0 ? hprob : DEFAULT_HPROB;
	c.hmult = hmult > 0 ? hmult : DEFAULT_HMULT;
	// c.BANK_NB_TRANSFERS = (BANK_NB_TRANSFERS > 1 && ((BANK_NB_TRANSFERS & 1) == 0)) ? BANK_NB_TRANSFERS : 2; // DEFAULT_BANK_NB_TRANSFERS

	return c;
}

extern "C"
cudaError_t cuda_configCpyMemcd(cuda_t *c)
{
	cudaError_t cudaStatus;
	int err = 1;

	while (err) {
		err = 0;
		cudaStatus = cudaMemcpyToSymbol(num_ways, &c->num_ways, sizeof(int), 0, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			printf("cudaMemcpy to device failed for num_ways!");
			continue;
		}
		cudaStatus = cudaMemcpyToSymbol(num_sets, &c->num_sets, sizeof(int), 0, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			printf("cudaMemcpy to device failed for num_sets!");
			continue;
		}
	}
	return cudaStatus;
}

extern "C"
void cuda_configCpy(cuda_config c)
{
	// TODO: recycled as prec read only
	CUDA_CHECK_ERROR(cudaMemcpyToSymbol(devParsedData, &parsedData, sizeof(thread_data_t), 0, cudaMemcpyHostToDevice), "");

	CUDA_CHECK_ERROR(cudaMemcpyToSymbol(BANK_SIZE, &c.size, sizeof(long), 0, cudaMemcpyHostToDevice ), "");
	CUDA_CHECK_ERROR(cudaMemcpyToSymbol(HASH_NUM, &c.hashNum, sizeof(int), 0, cudaMemcpyHostToDevice ), "");
	CUDA_CHECK_ERROR(cudaMemcpyToSymbol(txsInKernel, &c.TransEachThread, sizeof(int), 0, cudaMemcpyHostToDevice), "");
	CUDA_CHECK_ERROR(cudaMemcpyToSymbol(txsPerGPUThread, &c.TransEachThread, sizeof(int), 0, cudaMemcpyHostToDevice), "");
	CUDA_CHECK_ERROR(cudaMemcpyToSymbol(hprob, &c.hprob, sizeof(int), 0, cudaMemcpyHostToDevice), "");
	CUDA_CHECK_ERROR(cudaMemcpyToSymbol(hmult, &c.hmult, sizeof(float), 0, cudaMemcpyHostToDevice), "");
	CUDA_CHECK_ERROR(cudaMemcpyToSymbol(prec_read_intensive, &parsedData.nb_read_intensive, sizeof(int), 0, cudaMemcpyHostToDevice), "");
	CUDA_CHECK_ERROR(cudaMemcpyToSymbol(read_intensive_size, &parsedData.read_intensive_size, sizeof(int), 0, cudaMemcpyHostToDevice), "");
	// TODO: find a way of passing BANK_NB_TRANSFERS
};
