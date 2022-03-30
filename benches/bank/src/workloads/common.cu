#include "all_bank_parts.cuh"
#include "bankKernel.cuh"
#include "pr-stm.cuh"

#define INTEREST_RATE 0.5 // bank readIntensive transaction
#define FULL_MASK 0xffffffff

thread_data_t parsedData;
int isInterBatch = 0;

unsigned long long input_seed;
int bank_cpu_sample_data;

// memman_s GPU_input_buffer_good;
// memman_s GPU_input_buffer_bad;
// memman_s GPU_input_buffer;
// memman_s GPU_output_buffer;

memman::MemObjOnDev GPU_input_buffer_good;
memman::MemObjOnDev GPU_input_buffer_bad;
memman::MemObjOnDev GPU_input_buffer;
memman::MemObjOnDev GPU_output_buffer;

FILE *GPU_input_file;
FILE *CPU_input_file;

int *GPUoutputBuffer[HETM_NB_DEVICES];
int *CPUoutputBuffer;
int *GPUInputBuffer[HETM_NB_DEVICES];
int *CPUInputBuffer;

size_t size_of_GPU_input_buffer;
size_t size_of_CPU_input_buffer;

size_t currMaxCPUoutputBufferSize;
size_t maxGPUoutputBufferSize;


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

	// HeTM_GPU_log_s *GPU_log = (HeTM_GPU_log_s*)args.pr_args_ext;
	// long *state = GPU_log->state;
	// volatile int seedState;

	// TODO: new parameter for the spins
	// for (i = 0; i < devParsedData.GPU_backoff; i++) {
	// 	state[id] += id + 1234 * i;
	// }

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
	// int nbRetries = 0;
	PR_txBegin();
	// if (nbRetries++ > PR_maxNbRetries) break;
#endif /* BANK_DISABLE_PRSTM */

	// reads the accounts first, then mutates the locations
	float resF = 0;
	int resI;
	for (j = 0; j < read_intensive_size; ++j) { // saves 1 slot for the write
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

	// __syncthreads();
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

__device__ void update_tx2(PR_txCallDefArgs, int txCount)
{
	int id = threadIdx.x+blockDim.x*blockIdx.x;
	volatile int i = 0;
	double count_amount = 0;
	HeTM_bankTx_input_s *input = (HeTM_bankTx_input_s*)args.inBuf;
	PR_GRANULE_T *accounts = (PR_GRANULE_T*)input->accounts;
	int isInter = input->is_intersection;
	size_t nbAccounts = input->nbAccounts;
	int *input_buffer = input->input_buffer;
	int *output_buffer = input->output_buffer;
	int option = PR_rand(INT_MAX);
	int tot_nb_threads = blockDim.x*gridDim.x;

	unsigned randNum;
	unsigned accountIdx;

	HeTM_GPU_log_s *GPU_log = (HeTM_GPU_log_s*)args.pr_args_ext;
	long *state = GPU_log->state;
	int devId = GPU_log->devId;
	volatile int seedState;

	// TODO: new parameter for the spins
	// for (i = 0; i < devParsedData.GPU_backoff; i++) {
	// 	state[id] += id + 1234 * i;
	// }

	// volatile int nbRetries = 0;
	PR_txBegin();
	// if (nbRetries++ > PR_maxNbRetries) break;

	seedState = state[id];
	int loopFor = read_intensive_size;

#if BANK_PART == 10
	loopFor *= HETM_BANK_PART_SCALE;
#endif

	// #pragma unroll
	for (i = 0; i < loopFor; i++) {
		randNum = PR_rand(INT_MAX);
		if (!isInter) {
			accountIdx = GPU_ACCESS(devId, randNum, nbAccounts);
		} else {
#if BANK_PART == 9
			// deterministic abort
			accountIdx = /*(i == 0) ? id : */INTERSECT_ACCESS_GPU(devId, randNum, nbAccounts);
#else
			accountIdx = INTERSECT_ACCESS_GPU(devId, randNum, nbAccounts);
#endif /* BANK_PART == 9 */
		}
		// printf("randNum = %u\n", randNum);
		count_amount += PR_read(accounts + accountIdx);
	}

	state[id] = seedState;

	// #pragma unroll
	for (i = 0; i < read_intensive_size; i++) {
		randNum = PR_rand(INT_MAX); // 56; //
		if (!isInter) {
			accountIdx = GPU_ACCESS(devId, randNum, nbAccounts);
		} else {
#if BANK_PART == 9
			// deterministic abort
			printf("passou aqui!\n");
			accountIdx = /*(i == 0) ? id : */INTERSECT_ACCESS_GPU(devId, randNum, nbAccounts);
#else
			accountIdx = INTERSECT_ACCESS_GPU(devId, randNum, nbAccounts);
#endif /* BANK_PART == 9 */
		}
		// printf("randNum = %u\n", randNum);
		// PR_read(accounts + accountIdx);
		PR_write(accounts + accountIdx, count_amount * input_buffer[id+txCount*tot_nb_threads]);
	}

	PR_txCommit();

	// printf("output_buffer[id] (%p) \n", &(output_buffer[id]));
	// printf("output_buffer[id] = %i \n", output_buffer[id]);

	output_buffer[id] = *(accounts + accountIdx);
}

__device__ void readOnly_tx2(PR_txCallDefArgs, int txCount)
{
	int id = threadIdx.x+blockDim.x*blockIdx.x;
	int i = 0; //how many transactions one thread need to commit
	double count_amount = 0;
	HeTM_bankTx_input_s *input = (HeTM_bankTx_input_s*)args.inBuf;
	PR_GRANULE_T *accounts = (PR_GRANULE_T*)input->accounts;
	int isInter = input->is_intersection;
	size_t nbAccounts = input->nbAccounts;
	int *output_buffer = input->output_buffer;
	int option = PR_rand(INT_MAX);
	int loopFor = read_intensive_size;

	unsigned randNum;
	unsigned accountIdx;

	HeTM_GPU_log_s *GPU_log = (HeTM_GPU_log_s*)args.pr_args_ext;
	int devId = GPU_log->devId;
	// long *state = GPU_log->state;
	// volatile int seedState;

	// TODO: new parameter for the spins
	// for (i = 0; i < devParsedData.GPU_backoff; i++) {
	// 	state[id] += id + 1234 * i;
	// }

	// int nbRetries = 0;
#if BANK_PART == 10
	loopFor *= HETM_BANK_PART_SCALE;
#endif
	PR_txBegin();
	// if (nbRetries++ > PR_maxNbRetries) break;

	#pragma unroll
	for (i = 0; i < loopFor; i++) {
		randNum = PR_rand(INT_MAX);
		if (!isInter) {
			accountIdx = GPU_ACCESS(devId, randNum, nbAccounts);
		} else {
#if BANK_PART == 9
			// deterministic abort
			accountIdx = /*(i == 0) ? id : */INTERSECT_ACCESS_GPU(devId, randNum, nbAccounts);
#else
			accountIdx = INTERSECT_ACCESS_GPU(devId, randNum, nbAccounts);
#endif /* BANK_PART == 9 */
		}
		// printf("randNum = %u\n", randNum);
		count_amount += PR_read(accounts + accountIdx);
	}
	PR_txCommit();

	// printf("output_buffer[id] (%p) \n", &(output_buffer[id]));
	// printf("output_buffer[id] = %i \n", output_buffer[id]);
	output_buffer[id] = count_amount;
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

	// HeTM_GPU_log_s *GPU_log = (HeTM_GPU_log_s*)args.pr_args_ext;
	// long *state = GPU_log->state;
	// volatile int seedState;

	// TODO: new parameter for the spins
	// for (i = 0; i < devParsedData.GPU_backoff; i++) {
	// 	state[id] += id + 1234 * i;
	// }

	idx[0] = (input_buffer[id+txCount*tot_nb_threads]) % nbAccounts;
	#pragma unroll
	for (i = 1; i < max_size_of_reads; i++) {
		idx[i] = (idx[i-1] + 2) % nbAccounts; // access even accounts
	}

	// TODO:
	output_buffer[id] = 1;

#ifndef BANK_DISABLE_PRSTM
	// int nbRetries = 0;
	// printf("  [ IN KERNEL ] PR_txBegin %i \n", id);
	PR_txBegin();
	// if (nbRetries++ > PR_maxNbRetries) break;
#endif /* BANK_DISABLE_PRSTM */

	// reads the accounts first, then mutates the locations
	float resF = 0;
	// int resI;
	for (j = 0; j < max_size_of_reads; ++j) { // saves 1 slot for the write
		if (idx[j] < 0 || idx[j] >= nbAccounts) {
			break;
		}
#ifndef BANK_DISABLE_PRSTM
		// int targetAccount = idx[j] % (nbAccounts / devParsedData.access_controller);
		resF += (float)PR_read(accounts + idx[j]) * INTEREST_RATE;
		// resF = cos(resF);
#else /* PR-STM disabled */
		resF += accounts[idx[j]] * INTEREST_RATE;
#endif
	}
	resF += 1.0; // adds at least 1
	// resI = (int)resF;

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
// printf("  [ IN KERNEL ] PR_txCommit %i \n", id);
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
	// int access_controller = devParsedData.access_controller;


	unsigned writeAccountIdx;
	// unsigned writeAccountIdx2;
	unsigned readAccountIdx;

	// HeTM_GPU_log_s *GPU_log = (HeTM_GPU_log_s*)args.pr_args_ext;
	// long *state = GPU_log->state;
	// int devId = GPU_log->devId;
	// volatile int seedState;

	// TODO: new parameter for the spins
	// for (i = 0; i < devParsedData.GPU_backoff; i++) {
	// 	state[id] += id + 1234 * i;
	// }

	// before the transaction pick the accounts (needs PR-STM stuff)
	// random_Kernel(PR_txCallArgs, idx, nbAccounts, is_intersection); //get random index
	__syncthreads();
	idx[0] = input_buffer[id+txCount*tot_nb_threads];

	#pragma unroll
	for (i = 1; i < max_size_of_reads; i++) {
		// idx[i] = (idx[i-1] + 2) % nbAccounts; // access even accounts
		idx[i] = (idx[i-1] + 1) % nbAccounts; // access even accounts
	}
	output_buffer[id] = 0;

	// printf("[%i]update_tx idx = %i\n", id, idx[0]);
#ifndef BANK_DISABLE_PRSTM
	// int nbRetries = 0;
	PR_txBegin();
	// if (nbRetries++ > PR_maxNbRetries) break;
#endif

	// TODO: store must be controlled with parsedData.access_controller
	// -----------------

	target = 0;
	// int writeAccountIdx = idx[target];
	// writeAccountIdx = (unsigned)((double)idx[target] / (double)access_controller);
	writeAccountIdx = (unsigned)idx[target];
	// writeAccountIdx2 = (unsigned)((double)idx[target+1] / (double)access_controller);

	// must read the write
	#ifndef BANK_DISABLE_PRSTM
			count_amount += PR_read(&(accounts[writeAccountIdx]));
	#else /* PR-STM disabled */
			count_amount += accounts[writeAccountIdx];
	#endif

	// reads the accounts first, then mutates the locations
	#pragma unroll
	for (j = 0; j < read_intensive_size; j++)
	{
		readAccountIdx = idx[j];
#ifndef BANK_DISABLE_PRSTM
		count_amount += PR_read(&(accounts[readAccountIdx]));
#else /* PR-STM disabled */
		count_amount += accounts[readAccountIdx];
#endif
	}

	// nval = COMPUTE_TRANSFER(reads[target] - 1); // -money
	nval = COMPUTE_TRANSFER(count_amount - 1); // -money

	// nbAccounts / devParsedData.access_controller
#ifndef BANK_DISABLE_PRSTM
// printf("[%i] writes pos %i\n", id, writeAccountIdx);

	PR_write(&(accounts[writeAccountIdx]), nval); //write changes to write set
	// printf("  [ IN KERNEL ] WRITE id = %i devId = %i nbRetries = %i\n", id, args.devId, nbRetries);

	// printf("[devId=%i,tx=%i] idx=%u\n", devId, id, writeAccountIdx);
#else /* PR-STM disabled */
	accounts[writeAccountIdx] = nval; //write changes to write set
#endif

#ifndef BANK_DISABLE_PRSTM
	PR_txCommit();
#else
	if (args.nbCommits != NULL) args.nbCommits[PR_THREAD_IDX]++;
#endif

	// printf("[%i]  COMMIT  update_tx idx = %i\n", id, idx[0]);
}

__device__ void update_tx_simple(PR_txCallDefArgs, int txCount)
{
	const int amount_to_transfer = 1;
	int id = threadIdx.x+blockDim.x*blockIdx.x;
	int tot_nb_threads = blockDim.x*gridDim.x;
	int idx[2];

	HeTM_bankTx_input_s *input = (HeTM_bankTx_input_s*)args.inBuf;
	PR_GRANULE_T *accounts = (PR_GRANULE_T*)input->accounts;
	int *input_buffer = input->input_buffer;

	idx[0] = input_buffer[id+txCount*tot_nb_threads];
	idx[1] = idx[0]+1;

	PR_txBegin();
	int r1 = PR_read(accounts + idx[0]);
	int r2 = PR_read(accounts + idx[1]);

	PR_write(accounts + idx[0], r1-amount_to_transfer);
	PR_write(accounts + idx[1], r2+amount_to_transfer);
	PR_txCommit();
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

	// HeTM_GPU_log_s *GPU_log = (HeTM_GPU_log_s*)args.pr_args_ext;
	// long *state = GPU_log->state;
	// volatile int seedState;

	// TODO: new parameter for the spins
	// for (i = 0; i < devParsedData.GPU_backoff; i++) {
	// 	state[id] += id + 1234 * i;
	// }

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
	// int nbRetries = 0;
	PR_txBegin();
	// if (nbRetries++ > PR_maxNbRetries) break;
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





#ifndef HETM_BANK_PART_SCALE
#define HETM_BANK_PART_SCALE 10
#endif /* HETM_BANK_PART_SCALE */

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
	int targetKey = id / (num_ways + devParsedData.trans); // id of the key that each thread will take

	// num_ways threads will colaborate for the same input
	// REQUIREMENT: 1 block >= num_ways

	HeTM_memcdTx_input_s *input = (HeTM_memcdTx_input_s*)args.inBuf;
	PR_GRANULE_T       *keys = (PR_GRANULE_T*)input->key;
	PR_GRANULE_T     *values = (PR_GRANULE_T*)input->val;
	PR_GRANULE_T *timestamps = (PR_GRANULE_T*)input->ts;
	PR_GRANULE_T      *state = (PR_GRANULE_T*)input->state;

	// TODO: out is NULL
	memcd_get_output_t  *out = (memcd_get_output_t*)input->output;
	int           curr_clock = *((int*)input->curr_clock);
	int          *input_keys = (int*)input->input_keys;

	int nbSets = input->nbSets;
	int nbWays = input->nbWays;

	// __shared__ int aborted[1024];
	// if (wayId == 0) aborted[targetKey] = 0;
	// __syncthreads();

	for (int i = 0; i < devParsedData.trans; ++i) { // num_ways keys
		out[id*devParsedData.trans + i].isFound = 0;
	}

	for (int i = 0; i < (nbWays + devParsedData.trans); ++i) { // num_ways keys
		// TODO: for some reason input_key == 0 does not work --> PR-STM loops forever
		int input_key = input_keys[targetKey + i]; // input size is well defined
		int target_set = input_key % nbSets;
		int thread_pos = target_set*nbWays + wayId;
		int thread_is_found;
		PR_GRANULE_T thread_key;
		PR_GRANULE_T thread_val;
		PR_GRANULE_T thread_state;

		// PR_write(&timestamps[thread_pos], curr_clock);
		thread_key = keys[thread_pos];
		thread_state = state[thread_pos];

		__syncthreads(); // TODO: each num_ways thread helps on processing the targetKey

		// TODO: divergency here
		thread_is_found = (thread_key == input_key && ((thread_state & MEMCD_VALID) != 0));
		if (thread_is_found) {
			// int nbRetries = 0;
			int ts;
			PR_txBegin();
			// if (nbRetries > 1024) {
			// 	break; // retry, key changed
			// }
			// nbRetries++;

			PR_read(&keys[thread_pos]);
			// // TODO:
			// if (thread_key != PR_read(&keys[thread_pos])) {
			// 	i--; // retry, key changed
			// 	break;
			// }
			thread_val = PR_read(&values[thread_pos]);
			ts = PR_read(&timestamps[thread_pos]); // assume read-before-write
			if (ts < curr_clock /*&& nbRetries < 5*/) PR_write(&timestamps[thread_pos], curr_clock);
			else timestamps[thread_pos] = curr_clock; // TODO: cannot transactionally write this...

			PR_txCommit();

			out[targetKey + i].isFound = 1;
			out[targetKey + i].value = thread_val;
		}
		// if (aborted[targetKey]) {
		// 	// i--; // repeat this loop // TODO: blocks forever
		// }
		// aborted[targetKey] = 0;
		// __syncthreads();
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

	HeTM_memcdTx_input_s *input = (HeTM_memcdTx_input_s*)args.inBuf;
	memcd_get_output_t  *out = (memcd_get_output_t*)input->output;
	PR_GRANULE_T       *keys = (PR_GRANULE_T*)input->key;
	PR_GRANULE_T     *values = (PR_GRANULE_T*)input->val;
	PR_GRANULE_T *timestamps = (PR_GRANULE_T*)input->ts;
	PR_GRANULE_T      *state = (PR_GRANULE_T*)input->state;
	int           curr_clock = *((int*)input->curr_clock);
	int          *input_keys = (int*)input->input_keys;
	int          *input_vals = (int*)input->input_vals;

	int nbSets = (int)input->nbSets;
	int nbWays = (int)input->nbWays;

	int thread_is_found; // TODO: use shuffle instead
	int thread_is_empty; // TODO: use shuffle instead
	// int thread_is_older; // TODO: use shuffle instead
	PR_GRANULE_T thread_key;
	// PR_GRANULE_T thread_val;
	PR_GRANULE_T thread_ts;
	PR_GRANULE_T thread_state;

	int checkKey;
	int maxRetries = 0;

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
		int target_set = input_key % nbSets;
		int thread_pos = target_set*nbWays + wayId;

		thread_key = keys[thread_pos];
		// thread_val = values[thread_pos]; // assume read-before-write
		thread_state = state[thread_pos];
		thread_ts = timestamps[thread_pos]; // TODO: only needed for evict

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
			// int nbRetries = 0; //TODO: on fail should repeat the search
			PR_txBegin(); // TODO: I think this may not work
			// if (nbRetries > 0) {
			// 	// TODO: is ignoring the input
			// 	// someone got it; need to find a new spot for the key
			// 	failed_to_insert[warpSliceID] = 1;
			// 	break;
			// }
			// nbRetries++;

			checkKey = PR_read(&keys[thread_pos]); // read-before-write
			// TODO: does not work
			// if (checkKey != input_key) { // we are late
			// 	failed_to_insert[warpSliceID] = 1;
			// 	break;
			// }
			PR_read(&values[thread_pos]); // read-before-write
			PR_read(&timestamps[thread_pos]); // read-before-write
			// TODO: check if values changed: if yes abort
			PR_write(&keys[thread_pos], input_key);
			PR_write(&values[thread_pos], input_val);
			PR_write(&timestamps[thread_pos], curr_clock);
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
			// int nbRetries = 0;  //TODO: on fail should repeat the search
			PR_txBegin(); // TODO: I think this may not work
			// if (nbRetries > 0) {
			// 	// someone got it; need to find a new spot for the key
			// 	failed_to_insert[warpSliceID] = 1;
			// 	break;
			// }
			// nbRetries++;

			checkKey = PR_read(&keys[thread_pos]); // read-before-write
			// TODO: does not work
			// if (checkKey != input_key) { // we are late
			// 	failed_to_insert[warpSliceID] = 1;
			// 	break;
			// }
			PR_read(&values[thread_pos]); // read-before-write
			PR_read(&timestamps[thread_pos]); // read-before-write
			PR_read(&state[thread_pos]); // read-before-write
			// TODO: check if values changed: if yes abort
			PR_write(&keys[thread_pos], input_key);
			PR_write(&values[thread_pos], input_val);
			PR_write(&timestamps[thread_pos], curr_clock);
			int newState = MEMCD_VALID|MEMCD_WRITTEN;
			PR_write(&state[thread_pos], newState);
			PR_txCommit();
			out[targetKey + i].isFound = 0;
			out[targetKey + i].value = checkKey;
		}

		// not found, none empty --> evict the oldest
		if (!warp_is_found && !warp_is_empty && min_ts == thread_ts) {
			// int nbRetries = 0; //TODO: on fail should repeat the search
			PR_txBegin(); // TODO: I think this may not work
			// if (nbRetries > 0) {
		 	// 	// someone got it; need to find a new spot for the key
			// 	failed_to_insert[warpSliceID] = 1;
			// 	break;
			// }
			// nbRetries++;

			checkKey = PR_read(&keys[thread_pos]); // read-before-write
			// TODO: does not work
			// if (checkKey != input_key) { // we are late
			// 	failed_to_insert[warpSliceID] = 1;
			// 	break;
			// }
			PR_read(&values[thread_pos]); // read-before-write
			PR_read(&timestamps[thread_pos]); // read-before-write
			// TODO: check if values changed: if yes abort
			PR_write(&keys[thread_pos], input_key);
			PR_write(&values[thread_pos], input_val);
			PR_write(&timestamps[thread_pos], curr_clock);
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

