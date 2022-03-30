#include "murmurhash2.cuh"

#include <cuda_runtime.h>
#include "helper_timer.h"
#include "rnd.h"

// TODO: check write kernel max size
// #define PR_MAX_RWSET_SIZE   0x40

#include "pr-stm.cuh"

#include "zipf_dist.h"

#include <iostream>
#include <fstream>
#include <cstdio>
#include <ctime>


PR_DEVICE void PR_beforeKernel_EXT(PR_txCallDefArgs) { }
PR_DEVICE void PR_afterKernel_EXT (PR_txCallDefArgs) { }
PR_DEVICE void PR_beforeBegin_EXT (PR_txCallDefArgs) { }
PR_DEVICE void PR_afterCommit_EXT (PR_txCallDefArgs) { }
PR_DEVICE void PR_after_val_locks_EXT (PR_args_s *args) { }
PR_DEVICE void PR_after_writeback_EXT (PR_args_s *args, int i, PR_GRANULE_T *addr, PR_GRANULE_T val) { }



#define MEMCD_NB_SETS      4096
#define MEMCD_NB_WAYS         8
#define MEMCD_PAYLOAD         8
#define MEMCD_PAYLOAD_KEY     4
#define MEMCD_CONFL_SPACE    (MEMCD_NB_WAYS*MEMCD_NB_SETS)
#define NB_OF_GPU_BUFFERS     4
#define TXS_EACH_THREAD       1  // how many transactions each thread do
#define NB_ITERATIONS         1  // loop how many times

using namespace std;

enum { // state of a memcd cache entry
  MEMCD_INV     =0,
  MEMCD_VALID   =1,
  MEMCD_READ    =2,
  MEMCD_WRITTEN =4
};

typedef struct memcd_ {
	int devId;
	int *key;   /* keys in global memory */
	int *extraKey;
	int *val;   /* values in global memory */
	int *ts_CPU;    /* last access TS in global memory */
	int *ts_GPU;    /* last access TS in global memory */
	int *state; /* state in global memory */
	int *extraVals;
	unsigned *globalTs;
  long nbSets;
  int nbWays;
	int *inputs;
} memcd_t;

typedef struct {
	int devId;
	int64_t *state;
} memcd_pr_dev_buff_ext_t;

typedef struct {
	int devId;
	int64_t *state;
} memcd_pr_args_ext_t;

#define PR_rand(n) \
	PR_i_rand(args, n) \
//

#define NUMBER_OF_DEVICES 2

size_t size_of_GPU_input_buffer;
size_t maxGPUoutputBufferSize;
int *GPUoutputBuffer[NUMBER_OF_DEVICES];
int *GPUoutputBuffer_extraVals[NUMBER_OF_DEVICES];
int *GPUinputBuffer[NUMBER_OF_DEVICES];

__global__ void setupKernel(int64_t *state, int devId, memcd_t *appState);
__device__ unsigned PR_i_rand(pr_tx_args_dev_host_s args, unsigned n);

// These two are now __device__ (they were __global__)
__device__ void memcdWriteTx(PR_txCallDefArgs);
__device__ void memcdWriteTxSimple(PR_txCallDefArgs);
__device__ void memcdReadTx(PR_txCallDefArgs);
__global__ void memcdTx(PR_globalKernelArgs);

static void GPUbuffer_ZIPF(int devId);

__constant__ PR_DEVICE unsigned PR_seed = 1234; // TODO: set seed

// void impl_extra_pr_clbk_before_run_ext(pr_tx_args_s *args) {}

static void initResources(memcd_t*);

int main(int argc, char* argv[])
{
	PR_global_data_s *d;
	double avgKernelTime = 0.0;
	double avgAborts     = 0.0;
	double avgCommits    = 0.0;
	double totThroughput = 0.0;
	int nbOfBlocks, nbThrsPerBlock;
	memcd_t *a[NUMBER_OF_DEVICES];
	pr_buffer_s inBuf[NUMBER_OF_DEVICES];
	pr_buffer_s outBuf[NUMBER_OF_DEVICES];
	pr_tx_args_s prArgs[NUMBER_OF_DEVICES];
	memcd_pr_dev_buff_ext_t *cuRandBuf;

	if (argc == 3) { // TODO: more cache parameters
		for (int j = 0; j < NUMBER_OF_DEVICES; j++) {
			PR_curr_dev = j;
			d = &(PR_global[PR_curr_dev]);
			nbOfBlocks = d->PR_blockNum = atoi(argv[1]);
			nbThrsPerBlock = d->PR_threadNum = atoi(argv[2]) % 1025;
		}
	} else {
		printf("usage %s <number of blocks> <number of threads>\n", argv[0]);
		return -1;
	}

	for (int j = 0; j < 1; j++) {
		printf("Blocks: %d, Threads: %d\n", d->PR_blockNum, d->PR_threadNum);

		maxGPUoutputBufferSize = nbOfBlocks*nbThrsPerBlock*TXS_EACH_THREAD*sizeof(int);
		size_of_GPU_input_buffer = NB_OF_GPU_BUFFERS*maxGPUoutputBufferSize;

		for (int i = 0; i < NUMBER_OF_DEVICES; ++i) {
			PR_curr_dev = i;
			d = &(PR_global[PR_curr_dev]);

			int nbDevices, deviceId;
			PR_CHECK_CUDA_ERROR(cudaGetDeviceCount(&nbDevices), "");
			deviceId = PR_curr_dev % nbDevices;
			PR_CHECK_CUDA_ERROR(cudaSetDevice(deviceId), "");

			PR_CHECK_CUDA_ERROR(cudaMallocManaged(&(GPUoutputBuffer[i]), maxGPUoutputBufferSize
				+ MEMCD_PAYLOAD*maxGPUoutputBufferSize), "cudaMalloc");
			PR_CHECK_CUDA_ERROR(cudaMallocManaged(&(GPUinputBuffer[i]), size_of_GPU_input_buffer), "cudaMalloc");
			GPUoutputBuffer_extraVals[i] = GPUoutputBuffer[i] + maxGPUoutputBufferSize/sizeof(int);

			// TODO:
			PR_CHECK_CUDA_ERROR(cudaMallocManaged(&(a[i]), sizeof(memcd_t)), "cudaMalloc");
			GPUbuffer_ZIPF(i);
		}

		printf("MEMCD %dx%d slots(%d items) with %d threads!!\n",
			MEMCD_NB_SETS, MEMCD_NB_WAYS, MEMCD_CONFL_SPACE, d->PR_threadNum*d->PR_blockNum);
		/*int *state = new int[size];	//locks array
		printf("222!!\n");*/

		// Add vectors in parallel.
		// TODO: new interface

		extern void impl_pr_clbk_before_run_ext(pr_tx_args_s *args);
		extern void impl_pr_clbk_after_run_ext(pr_tx_args_s *args);
		extern void(*extra_pr_clbk_before_run_ext)(pr_tx_args_s *args); /* in aux.cu */
		extern int prstm_track_rwset_max_size;

		pr_clbk_before_run_ext = impl_pr_clbk_before_run_ext;
		pr_clbk_after_run_ext = impl_pr_clbk_after_run_ext;
		// extra_pr_clbk_before_run_ext = impl_extra_pr_clbk_before_run_ext;
		prstm_track_rwset_max_size = PRSTM_TRACK_RWSET_MAX_SIZE;

		// ------------------------- INIT EACH DEV
		for (int nbDevs = 0; nbDevs < NUMBER_OF_DEVICES; nbDevs++) {
			PR_curr_dev = nbDevs;
			d = &(PR_global[PR_curr_dev]);

			int nbDevices, deviceId;
			PR_CHECK_CUDA_ERROR(cudaGetDeviceCount(&nbDevices), "");
			deviceId = PR_curr_dev % nbDevices;
			PR_CHECK_CUDA_ERROR(cudaSetDevice(deviceId), "");

			PR_init((PR_init_params_s){
				.nbStreams = 2,
				.lockTableSize = PR_LOCK_TABLE_SIZE
			});

			PR_ALLOC(prArgs[PR_curr_dev].dev.pr_args_ext1, sizeof(memcd_pr_dev_buff_ext_t));
			prArgs[PR_curr_dev].host.pr_args_ext1 = malloc(sizeof(memcd_pr_dev_buff_ext_t));

			cuRandBuf = (memcd_pr_dev_buff_ext_t*)prArgs[PR_curr_dev].host.pr_args_ext1;
			cuRandBuf->devId = PR_curr_dev;
			PR_ALLOC(cuRandBuf->state, d->PR_blockNum * d->PR_threadNum * sizeof(int64_t));
			PR_CPY_TO_DEV(prArgs[PR_curr_dev].dev.pr_args_ext1, prArgs[PR_curr_dev].host.pr_args_ext1, sizeof(memcd_pr_dev_buff_ext_t));

			PR_createStatistics(&prArgs[PR_curr_dev]); // statistics on device 0 only

			CUDA_CHECK_ERROR(cudaFuncSetCacheConfig(setupKernel, cudaFuncCachePreferL1), "");

			initResources(a[nbDevs]);
			a[nbDevs]->devId = nbDevs;
			a[nbDevs]->inputs = GPUinputBuffer[nbDevs];
			inBuf[nbDevs].buf = a[nbDevs];
			inBuf[nbDevs].size = sizeof(memcd_t);
			outBuf[nbDevs].buf = (void*)(GPUoutputBuffer[nbDevs]);
			outBuf[nbDevs].size = maxGPUoutputBufferSize;
			// TODO: memset fails

			PR_prepareIO(&prArgs[nbDevs], inBuf[nbDevs], outBuf[nbDevs]);
			setupKernel<<< d->PR_blockNum, d->PR_threadNum >>>(cuRandBuf->state, cuRandBuf->devId, a[cuRandBuf->devId]);

			CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "");
		}
		// ------------------------- INIT EACH DEV

		for (int i = 0; i < NB_ITERATIONS; ++i) {

			for (int j = 0; j < NUMBER_OF_DEVICES; ++j) {
				PR_curr_dev = j;
				PR_run(memcdTx, &prArgs[PR_curr_dev]); // aborts on fail
			}

			// if (i % 4 == 0) {
			// 	PR_checkpointAbortsCommits();
			// 	printf("[it:%i] ---------- \n", i);
			// }

			for (int j = 0; j < NUMBER_OF_DEVICES; ++j) {
				PR_curr_dev = j;

				int nbDevices, deviceId;
				PR_CHECK_CUDA_ERROR(cudaGetDeviceCount(&nbDevices), "");
				deviceId = PR_curr_dev % nbDevices;
				PR_CHECK_CUDA_ERROR(cudaSetDevice(deviceId), "");

				PR_waitKernel(&prArgs[PR_curr_dev]);
				PR_useNextStream(&prArgs[PR_curr_dev]);
				CUDA_CHECK_ERROR(cudaStreamSynchronize(PR_getCurrentStream()), "");
			}
			// printf("PR_sumNbAborts             %12lli PR_sumNbCommits             %12lli\n",
			// 	*d->PR_sumNbAborts, *d->PR_sumNbCommits);
			// printf("PR_nbAbortsLastKernel      %12lli PR_nbCommitsLastKernel      %12lli\n",
			// 	d->PR_nbAbortsLastKernel, d->PR_nbCommitsLastKernel);
			// printf("PR_nbAbortsSinceCheckpoint %12lli PR_nbCommitsSinceCheckpoint %12lli\n\n",
			// 	d->PR_nbAbortsSinceCheckpoint, d->PR_nbCommitsSinceCheckpoint);
		}


		cudaEvent_t start, stop, stop2, copy;
		float elapsedMSKernel = 0, elapsedMSKernel2 = 0, elapsedMSCopy = 0;
		uint64_t *hostCount;
		uint64_t *devCount;
		size_t countSize = sizeof(uint64_t)*2;

		for (int nbDevs = 0; nbDevs < NUMBER_OF_DEVICES; nbDevs++) {
			PR_curr_dev = nbDevs;
			d = &(PR_global[PR_curr_dev]);

			int nbDevices, deviceId;
			PR_CHECK_CUDA_ERROR(cudaGetDeviceCount(&nbDevices), "");
			deviceId = PR_curr_dev % nbDevices;
			PR_CHECK_CUDA_ERROR(cudaSetDevice(deviceId), "");

			// must be done after selecting the device
			CUDA_CHECK_ERROR(cudaEventCreate(&start), "");
			CUDA_CHECK_ERROR(cudaEventCreate(&stop), "");
			CUDA_CHECK_ERROR(cudaEventCreate(&stop2), "");
			CUDA_CHECK_ERROR(cudaEventCreate(&copy), "");

			printf("\n\n\n -------------------- Device %i \n", PR_curr_dev);
			PR_retrieveIO(&prArgs[PR_curr_dev]); // updates PR_nbAborts and PR_nbCommits
			printf("[1] Aborts = %lli Commits = %lli\n", d->PR_nbAborts, d->PR_nbCommits);

			CUDA_CHECK_ERROR(cudaMallocHost(&hostCount, countSize), "");
			CUDA_CHECK_ERROR(cudaMalloc(&devCount, countSize), "");

			// [0] -> commits [1] -> aborts
			CUDA_CHECK_ERROR(cudaStreamSynchronize(PR_getCurrentStream()), "");
			CUDA_CHECK_ERROR(cudaEventRecord(start, PR_getCurrentStream()), "");
			PR_reduceCommitAborts<<<d->PR_blockNum, d->PR_threadNum, 0, PR_getCurrentStream()>>>
				(0, 0, prArgs[PR_curr_dev].dev, devCount, devCount + 1);
			CUDA_CHECK_ERROR(cudaEventRecord(stop, PR_getCurrentStream()), "");
			PR_reduceCommitAborts<<<d->PR_blockNum, d->PR_threadNum, 0, PR_getCurrentStream()>>>
				(0, 1, prArgs[PR_curr_dev].dev, devCount, devCount + 1);
			CUDA_CHECK_ERROR(
				cudaEventRecord(stop2, PR_getCurrentStream()), "");

			CUDA_CHECK_ERROR(cudaMemcpyAsync(
				hostCount,
				devCount,
				countSize,
				cudaMemcpyDeviceToHost,
				PR_getCurrentStream()),
			"");
			cudaEventRecord(copy, PR_getCurrentStream());

			CUDA_CHECK_ERROR(cudaStreamSynchronize(PR_getCurrentStream()), "");

			CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsedMSKernel, start, stop), "");
			CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsedMSKernel2, start, stop2), "");
			CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsedMSCopy, stop, copy), "");
			printf("reduction kernel took: %f ms (the two %f ms, copy %f ms)\n",
				elapsedMSKernel, elapsedMSKernel2, elapsedMSCopy);

			printf("[2] Aborts = %lu Commits = %lu\n", hostCount[1], hostCount[0]);

			//PR_CPY_TO_HOST(a, inBuf.buf, inBuf.size); // this serializes
			//TODO: retrieveIO fails copying commits and aborts...

			avgCommits += d->PR_nbCommits / (double)NB_ITERATIONS;
			avgAborts += d->PR_nbAborts / (double)NB_ITERATIONS;
			avgKernelTime += d->PR_kernelTime / (double)NB_ITERATIONS;


			// PR_CHECK_CUDA_ERROR(cudaFree(inBuf.buf), "free");

			double abort_rate = avgAborts*NB_ITERATIONS / (TXS_EACH_THREAD*d->PR_threadNum); // TODO:
			double throughput = d->PR_blockNum*d->PR_threadNum*TXS_EACH_THREAD / ((avgKernelTime) / 1000);
			totThroughput += throughput;
			printf("Time for the kernel: %f ms\n", avgKernelTime);
			printf("Nb. Aborts (avg)  = %f \n", avgAborts);
			printf("Nb. Commits (avg) = %f \n", avgCommits);
			printf("Abort rate  = %f \n", abort_rate);
			printf("throughput is = %lf txs/second\n", throughput);

			//printf("max: %d\n",(numeric_limits<int>::max)());
			printf(" Device %i --------------------\n", PR_curr_dev);
			
			CUDA_CHECK_ERROR(cudaEventDestroy(start), "");
			CUDA_CHECK_ERROR(cudaEventDestroy(stop), "");
			CUDA_CHECK_ERROR(cudaEventDestroy(stop2), "");
			CUDA_CHECK_ERROR(cudaEventDestroy(copy), "");
		}

		FILE *stats_file_fp = fopen("stats.txt", "a");
		if (ftell(stats_file_fp) < 1) {
			fprintf(stats_file_fp, "#"
				"BLOCKS\t"
				"THREADS\t"
				"THROUGHPUT\t"
				"PROB_ABORT\n"
			);
		}
		fprintf(stats_file_fp, "%i\t", d->PR_blockNum);
		fprintf(stats_file_fp, "%i\t", d->PR_threadNum);
		fprintf(stats_file_fp, "%.0lf\t", totThroughput);
		fprintf(stats_file_fp, "%lf\n", avgAborts/(avgAborts+avgCommits));
		fclose(stats_file_fp);

		for (int j = 0; j < NUMBER_OF_DEVICES; ++j) {
			PR_curr_dev = j;
			d = &(PR_global[PR_curr_dev]);
			int nbDevices, deviceId;
			PR_CHECK_CUDA_ERROR(cudaGetDeviceCount(&nbDevices), "");
			deviceId = PR_curr_dev % nbDevices;
			PR_CHECK_CUDA_ERROR(cudaSetDevice(deviceId), "");
			cudaFree(((memcd_pr_dev_buff_ext_t*)prArgs[j].host.pr_args_ext)->state);
			PR_disposeIO(&prArgs[j]);
			cudaFree(prArgs[j].host.pr_args_ext);
			// cudaFree(prArgs[j].dev.pr_args_ext); // TODO: used managed malloc here
		}
		PR_teardown();

		//delete[] a;
		for (int j = 0; j < NUMBER_OF_DEVICES; ++j) {
			cudaFree(a[j]->globalTs);
			cudaFree(a[j]->key);
			cudaFree(a[j]->state);
			cudaFree(a[j]->extraKey);
			cudaFree(a[j]->val);
			cudaFree(a[j]->ts_CPU);
			cudaFree(a[j]->ts_GPU);
			cudaFree(a[j]->extraVals);
			cudaFree(a[j]);
		}
		PR_CHECK_CUDA_ERROR(cudaDeviceReset(), "cudaDeviceReset failed!");
	}
	return 0;
}

static void initResources(memcd_t *memcdPtr)
{
	size_t sizeCache = MEMCD_NB_SETS * MEMCD_NB_WAYS * sizeof(int);
	// ...
	PR_CHECK_CUDA_ERROR(cudaMallocManaged(&(memcdPtr->globalTs), sizeof(unsigned)), "cudaMalloc");
	PR_CHECK_CUDA_ERROR(cudaMallocManaged(&(memcdPtr->key), sizeCache), "cudaMalloc");
	PR_CHECK_CUDA_ERROR(cudaMallocManaged(&(memcdPtr->state), sizeCache), "cudaMalloc");
	PR_CHECK_CUDA_ERROR(cudaMemset(memcdPtr->state, 0, sizeCache), ""); // empty slot
	PR_CHECK_CUDA_ERROR(cudaMallocManaged(&(memcdPtr->extraKey), sizeCache*MEMCD_PAYLOAD_KEY), "cudaMalloc");
	PR_CHECK_CUDA_ERROR(cudaMallocManaged(&(memcdPtr->val), sizeCache), "cudaMalloc");
	PR_CHECK_CUDA_ERROR(cudaMallocManaged(&(memcdPtr->ts_CPU), sizeCache), "cudaMalloc");
	PR_CHECK_CUDA_ERROR(cudaMallocManaged(&(memcdPtr->ts_GPU), sizeCache), "cudaMalloc");
	PR_CHECK_CUDA_ERROR(cudaMallocManaged(&(memcdPtr->extraVals), sizeCache*MEMCD_PAYLOAD), "cudaMalloc");
	*(memcdPtr->globalTs) = 0;
	memcdPtr->nbSets = MEMCD_NB_SETS;
	memcdPtr->nbWays = MEMCD_NB_WAYS;
}

static void GPUbuffer_ZIPF(int devId)
{
  int buffer_last = size_of_GPU_input_buffer/sizeof(int);

  // 1st item is generated 10% of the times
  unsigned maxGen = MEMCD_CONFL_SPACE * MEMCD_NB_WAYS;
  zipf_setup(maxGen, 0.5);

	// Creates a buffer WITHOUT conflicts
	int *cpu_ptr = GPUinputBuffer[devId];
  unsigned rnd, zipfRnd; // = (*zipf_dist)(generator);

  // int done = 0;

	for (int i = 0; i < buffer_last; ++i) {
    zipfRnd = zipf_gen();
    rnd = ((zipfRnd / MEMCD_CONFL_SPACE) * 2) * MEMCD_CONFL_SPACE
      + (zipfRnd % MEMCD_CONFL_SPACE);
		cpu_ptr[i] = rnd;
		printf("GenKey: %i\n", rnd);
	}

	// Creates a buffer WITH conflicts
	cpu_ptr += buffer_last; // advances the buffer space
  // done = 0;
	for (int i = 0; i < buffer_last; ++i) {
    zipfRnd = zipf_gen();
		// cpu_ptr[i] = i % 100;// maxGen - zipfRnd;
    rnd = ((zipfRnd / MEMCD_CONFL_SPACE) * 2 + 1) * MEMCD_CONFL_SPACE
      + (zipfRnd % MEMCD_CONFL_SPACE);
		cpu_ptr[i] = rnd;
		printf("GenKey: %i\n", rnd);
	}
}

__global__ void setupKernel(int64_t *state, int devId, memcd_t *appState)
{
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	state[id] = id;
	RAND_R_FNC(state[id]);
}

__device__ unsigned PR_i_rand(pr_tx_args_dev_host_s args, unsigned n)
{
	memcd_pr_dev_buff_ext_t *_args = (memcd_pr_dev_buff_ext_t*)args.pr_args_ext1;
	int64_t *state = _args->state;
	int id = PR_THREAD_IDX;
	int x = 0;
	/* Copy state to local memory for efficiency */
	int64_t localState = state[id];
	x = RAND_R_FNC(localState);
	state[id] = x;
	return x % n;
}

__global__ void memcdTx(PR_globalKernelArgs)
{
	int tid = PR_THREAD_IDX;

	PR_enterKernel(tid);

	memcd_t *input = (memcd_t*)(args.inBuf);
	int devId = input->devId;
	unsigned clock = *(input->globalTs);

	if (tid == 0) {
		printf("In dev%i with clock = %u\n", devId, clock);
	}

	// TODO: Populate the cache first
	// // Do 50% writes / 50% reads (TODO: change this)
	// if ((tid / MEMCD_NB_WAYS) & 1) {
	// 	// do write
	// 	memcdReadTx(PR_txCallArgs);
	// } else {
	// 	// do read
		memcdWriteTxSimple(PR_txCallArgs);
	// }

	PR_exitKernel();
}

// TODO: this is not ok
__device__ unsigned memcached_global_clock = 0;

/*********************************
 *	readKernelTransaction()
 *
 *  Main PR-STM transaction kernel
 **********************************/
 // PR_MAX_RWSET_SIZE = 4
__device__
// __launch_bounds__(1024, 1) // TODO: what is this for?
void memcdReadTx(PR_txCallDefArgs)
{
	int tid = PR_THREAD_IDX;

	int wayId = tid % (MEMCD_NB_WAYS /*+ devParsedData.trans*/);
	int targetKeyIdx = tid / (MEMCD_NB_WAYS /* +TXid in thread*/); // id of the key that each thread will take

	// num_ways threads will colaborate for the same input
	// REQUIREMENT: 1 block >= num_ways

	memcd_t *input = (memcd_t*)args.inBuf;
	PR_GRANULE_T       *keys = (PR_GRANULE_T*)input->key;
	PR_GRANULE_T  *extraKeys = (PR_GRANULE_T*)input->extraKey;
	PR_GRANULE_T     *values = (PR_GRANULE_T*)input->val;
	PR_GRANULE_T  *extraVals = (PR_GRANULE_T*)(input->extraVals);
	PR_GRANULE_T     *ts_CPU = (PR_GRANULE_T*)input->ts_CPU;
	PR_GRANULE_T     *ts_GPU = (PR_GRANULE_T*)input->ts_GPU;
	PR_GRANULE_T      *state = (PR_GRANULE_T*)input->state;

	// TODO: out is NULL
	int  *out = (int*)args.outBuf;
	// int           curr_clock = *((int*)input->curr_clock);
	int          *input_keys = (int*)input->inputs;
	int               nbSets = input->nbSets;
	int               nbWays = input->nbWays;
	int            sizeCache = nbSets * nbWays;
	int *outExtraVals = out + blockDim.x*gridDim.x*TXS_EACH_THREAD;

	for (int i = 0; i < TXS_EACH_THREAD; ++i) { // num_ways keys
		out[threadIdx.x+blockDim.x*blockIdx.x*TXS_EACH_THREAD + i] = 0;
	}

	for (int i = 0; i < (nbWays + TXS_EACH_THREAD); ++i) { // num_ways keys
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

			// // TODO: some BUG on verifying the key (blocks PR-STM)
			// /*thread_key_check = */PR_read(&keys[thread_pos]);
			PR_read(&extraKeys[thread_pos]);
			PR_read(&extraKeys[thread_pos+sizeCache]);
			PR_read(&extraKeys[thread_pos+2*sizeCache]);

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
			}

			out[targetKeyIdx + i] = 1;
			// TODO: out also returns the payload
			int sizeOut = blockDim.x*gridDim.x*TXS_EACH_THREAD;
			outExtraVals[(targetKeyIdx + i)] = thread_val;
			outExtraVals[sizeOut+(targetKeyIdx + i)] = thread_val1;
			outExtraVals[2*sizeOut+(targetKeyIdx + i)] = thread_val2;
			outExtraVals[3*sizeOut+(targetKeyIdx + i)] = thread_val3;
			outExtraVals[4*sizeOut+(targetKeyIdx + i)] = thread_val4;
			outExtraVals[5*sizeOut+(targetKeyIdx + i)] = thread_val5;
			outExtraVals[6*sizeOut+(targetKeyIdx + i)] = thread_val6;
			outExtraVals[7*sizeOut+(targetKeyIdx + i)] = thread_val7;
		}

		PR_txCommit(); // TODO: do we want tansactions this long?

		// __syncthreads();
		// if (!foundKey[threadIdx.x]) {
		// 	printf("Key %i not found \n", input_key);
		// }
	}
}

// TODO: this is a naive approach
__device__ void memcdWriteTxSimple(PR_txCallDefArgs)
{
	int tid = PR_THREAD_IDX;

	memcd_t           *input = (memcd_t*)args.inBuf;
	// int                 *out = (int*)args.outBuf;
	// int        *outExtraVals = out + blockDim.x*gridDim.x*TXS_EACH_THREAD;
	PR_GRANULE_T       *keys = (PR_GRANULE_T*)input->key;
	// PR_GRANULE_T  *extraKeys = (PR_GRANULE_T*)input->extraKey;
	PR_GRANULE_T     *values = (PR_GRANULE_T*)input->val;
	// PR_GRANULE_T  *extraVals = (PR_GRANULE_T*)input->extraVals;
	// PR_GRANULE_T     *ts_CPU = (PR_GRANULE_T*)input->ts_CPU;
	PR_GRANULE_T     *ts_GPU = (PR_GRANULE_T*)input->ts_GPU;
	PR_GRANULE_T      *state = (PR_GRANULE_T*)input->state;
	unsigned      curr_clock = *((unsigned*)input->globalTs);
	int          *input_keys = (int*)input->inputs;
	int          *input_vals = (int*)input->inputs;
	int               nbSets = (int)input->nbSets;
	int               nbWays = (int)input->nbWays;
	// int            sizeCache = nbSets*nbWays;

	int input_key = input_keys[tid];
	int input_val = input_vals[tid];
	int mod_key = input_key % nbSets;

	PR_txBegin();
	int posiblePos = -1;
	for (int i = 0; i < nbWays; ++i) {
		int thread_pos = mod_key*nbWays + i;

		int r_key = PR_read(&(keys[thread_pos]));
		int r_state = PR_read(&(state[thread_pos]));

		int is_found = (r_key == input_key && (r_state & MEMCD_VALID));
		int is_empty = !(r_state & MEMCD_VALID);

		if (is_found) {
			PR_write(&(values[thread_pos]), input_val); // TODO: payload
			PR_write(&(ts_GPU[thread_pos]), curr_clock);
			posiblePos = -1;
			break;
		}

		if (is_empty && posiblePos == -1) {
			posiblePos = thread_pos;
		}
	}
	if (posiblePos != -1) {
		PR_write(&(keys[posiblePos]), input_key); // TODO: payload
		PR_write(&(values[posiblePos]), input_val); // TODO: payload
		PR_write(&(ts_GPU[posiblePos]), curr_clock);
	}
	PR_txCommit();
}


// TODO: IMPORTANT ---> set PR_MAX_RWSET_SIZE to number of ways
// TODO: the function below does not work!!!
//       need to do EXACTLY 1 TX per thread
/*********************************
 *	writeKernelTransaction()
 *
 *  Main PR-STM transaction kernel
 **********************************/
__device__ void memcdWriteTx(PR_txCallDefArgs)
{
	int tid = PR_THREAD_IDX;

	const int maxWarpSlices = 32; // 32*32 == 1024
	int warpSliceID = threadIdx.x / MEMCD_NB_WAYS;
	int wayId = tid % (MEMCD_NB_WAYS /*+ devParsedData.trans*/);
	int reductionID = wayId / 32;
	int reductionSize = max(MEMCD_NB_WAYS / 32, 1);
	int targetKey = tid / (MEMCD_NB_WAYS + TXS_EACH_THREAD); // id of the key that each group of num_ways thread will take

	__shared__ int reduction_is_found[maxWarpSlices]; // TODO: use shuffle instead
	__shared__ int reduction_is_empty[maxWarpSlices]; // TODO: use shuffle instead
	__shared__ int reduction_empty_min_id[maxWarpSlices]; // TODO: use shuffle instead
	__shared__ int reduction_min_ts[maxWarpSlices]; // TODO: use shuffle instead

	// num_ways threads will colaborate for the same input
	// REQUIREMENT: 1 block >= num_ways

	memcd_t *input = (memcd_t*)args.inBuf;
	int  *out = (int*)args.outBuf;
	int *outExtraVals = out + blockDim.x*gridDim.x*TXS_EACH_THREAD;
	PR_GRANULE_T       *keys = (PR_GRANULE_T*)input->key;
	PR_GRANULE_T  *extraKeys = (PR_GRANULE_T*)input->extraKey;
	PR_GRANULE_T     *values = (PR_GRANULE_T*)input->val;
	PR_GRANULE_T  *extraVals = (PR_GRANULE_T*)input->extraVals;
	PR_GRANULE_T     *ts_CPU = (PR_GRANULE_T*)input->ts_CPU;
	PR_GRANULE_T     *ts_GPU = (PR_GRANULE_T*)input->ts_GPU;
	PR_GRANULE_T      *state = (PR_GRANULE_T*)input->state;
	unsigned      curr_clock = *((unsigned*)input->globalTs);
	int          *input_keys = (int*)input->inputs;
	int          *input_vals = (int*)input->inputs;
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

	// TODO: write kernel is too slow
	for (int i = 0; i < nbWays + TXS_EACH_THREAD; ++i) {

		__syncthreads(); // TODO: check with and without this
		// TODO
		__syncthreads(); // TODO: check with and without this

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

		thread_is_found = (thread_key == input_key && (thread_state & MEMCD_VALID));
		thread_is_empty = !(thread_state & MEMCD_VALID);
		int empty_min_id = thread_is_empty ? tid : tid + 32; // warpSize == 32
		int min_ts = thread_ts;

		int warp_is_found = thread_is_found; // 1 someone has found; 0 no one found
		int warp_is_empty = thread_is_empty; // 1 someone has empty; 0 no empties
		const int FULL_MASK = 0xffffffff;
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
			PR_txBegin(); // TODO: I think this may not work

			checkKey = PR_read(&keys[thread_pos]); // read-before-write
			/*checkKey1 = */PR_read(&keys[thread_pos]); // read-before-write
			/*checkKey2 = */PR_read(&keys[thread_pos+sizeCache]); // read-before-write
			/*checkKey3 = */PR_read(&keys[thread_pos+2*sizeCache]); // read-before-write
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
			// TODO: it seems not to reach this if but the nbRetries is needed
						// if (nbRetries == 8191) printf("thr%i aborted 8191 times for key%i thread_pos=%i rsetSize=%lu, wsetSize=%lu\n",
						// 	id, input_key, thread_pos, pr_args.rset.size, pr_args.wset.size);
			PR_txCommit();
			out[targetKey + i] = 1;
			outExtraVals[targetKey + i] = checkKey;
		}

		// if(id == 0) printf("is found=%i\n", thread_is_found);

		// TODO: if num_ways > 32 this does not work very well... (must use shared memory)
		//       using shared memory --> each warp compute the min then: min(ResW1, ResW2)
		//       ResW1 and ResW2 are shared
		// was it found?

		if (!warp_is_found && thread_is_empty && empty_min_id == tid) {
			// the low id thread must be the one that writes
			PR_txBegin(); // TODO: I think this may not work

			checkKey = PR_read(&keys[thread_pos]); // read-before-write
			/*checkKey1 = */PR_read(&keys[thread_pos]); // read-before-write
			/*checkKey2 = */PR_read(&keys[thread_pos+sizeCache]); // read-before-write
			/*checkKey3 = */PR_read(&keys[thread_pos+2*sizeCache]); // read-before-write
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
			out[targetKey + i] = 0;
			outExtraVals[targetKey + i] = checkKey;
		}

		// // not found, none empty --> evict the oldest
		if (!warp_is_found && !warp_is_empty && min_ts == thread_ts) {
			PR_txBegin(); // TODO: I think this may not work

			checkKey = PR_read(&keys[thread_pos]); // read-before-write
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
			// 	id, input_key, thread_pos, pr_args.rset.size, pr_args.wset.size);
			PR_txCommit();
			out[targetKey + i] = 0;
			outExtraVals[targetKey + i] = checkKey;
		}
	}
}
// ----------------------
