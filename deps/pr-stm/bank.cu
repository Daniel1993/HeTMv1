#include "murmurhash2.cuh"

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "helper_timer.h"

// needs to be defined when the library is compiled
// #define PR_MAX_RWSET_SIZE 8

#include "pr-stm.cuh" // implementation
#include "pr-stm-track-rwset.cuh"

typedef struct {
	int devId;
	curandState *state;
} bank_pr_dev_buff_ext_t;

typedef struct {
	int devId;
	curandState *state;
} bank_pr_args_ext_t;

#define PR_rand(n) \
	PR_i_rand(args, n) \
//

#define NUMBER_OF_DEVICES 1

__global__ void setupKernel(curandState *state, int devId);
__device__ unsigned PR_i_rand(pr_tx_args_dev_host_s args, unsigned n);


PR_DEVICE void PR_beforeKernel_EXT(PR_txCallDefArgs) { }
PR_DEVICE void PR_afterKernel_EXT (PR_txCallDefArgs) { }
PR_DEVICE void PR_beforeBegin_EXT (PR_txCallDefArgs) { }
PR_DEVICE void PR_afterCommit_EXT (PR_txCallDefArgs) { }

// in pr-stm-track-rwset.cuh
// PR_DEVICE void PR_after_val_locks_EXT (PR_args_s *args) { }
// PR_DEVICE void PR_after_writeback_EXT (PR_args_s *args, int i, PR_GRANULE_T *addr, PR_GRANULE_T val) { }

__constant__ PR_DEVICE unsigned PR_seed = 1234; // TODO: set seed

__global__ void setupKernel(curandState *state, int devId)
{
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	curand_init(PR_seed, id, 0, &state[id]);
}

__device__ unsigned PR_i_rand(pr_tx_args_dev_host_s args, unsigned n)
{
	bank_pr_dev_buff_ext_t *_args = (bank_pr_dev_buff_ext_t*)args.pr_args_ext1;
	curandState *state = _args->state;
	int id = PR_THREAD_IDX;
	int x = 0;
	/* Copy state to local memory for efficiency */
	curandState localState = state[id];
	x = curand(&localState);
	state[id] = localState;
	return x % n;
}


#include <iostream>
#include <fstream>
#include <cstdio>
#include <ctime>

//For testing purposses
#define firstXEn           0    //Set to 1 to force transactions to happen between the firstX accounts
#define firstX             200  //Used by firstXEn

#define hashNum            1    // how many accounts shared 1 lock
#define BANK_NB_ACCOUNTS   16 //total accounts number 10M = 2621440 integer
//#define PR_threadNum 128 //threads number
#define PR_WRITE_ONLY_THRS 0   // how many threads do all write
#define PR_READ_ONLY_THRS  0   // how many threads do all read
//#define PR_blockNum 5	//block number
#define BANK_NB_TRANSFERS  2   // transfer money between 2 accounts
#define TransEachThread    1   // how many transactions each thread do
#define iterations         1  // loop how many times

using namespace std;

// random Function random several different numbers and store them into idx(Local array which stores idx of every source in Global memory).
__device__ void random_Kernel(PR_txCallDefArgs, int *idx, curandState *state, int size)
{
	int i, j;

	for (i = 0; i < BANK_NB_TRANSFERS; i++)
	{
		int m = 0;
		while (m < 1){
			// idx[i] = ((threadIdx.x + blockIdx.x*blockDim.x)*2 + m) % size;
			idx[i] = PR_rand(size);
#if firstXEn==1
			idx[i] = PR_rand(firstX);
#endif
			bool hasEqual = 0;
			for (j = 0; j < i; j++)	//random different numbers
			{
				if (idx[i] == idx[j])
				{
					hasEqual = 1;
					break;
				}
			}
			if (hasEqual != 1)	//make sure they are different
				m++;
		}
	}
}

__global__ void addKernelTransaction2(PR_globalKernelArgs)
{
	bank_pr_dev_buff_ext_t *_args = (bank_pr_dev_buff_ext_t*)args.pr_args_ext1;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	PR_enterKernel(tid);

	int i = 0, j;	//how many transactions one thread need to commit
	int target;
	PR_GRANULE_T nval;
	int idx[BANK_NB_TRANSFERS];
	PR_GRANULE_T reads[BANK_NB_TRANSFERS];
	PR_GRANULE_T *accounts = (PR_GRANULE_T*)args.inBuf;
	size_t nbAccounts = args.inBuf_size / sizeof(PR_GRANULE_T);
	curandState *state = _args->state;

	random_Kernel(PR_txCallArgs, idx, state, nbAccounts);	//get random index

	while (i++ < TransEachThread) { // each thread need to commit x transactions
		PR_txBegin();

		// printf("[%d] accesses %d & %d\n", tid, idx[0], idx[1]);
		for (j = 0; j < BANK_NB_TRANSFERS; j++)	{ //open read all accounts from global memory
//			reads[j] = accounts[idx[j]];
			reads[j] = PR_read(accounts + idx[j]);
		}

		for (j = 0; j < BANK_NB_TRANSFERS / 2; j++) {
			target = j*2;
			nval = reads[target] - 1; // -money
//			accounts[idx[target]] = nval;
			PR_write(accounts + idx[target], nval); //write changes to write set

			target = j*2+1;
			nval = reads[target] + 1; // +money
//			accounts[idx[target]] = nval;
			PR_write(accounts + idx[target], nval); //write changes to write set
		}
		PR_txCommit();
	}

	PR_exitKernel();
}

// void impl_extra_pr_clbk_before_run_ext(pr_tx_args_s *args) {}

int main(int argc, char* argv[])
{
	PR_global_data_s *d;
	double avgKernelTime = 0.0;
	double avgAborts     = 0.0;
	double avgCommits    = 0.0;
	double totThroughput = 0.0;

	if (argc == 3) {
		for (int j = 0; j < NUMBER_OF_DEVICES; j++) {
			PR_curr_dev = j;
			d = &(PR_global[PR_curr_dev]);
			d->PR_blockNum = atoi(argv[1]);
			d->PR_threadNum = atoi(argv[2]) % 1025;
		}
	} else {
		printf("usage %s <number of blocks> <number of threads>\n", argv[0]);
		return -1;
	}

	for (int j = 0; j < 1; j++) {
		printf("Blocks: %d, Threads: %d\n", d->PR_blockNum, d->PR_threadNum);

		//int size = (BANK_NB_ACCOUNTS + 2000000 * j);
		int size = BANK_NB_ACCOUNTS;
		int sum;
		// int *a = new int[size];	//accounts array
		int *a[NUMBER_OF_DEVICES];
		pr_buffer_s inBuf[NUMBER_OF_DEVICES], outBuf;

		for (int i = 0; i < NUMBER_OF_DEVICES; ++i) {
			PR_CHECK_CUDA_ERROR(cudaMallocManaged(&a[i], BANK_NB_ACCOUNTS * sizeof(int)), "cudaMalloc");
			PR_CHECK_CUDA_ERROR(cudaMallocManaged(&a[i], BANK_NB_ACCOUNTS * sizeof(int)), "cudaMalloc");
			inBuf[i].buf = a[i];
			inBuf[i].size = BANK_NB_ACCOUNTS * sizeof(int);
		}

		outBuf.buf = NULL;
		outBuf.size = 0;

		printf("My Algorithm for %d accounts(transfer among %d accounts) with %d threads!!\n", size, BANK_NB_ACCOUNTS, d->PR_threadNum*d->PR_blockNum);
		/*int *state = new int[size];	//locks array
		printf("222!!\n");*/

		PR_CHECK_CUDA_ERROR(cudaMemset(a[0], 10, BANK_NB_ACCOUNTS * sizeof(int)), "");
		PR_CHECK_CUDA_ERROR(cudaMemset(a[1], 10, BANK_NB_ACCOUNTS * sizeof(int)), "");

		for (int i = 0; i < size; i++) { for (int j = 0; j < NUMBER_OF_DEVICES; ++j) { a[j][i] = 10; }}
		sum = 0;
		for (int i = 0; i < size; i++) { sum += a[0][i]; }
		printf("sum before = %d\n", sum);

		// Add vectors in parallel.
		// TODO: new interface
		pr_tx_args_s prArgs[NUMBER_OF_DEVICES];
		bank_pr_dev_buff_ext_t *cuRandBuf;

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

			PR_init((PR_init_params_s){
				.nbStreams = 2,
				.lockTableSize = PR_LOCK_TABLE_SIZE
			});

			PR_ALLOC(prArgs[PR_curr_dev].dev.pr_args_ext1, sizeof(bank_pr_dev_buff_ext_t));
			prArgs[PR_curr_dev].host.pr_args_ext1 = malloc(sizeof(bank_pr_dev_buff_ext_t));

			cuRandBuf = (bank_pr_dev_buff_ext_t*)prArgs[PR_curr_dev].host.pr_args_ext1;
			cuRandBuf->devId = PR_curr_dev;
			PR_ALLOC(cuRandBuf->state, d->PR_blockNum * d->PR_threadNum * sizeof(curandState));
			PR_CPY_TO_DEV(prArgs[PR_curr_dev].dev.pr_args_ext1, prArgs[PR_curr_dev].host.pr_args_ext1, sizeof(bank_pr_dev_buff_ext_t));

			PR_createStatistics(&prArgs[PR_curr_dev]); // statistics on device 0 only

			// PR_CHECK_CUDA_ERROR(cudaMalloc((void**)&inBuf.buf, BANK_NB_ACCOUNTS * sizeof(int)), "cudaMalloc");

			CUDA_CHECK_ERROR(cudaFuncSetCacheConfig(setupKernel, cudaFuncCachePreferL1), "");
			setupKernel <<< d->PR_blockNum, d->PR_threadNum >>>(cuRandBuf->state, cuRandBuf->devId);
			PR_prepareIO(&prArgs[PR_curr_dev], inBuf[PR_curr_dev], outBuf);

			CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "");
		}
		// ------------------------- INIT EACH DEV

		for (int i = 0; i < iterations; ++i) {

			for (int j = 0; j < NUMBER_OF_DEVICES; ++j) {
				PR_curr_dev = j;
				PR_run(addKernelTransaction2, &prArgs[PR_curr_dev]); // aborts on fail
			}

			// if (i % 4 == 0) {
			// 	PR_checkpointAbortsCommits();
			// 	printf("[it:%i] ---------- \n", i);
			// }

			for (int j = 0; j < NUMBER_OF_DEVICES; ++j) {
				PR_curr_dev = j;
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

			avgCommits += d->PR_nbCommits / (double)iterations;
			avgAborts += d->PR_nbAborts / (double)iterations;
			avgKernelTime += d->PR_kernelTime / (double)iterations;


			// PR_CHECK_CUDA_ERROR(cudaFree(inBuf.buf), "free");

			sum = 0;
			for (int i = 0; i < size; i++) { sum += a[0][i]; }
			printf("sum after = %d\n", sum);

			double abort_rate = avgAborts*iterations / (TransEachThread*d->PR_threadNum); // TODO:
			double throughput = d->PR_blockNum*d->PR_threadNum*TransEachThread / ((avgKernelTime) / 1000);
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
			cudaFree(((bank_pr_dev_buff_ext_t*)prArgs[j].host.pr_args_ext)->state);
			PR_disposeIO(&prArgs[j]);
			cudaFree(prArgs[j].host.pr_args_ext);
			// cudaFree(prArgs[j].dev.pr_args_ext); // TODO: used managed malloc here
		}
		PR_teardown();

		//delete[] a;
		for (int j = 0; j < NUMBER_OF_DEVICES; ++j) {
			cudaFree(a[j]);
		}
		PR_CHECK_CUDA_ERROR(cudaDeviceReset(), "cudaDeviceReset failed!");
	}
	return 0;
}
