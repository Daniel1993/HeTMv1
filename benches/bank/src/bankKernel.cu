#include <stdio.h>
#include <time.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <curand_kernel.h>
#include "helper_cuda.h"
#include "helper_timer.h"

#include "hetm-log.h"
#include "bankKernel.cuh"
#include "all_bank_parts.cuh"
#include "bitmap.hpp"

#include "pr-stm-wrapper.cuh" // enables the granularity
#include "setupKernels.cuh"

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

__device__ void update_tx(PR_txCallDefArgs, int txCount);
__device__ void update_tx2(PR_txCallDefArgs, int txCount);
__device__ void readIntensive_tx(PR_txCallDefArgs, int txCount);
__device__ void readOnly_tx(PR_txCallDefArgs, int txCount);


/*********************************
 *	bankTx()
 *
 *  Main PR-STM transaction kernel
 **********************************/
__global__ void bankTx(PR_globalKernelArgs)
{
	int id = PR_THREAD_IDX;
	PR_enterKernel(id);
#if BANK_PART == 1 /* Uniform at random: split */
	bank_part1_gpu_run(id, PR_txCallArgs);
#elif BANK_PART == 2 /* Uniform at random: interleaved CPU accesses incremental pages */
	bank_part2_gpu_run(id, PR_txCallArgs);
#elif BANK_PART == 3 /* Zipf: from file */
	bank_part3_gpu_run(id, PR_txCallArgs);
#elif BANK_PART == 4 /* Zipf */
	bank_part4_gpu_run(id, PR_txCallArgs);
#elif BANK_PART == 5
	bank_part5_gpu_run(id, PR_txCallArgs);
#elif BANK_PART == 6 /* contiguous */
	bank_part6_gpu_run(id, PR_txCallArgs);
#elif BANK_PART == 7 /* not used on CPU, GPU blocks */
	bank_part7_gpu_run(id, PR_txCallArgs);
#elif BANK_PART == 8 // same as 7
	bank_part8_gpu_run(id, PR_txCallArgs);
#elif BANK_PART == 9 // same as 5 
	bank_part9_gpu_run(id, PR_txCallArgs);
#elif BANK_PART == 10 // same as 5
	bank_part10_gpu_run(id, PR_txCallArgs);
#else
#pragma "ERROR: Bank part not defined"
#endif
	volatile uint64_t clockBefore = clock64();
	while ((clock64() - clockBefore) < devParsedData.GPU_backoff);

	PR_exitKernel();
}



// random Function random several different numbers and store them into idx(Local array which stores idx of every source in Global memory).
__device__ void random_Kernel(PR_txCallDefArgs, int *idx, int size, int is_intersection)
{
	int i = 0;
	HeTM_GPU_log_s *GPU_log = (HeTM_GPU_log_s*)args.pr_args_ext;
	int devId = GPU_log->devId;

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
	HeTM_GPU_log_s *GPU_log = (HeTM_GPU_log_s*)args.pr_args_ext;
	int devId = GPU_log->devId;

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
