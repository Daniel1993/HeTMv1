#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define __USE_GNU

#include "test.hpp"
#include "hetm-cmp-kernels.cuh"
#include "cuda_util.h"
#include "hetm-log.h"

/* ################################################################### *
* GLOBALS
* ################################################################### */

// global
const int TOT_NB_THREADS_CPU = 4;
const int THREADS_PER_BLOCK_GPU = 512;
const int TOT_NB_BLOCKS_GPU = 256;
const int MAX_ROUNDS = 20;

static thread_local HETM_LOG_T *CPUlog; // no STM
static kernel_input_s globalState;
static kernel_input_s *globalStateDev;
static kernel_input_s *globalStateDevOnGPU;

int *mempoolCPU;
int *mempoolGPU;

barrier_t stopTestBarrier;

static int doneCPUThread[TOT_NB_THREADS_CPU];

/* ################################################################### *
* TRANSACTION THREADS
* ################################################################### */

static void test(int id, void *data)
{
	if (doneCPUThread[id]) return;
	int pos;
  if (globalState.roundId & 1) {
    pos = id;
  } else {
		pos = TOT_NB_THREADS_CPU-id-1;
  }
	mempoolCPU[pos] = id;
	stm_log_newentry(CPUlog, (long*)&mempoolCPU[pos], id, globalState.roundId);

	// if (id == 0) {
	// 	// deterministic conflict
	// 	stm_log_newentry(CPUlog, (long*)&mempoolCPU[TOT_NB_THREADS_CPU], -1, globalState.roundId);
	// }

	if (globalState.roundId >= MAX_ROUNDS) {
		if (id == 0 && !HeTM_is_stop()) {
			HeTM_set_is_stop(1);
			barrier_cross(stopTestBarrier);
			__sync_synchronize();
		}
	}
	doneCPUThread[id] = 1;
}

static void beforeCPU(int id, void *data)
{
	CPUlog = stm_log_init();
}

static void afterCPU(int id, void *data) { }

/* ################################################################### *
* CUDA THREAD
* ################################################################### */

// TODO: add a beforeBatch and afterBatch callbacks

static void before_batch(int id, void *data)
{
	for (int i = 0; i < TOT_NB_THREADS_CPU; ++i) {
		doneCPUThread[i] = 0;
	}
}

static void after_batch(int id, void *data)
{
	printf(" --- ROUND %i FINISHED ---\n", globalState.roundId);
	for (int i = 0; i < TOT_NB_THREADS_CPU; i++) {
		if (mempoolCPU[i] != i && (globalState.roundId & 1)) {
			printf("[%i i=%i] error on CPU: %i expected but got %i\n", globalState.roundId, i,
				i, mempoolCPU[i]);
			break;
		} else if (mempoolCPU[i] != TOT_NB_THREADS_CPU-i-1 && (globalState.roundId & 1 == 0)) {
			printf("[%i i=%i] error on CPU: %i expected but got %i\n", globalState.roundId, i,
				TOT_NB_THREADS_CPU-i-1, mempoolCPU[i]);
			break;
		}
	}
	for (int i = 0; i < THREADS_PER_BLOCK_GPU*TOT_NB_BLOCKS_GPU; i++) {
		if (mempoolCPU[i+TOT_NB_THREADS_CPU] != i && (globalState.roundId & 1)) {
			printf("[%i i=%i] error on GPU: %i expected but got %i\n", globalState.roundId, i,
				i, mempoolCPU[i+TOT_NB_THREADS_CPU]);
			break;
		} else if (mempoolCPU[i+TOT_NB_THREADS_CPU] != THREADS_PER_BLOCK_GPU*TOT_NB_BLOCKS_GPU-i-1 && (globalState.roundId & 1 == 0) ) {
			printf("[%i i=%i] error on GPU: %i expected but got %i\n", globalState.roundId, i,
				THREADS_PER_BLOCK_GPU*TOT_NB_BLOCKS_GPU-i-1, mempoolCPU[i+TOT_NB_THREADS_CPU]);
			break;
		}
	}
	globalState.roundId++;
	globalStateDev->roundId++;
	CUDA_CPY_TO_DEV(globalStateDevOnGPU, globalStateDev, sizeof(kernel_input_s));
	// TODO: conflict mechanism
}

static void choose_policy(int, void*) {
  // -----------------------------------------------------------------------
  int idGPUThread = HeTM_shared_data.nbCPUThreads;
  long TXsOnCPU = 0,
		TXsOnGPU = THREADS_PER_BLOCK_GPU*TOT_NB_BLOCKS_GPU;
  for (int i = 0; i < HeTM_shared_data.nbCPUThreads; ++i) {
    TXsOnCPU += HeTM_shared_data.threadsInfo[i].curNbTxs;
  }

  // TODO: this picks the one with higher number of TXs
	// TODO: GPU only gets the stats in the end --> need to remove the dropped TXs
	HeTM_stats_data.nbTxsGPU += TXsOnGPU;
  if (TXsOnCPU > TXsOnGPU) {
    HeTM_gshared_data.policy = HETM_GPU_INV;
		if (HeTM_is_interconflict()) {
			HeTM_stats_data.nbDroppedTxsGPU += TXsOnGPU;
		} else {
			HeTM_stats_data.nbCommittedTxsGPU += TXsOnGPU;
		}
  } else {
    HeTM_gshared_data.policy = HETM_CPU_INV;
		HeTM_stats_data.nbCommittedTxsGPU += TXsOnGPU;
  }
  // -----------------------------------------------------------------------
}

static void test_cuda(int id, void *data)
{
  HeTM_async_request((HeTM_async_req_s){
      .args = NULL,
      .fn = LaunchTestKernel
    });
	// TODO: check mempool
}

static void afterGPU(int id, void *data)
{
  printf("nb_batches=%li\n", HeTM_stats_data.nbBatches);

  // leave this one
  printf("CUDA thread terminated after %li(%li successful) run(s). \nTotal cuda execution time: %f ms.\n",
    HeTM_stats_data.nbBatches, HeTM_stats_data.nbBatchesSuccess, HeTM_stats_data.timeGPU);
}

/* ################################################################### *
*
* MAIN
*
* ################################################################### */

int main(int argc, char **argv)
{
	int nb_threadsCPU = TOT_NB_THREADS_CPU;
	int tot_nb_threads = TOT_NB_THREADS_CPU + 1;
	size_t sizeOfMempool = TOT_NB_THREADS_CPU + THREADS_PER_BLOCK_GPU*TOT_NB_BLOCKS_GPU;

	memman_alloc_dual("test_kernel_input", sizeof(kernel_input_s), 0);
	globalStateDev = (kernel_input_s*)memman_get_cpu(NULL);
	globalStateDevOnGPU = (kernel_input_s*)memman_get_gpu(NULL);

  HeTM_init((HeTM_init_s){
#if CPU_INV == 1
    .policy       = HETM_CPU_INV,
#else /* GPU_INV */
    .policy       = HETM_GPU_INV,
#endif /**/
    .nbCPUThreads = TOT_NB_THREADS_CPU,
    .nbGPUBlocks  = THREADS_PER_BLOCK_GPU,
    .nbGPUThreads = TOT_NB_BLOCKS_GPU,
// #if HETM_CPU_EN == 0
//     .isCPUEnabled = 0,
//     .isGPUEnabled = 1
// #elif HETM_GPU_EN == 0
//     .isCPUEnabled = 1,
//     .isGPUEnabled = 0
// #else /* both on */
    .isCPUEnabled = 1,
    .isGPUEnabled = 1
// #endif
  });

	// TODO: HETM_CPU_EN | HETM_GPU_EN

	nb_threadsCPU  = HeTM_shared_data.nbCPUThreads;
	tot_nb_threads = HeTM_shared_data.nbThreads;
	PR_blockNum    = TOT_NB_BLOCKS_GPU;
  PR_threadNum   = THREADS_PER_BLOCK_GPU;

  HeTM_mempool_init(sizeOfMempool*sizeof(int), 0);
	barrier_init(stopTestBarrier, 2); // thread 0 + main thread

  CUDA_CHECK_ERROR(cudaSetDevice(0), "");
  CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "");

	// allocates the memory pool that will be shared among CPU/GPU
  HeTM_alloc((void**)&mempoolCPU, (void**)&mempoolGPU, sizeOfMempool*sizeof(int));

	globalStateDev->roundId = 0;
	globalStateDev->mempool = mempoolGPU + TOT_NB_THREADS_CPU; // no colisions
	globalState.roundId = 0;
	globalState.mempool = mempoolCPU;

	memset(mempoolCPU, 0, sizeOfMempool*sizeof(int));
	CUDA_CPY_TO_DEV(mempoolGPU, mempoolCPU, sizeOfMempool*sizeof(int));
	CUDA_CPY_TO_DEV(globalStateDevOnGPU, globalStateDev, sizeof(kernel_input_s));

  //Clear flags
  HeTM_set_is_stop(0);

  /* Start threads */
  HeTM_after_gpu_finish(afterGPU);
  HeTM_before_cpu_start(beforeCPU);
  HeTM_after_cpu_finish(afterCPU);
  HeTM_start(test, test_cuda, NULL);
	HeTM_after_batch(after_batch);
	HeTM_before_batch(before_batch);

	HeTM_choose_policy(choose_policy);

  printf("STARTING...\n");
  barrier_cross(stopTestBarrier);
  printf("STOPPING...\n");

  /* Wait for thread completion */
  HeTM_join_CPU_threads();

  /*Cleanup GPU*/
  cudaDeviceSynchronize();
	HeTM_destroy();
  HeTM_destroyCurandState();
  HeTM_mempool_destroy();

  memman_select("test_kernel_input");
  memman_free_dual();

  return EXIT_SUCCESS;
}
