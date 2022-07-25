#include "hetm-timer.h"
#include "hetm-cmp-kernels.cuh"
#include "knlman.hpp"
#include "arch.h"
#include "hetm-aux.cuh"
#include "hetm.cuh"
#include "pr-stm.cuh"

#include <list>
#include <mutex>

using namespace memman;

extern std::mutex HeTM_statsMutex; // defined in hetm-threading-cpu

// // ------------ aux functions
// static void enterGPU();
// static void exitGPU();
// static int timeIsOver();
// static void checkIsExit();
// static void notifyCPUNextBatch();
// // --------------------------

// ------------ aux vars
static int threadId;
static void *clbkArgs;
static TIMER_T t_start;
static TIMER_T t_now;
// ---------------------

// -----------------------------------------------------------------------------
// .88b  d88.  .d8b.  d888888b d8b   db      db       .d88b.   .d88b.  d8888b. 
// 88'YbdP`88 d8' `8b   `88'   888o  88      88      .8P  Y8. .8P  Y8. 88  `8D 
// 88  88  88 88ooo88    88    88V8o 88      88      88    88 88    88 88oodD' 
// 88  88  88 88~~~88    88    88 V8o88      88      88    88 88    88 88~~~   
// 88  88  88 88   88   .88.   88  V888      88booo. `8b  d8' `8b  d8' 88      
// YP  YP  YP YP   YP Y888888P VP   V8P      Y88888P  `Y88P'   `Y88P'  88       
// -----------------------------------------------------------------------------

void HeTM_gpu_thread()
{
  TIMER_T t1, t2, t3;

  if (!HeTM_gshared_data.isGPUEnabled) return;

  enterGPU();

  do {
    runGPUBeforeBatch(threadId, (void*)HeTM_thread_data[0]);
    TIMER_READ(t1);

    NVTX_PUSH_RANGE("Run TX Batch", NVTX_RUN_TX_BATCH);
    roundCountAfterBatch = *hetm_batchCount;
    do {
      runGPUBeforeKernel(threadId, (void*)HeTM_thread_data[0]);
      runGPUBatch();
      waitGPUBatchEnd();
      runGPUAfterKernel(threadId, (void*)HeTM_thread_data[0]);
    } 
    while(!timeIsOver());

    // ------------- takes statistics
    extern int PR_enable_auto_stats;
    extern pr_tx_args_s HeTM_pr_args[HETM_NB_DEVICES];
    PR_enable_auto_stats = 1;
    runGPUBeforeKernel(threadId, (void*)HeTM_thread_data[0]);
    runGPUBatch();
    waitGPUBatchEnd();
    runGPUAfterKernel(threadId, (void*)HeTM_thread_data[0]);
    PR_enable_auto_stats = 0;
    // -------------

    NVTX_POP_RANGE(); // NVTX_RUN_TX_BATCH
    NVTX_PUSH_RANGE("Run Sync Phase", NVTX_PROF_SYNC_PHASE);

    TIMER_READ(t2);
    // TODO: I'm taking GPU time in PR-STM
    double timeLastBatch = TIMER_DIFF_SECONDS(t1, t2);
    HeTM_stats_data.timePRSTM += timeLastBatch;

    HETM_DEB_THRD_GPU("\n-----------------------\n --- Batch waitCMPEnd (%li sucs, %li fail) \n-----------------------\n",
      HeTM_stats_data.nbBatchesSuccess, HeTM_stats_data.nbBatchesFail);

    notifyBatchIsDone();
    if (HeTM_gshared_data.isCPUEnabled) {
      waitGPUCMPEnd(0);
    }
    TIMER_READ(t3);
     // TODO: let the next batch begin if dataset is still not copied

    HeTM_stats_data.timeCMP += TIMER_DIFF_SECONDS(t2, t3);

    if (!HeTM_gshared_data.isCPUEnabled) {
      getGPUStatistics(NULL); // usually done in mergeGPUDataset() but is disabled
      checkIsExit();
    }
// ---------------------
    HETM_DEB_THRD_GPU("\n-----------------------\n --- Batch mergeDataset (%li sucs, %li fail) \n-----------------------\n",
      HeTM_stats_data.nbBatchesSuccess, HeTM_stats_data.nbBatchesFail);
    if (HeTM_gshared_data.isCPUEnabled)
    { // TODO: what if there is 2 GPUs?
      mergeGPUDataset();
      // ---------------------
      WAIT_ON_FLAG(isDatasetSyncDone);
      // ---------------------
    }

    TIMER_READ(t1);
    HeTM_stats_data.timeAfterCMP += TIMER_DIFF_SECONDS(t3, t1);

    HETM_DEB_THRD_GPU("\n-----------------------\n --- End of batch (%li sucs, %li fail) \n-----------------------\n",
      HeTM_stats_data.nbBatchesSuccess, HeTM_stats_data.nbBatchesFail);
  // }

    accumulateStatistics();

    runGPUAfterBatch(threadId, (void*)HeTM_thread_data[0]);
    
    notifyCPUNextBatch();
    
    doGPUStateReset();
    WAIT_ON_FLAG(isGPUResetDone); // forces GPU reset before start

    notifyCPUNextBatch();
    NVTX_POP_RANGE(); // NVTX_PROF_SYNC_PHASE
    if (!CONTINUE_COND)
      break;
  } while (1);

  exitGPU();
}

// ---------------------------------------------------------------
//  .d8b.  db    db db    db      d8888b.  .d8b.  d8888b. d888888b 
// d8' `8b 88    88 `8b  d8'      88  `8D d8' `8b 88  `8D `~~88~~' 
// 88ooo88 88    88  `8bd8'       88oodD' 88ooo88 88oobY'    88    
// 88~~~88 88    88  .dPYb.       88~~~   88~~~88 88`8b      88    
// 88   88 88b  d88 .8P  Y8.      88      88   88 88 `88.    88    
// YP   YP ~Y8888P' YP    YP      88      YP   YP 88   YD    YP    
// ---------------------------------------------------------------

void enterGPU()
{
  int nbGPUs = Config::GetInstance()->NbGPUs();
  threadId = HeTM_thread_data[0]->id;
  clbkArgs = HeTM_thread_data[0]->args;
  initGPUPeerCpy();
  // TODO: check order
  for (int j = 0; j < nbGPUs; j++)
  {
    PR_curr_dev = j;
    pr_tx_args_s *pr_args = getPrSTMmetaData(j);
    PR_createStatistics(pr_args);

    Config::GetInstance()->SelDev(j);
    HETM_DEB_THRD_GPU("THREAD%i init cuda events for GPU %i\n", threadId, j);
    CUDA_CHECK_ERROR(cudaEventCreate(&HeTM_thread_data[j]->cmpStartEvent), "");
    CUDA_CHECK_ERROR(cudaEventCreate(&HeTM_thread_data[j]->cmpStopEvent), "");
    CUDA_CHECK_ERROR(cudaEventCreate(&HeTM_thread_data[j]->cpyWSetStartEvent), "");
    CUDA_CHECK_ERROR(cudaEventCreate(&HeTM_thread_data[j]->cpyWSetStopEvent), "");
    CUDA_CHECK_ERROR(cudaEventCreate(&HeTM_thread_data[j]->cpyDatasetStartEvent), "");
    CUDA_CHECK_ERROR(cudaEventCreate(&HeTM_thread_data[j]->cpyDatasetStopEvent), "");
    // knlman_add_stream(); // each thread has its stream
    // HeTM_thread_data[j]->stream = knlman_get_current_stream();
    HeTM_thread_data[j]->stream = HeTM_memStream[j];
  }

  runBeforeGPU(threadId, clbkArgs);

  for (int j = 0; j < HETM_NB_DEVICES; j++) {
    HeTM_sync_barrier(j);
  }

  // TIMER_READ(t1WCpy);
  roundCountAfterBatch = 1;

  TIMER_READ(t_start);
}

int timeIsOver()
{
  TIMER_READ(t_now);
  double diff = TIMER_DIFF_SECONDS(t_start, t_now);
  // if (!(diff < HeTM_gshared_data.timeBudget))
  //   printf("diff = %f (budget = %f)\n", diff, HeTM_gshared_data.timeBudget);
  return !(diff < HeTM_gshared_data.timeBudget);
}

void checkIsExit()
{
  if (!HeTM_is_stop()) {
    for (int j = 0; j < HETM_NB_DEVICES; j++) {
      HeTM_set_GPU_status(j, HETM_BATCH_RUN);
    }
  } else { // Times up
    for (int j = 0; j < HETM_NB_DEVICES; j++) {
      HeTM_set_GPU_status(j, HETM_IS_EXIT);
      HeTM_flush_barrier(j);
    }
  }
}

// #include <unistd.h>

void notifyCPUNextBatch() // syncs with waitNextBatch in threading-cpu
{
  TIMER_READ(t_start);
  HETM_DEB_THRD_GPU("{BLOCK_WAIT_NEXT_BATCH} GPUs notify threads can start next batch\n");
  // printf("          ---- GPU notifies next batch\n");
  // sleep(1);
  HeTM_sync_next_batch(); // wakes the threads (which wait the last GPU), GPU will re-run
}

void exitGPU()
{
  CUDA_CHECK_ERROR(cudaDeviceSynchronize(), ""); // before terminating the benchmark

  notifyCPUNextBatch();

  // TODO: this was per iteration
  // This will run sync
  RUN_ASYNC(getGPUPRSTMStats, HeTM_thread_data[0]);
  RUN_ASYNC(getGPUStatistics, -1);

// ---------------------
  WAIT_ON_FLAG(isAfterCmpDone); // avoids a warning in the compiler
  WAIT_ON_FLAG(isGetStatsDone);
  WAIT_ON_FLAG(isGetPRStatsDone);
// ---------------------

  HeTM_statsMutex.lock();
  for (int j = 0; j < HETM_NB_DEVICES; ++j) {
    HeTM_stats_data.totalTimeCpyWSet += HeTM_thread_data[j]->timeCpySum;
    HeTM_stats_data.totalTimeCmp += HeTM_thread_data[j]->timeCmpSum;
    HeTM_stats_data.totalTimeCpyDataset += HeTM_thread_data[j]->timeCpyDatasetSum;
  }
  HeTM_statsMutex.unlock();

  HeTM_stats_data.timeGPU = 0;
  PR_global_data_s *d;
  for (int j = 0; j < HETM_NB_DEVICES; j++) {
    PR_curr_dev = j;
    d = &(PR_global[PR_curr_dev]);
    HeTM_stats_data.timeGPU = d->PR_kernelTime;
  }

  runAfterGPU(threadId, clbkArgs);
  HETM_DEB_THRD_GPU("Time copy dataset %10fms - Time cpy WSet %10fms - Time cmp %10fms\n",
    HeTM_thread_data[0]->timeCpyDatasetSum,
    HeTM_thread_data[0]->timeCpySum,
    HeTM_thread_data[0]->timeCmpSum);
  CHUNKED_LOG_TEARDOWN(); // deletes the freed nodes from the CPU

  destroyGPUPeerCpy();
}

