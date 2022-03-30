#include "hetm-log.h"
#include "hetm.cuh"
#include "stm-wrapper.h"
#include "stm.h" // depends on STM
#include "knlman.hpp"
#include "hetm-cmp-kernels.cuh"
// #include ".h" // depends on STM

#include "bb_lib.h"

#include <list>
#include <mutex>

using namespace memman;

extern std::mutex HeTM_statsMutex; // defined in hetm-threading-cpu

// ------------ aux functions
// static void enterCPU();
// static void exitCPU();
static void hetmHandleSyncPhase();
// --------------------------

// ------------ aux vars
static int threadId;
static HeTM_callback callback;
static void *clbkArgs;
// ---------------------


// -----------------------------------------------------------------------------
// .88b  d88.  .d8b.  d888888b d8b   db      db       .d88b.   .d88b.  d8888b. 
// 88'YbdP`88 d8' `8b   `88'   888o  88      88      .8P  Y8. .8P  Y8. 88  `8D 
// 88  88  88 88ooo88    88    88V8o 88      88      88    88 88    88 88oodD' 
// 88  88  88 88~~~88    88    88 V8o88      88      88    88 88    88 88~~~   
// 88  88  88 88   88   .88.   88  V888      88booo. `8b  d8' `8b  d8' 88      
// YP  YP  YP YP   YP Y888888P VP   V8P      Y88888P  `Y88P'   `Y88P'  88       
// -----------------------------------------------------------------------------

void HeTM_cpu_thread()
{
  enterCPU();

  do
  {
    hetmHandleSyncPhase();

    // HETM_DEB_THRD_CPU("Thread %i callback", threadId);
    callback(threadId, clbkArgs); // does 1 transaction

    HeTM_thread_data[0]->curNbTxs++;
    if (HeTM_get_GPU_status(0) == HETM_BATCH_DONE) {
      // transaction done while comparing
      HeTM_thread_data[0]->curNbTxsNonBlocking++;
    }
  }
  while (CONTINUE_COND);

  exitCPU();
}

// ---------------------------------------------------------------
//  .d8b.  db    db db    db      d8888b.  .d8b.  d8888b. d888888b 
// d8' `8b 88    88 `8b  d8'      88  `8D d8' `8b 88  `8D `~~88~~' 
// 88ooo88 88    88  `8bd8'       88oodD' 88ooo88 88oobY'    88    
// 88~~~88 88    88  .dPYb.       88~~~   88~~~88 88`8b      88    
// 88   88 88b  d88 .8P  Y8.      88      88   88 88 `88.    88    
// YP   YP ~Y8888P' YP    YP      88      YP   YP 88   YD    YP    
// ---------------------------------------------------------------

static void hetmHandleSyncPhase()
{
  if (HeTM_gshared_data.isGPUEnabled != 0) {
    pollIsRoundComplete(0);
  }
}

void enterCPU()
{
  threadId = HeTM_thread_data[0]->id;
  callback = HeTM_thread_data[0]->callback;
  clbkArgs = HeTM_thread_data[0]->args;

  // TIMER_T t1, t2;
  // static thread_local double addedTime = 0;
  // static thread_local int nbTimes = 0;
  // TIMER_READ(t1);

  // TODO: check order
  TM_INIT_THREAD(HeTM_shared_data[0].mempool_hostptr, HeTM_shared_data[0].sizeMemPool);

  HETM_DEB_THRD_CPU("starting CPU worker %i", threadId);
  for (int j = 0; j < Config::GetInstance()->NbGPUs(); ++j)
  {
    Config::GetInstance()->SelDev(j);
    // knlman_add_stream(); // each thread has its stream
    // HeTM_thread_data[j]->stream = knlman_get_current_stream();
    HeTM_thread_data[j]->stream = HeTM_memStream[j];
    HETM_DEB_THRD_CPU("   dev %i stream %p\n", j, HeTM_thread_data[j]->stream);
    HeTM_thread_data[j]->wSetLog = stm_thread_local_log;
    CUDA_CHECK_ERROR(cudaEventCreate(&HeTM_thread_data[j]->cmpStartEvent), "");
    CUDA_CHECK_ERROR(cudaEventCreate(&HeTM_thread_data[j]->cmpStopEvent), "");
    CUDA_CHECK_ERROR(cudaEventCreate(&HeTM_thread_data[j]->cpyWSetStartEvent), "");
    CUDA_CHECK_ERROR(cudaEventCreate(&HeTM_thread_data[j]->cpyWSetStopEvent), "");

    for (int i = 0; i < STM_LOG_BUFFER_SIZE; ++i)
    {
      CUDA_CHECK_ERROR(cudaEventCreate(&(HeTM_thread_data[j]->cpyLogChunkStartEvent[i])), "");
      CUDA_CHECK_ERROR(cudaEventCreate(&(HeTM_thread_data[j]->cpyLogChunkStopEvent[i])), "");
    }
    HeTM_thread_data[j]->logChunkEventCounter = 0;
    HeTM_thread_data[j]->logChunkEventStore = 0;

    CUDA_CHECK_ERROR(cudaEventCreate(&HeTM_thread_data[j]->cpyDatasetStartEvent), "");
    CUDA_CHECK_ERROR(cudaEventCreate(&HeTM_thread_data[j]->cpyDatasetStopEvent), "");

    hetm_batchCount = &HeTM_gshared_data.batchCount; // TODO: same code in GPU!!!

    stm_log_init();

    HeTM_sync_barrier(j);
  }
  HETM_DEB_THRD_CPU("Thread %i passes barrier", threadId);

  runBeforeCPU(threadId, clbkArgs);
}

void exitCPU()
{
  if (HeTM_gshared_data.isGPUEnabled == 0) {
    __sync_add_and_fetch(&HeTM_stats_data.nbTxsCPU, HeTM_thread_data[0]->curNbTxs);
    __sync_add_and_fetch(&HeTM_stats_data.nbCommittedTxsCPU, HeTM_thread_data[0]->curNbTxs);
    // HeTM_stats_data.nbDroppedTxsCPU == 0;
  } else {
    Config::GetInstance()->SelDev(0);
    NVTX_POP_RANGE();
  }

  HETM_DEB_THRD_CPU("exiting CPU worker %i", threadId);

  // printf("[%2i] doing %i TXs for %f s (%f TXs/s) \n", HeTM_thread_data->id,
  //   nbTimes, addedTime, nbTimes / addedTime);

  HETM_DEB_THRD_CPU("Time cpy WSet = %10fms - Time cmp = %10fms \n"
                    "Total empty space first chunk = %zu B\n",
  HeTM_thread_data[0]->timeCpySum, HeTM_thread_data[0]->timeCmpSum,
  HeTM_thread_data[0]->emptySpaceFirstChunk);

  HeTM_statsMutex.lock();
  HeTM_stats_data.totalTimeCpyWSet += HeTM_thread_data[0]->timeCpySum;
  HeTM_stats_data.totalTimeCmp += HeTM_thread_data[0]->timeCmpSum;
  HeTM_stats_data.totalTimeCpyDataset += HeTM_thread_data[0]->timeCpyDatasetSum;
  HeTM_stats_data.timeNonBlocking += HeTM_thread_data[0]->timeBackoff;
  HeTM_stats_data.timeBlocking += HeTM_thread_data[0]->timeBlocked;
  HeTM_statsMutex.unlock();

  runAfterCPU(threadId, clbkArgs);
  
  TM_EXIT_THREAD();
}
