#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <assert.h>

#include "hetm-log.h"
#include "cuda_util.h"
#include "memman.hpp"
#include "knlman.hpp"

#include "hetm.cuh"
#include "hetm-cmp-kernels.cuh"

#include "bb_lib.h"
#include "stm.h"
#include "stm-wrapper.h"

using namespace memman;
using namespace knlman;

// --------------------
// Variables (global)
HeTM_shared_s HeTM_shared_data[HETM_NB_DEVICES];
HeTM_gshared_s HeTM_gshared_data;
HeTM_statistics_s HeTM_stats_data;
hetm_pc_s *HeTM_offload_pc;
void *HeTM_memStream[HETM_NB_DEVICES];
void *HeTM_memStream2[HETM_NB_DEVICES];

MemObjOnDev HeTM_interGPUConflDetect_entryObj;
MemObjOnDev HeTM_CPUGPUConflDetect_entryObj;
MemObjOnDev HeTM_CPUrsGPUwsConflDetect_entryObj;

KnlObj *HeTM_checkTxCompressed;
KnlObj *HeTM_earlyCheckTxCompressed;
KnlObj *HeTM_applyTxCompressed;
KnlObj *HeTM_interGPUConflDetect;
KnlObj *HeTM_CPUGPUConflDetect;
KnlObj *HeTM_CPUrsGPUwsConflDetect;
KnlObj *HeTM_checkTxExplicit;
// --------------------

static void run_interGPUConflDetect(knlman_callback_params_s params);
static void CUDART_CB interGPUconflCallback(cudaStream_t event, cudaError_t status, void *data);

static void run_CPUGPUConflDetect(knlman_callback_params_s params);
static void run_CPUrsGPUwsConflDetect(knlman_callback_params_s params);

#define MAX_FREE_NODES       0x800
static unsigned long freeNodesPtr;
static HeTM_async_req_s reqsBuffer[MAX_FREE_NODES];

// static void hetm_impl_pr_clbk_before_run_ext(pr_tx_args_s *args) {}
// static void hetm_impl_pr_clbk_after_run_ext(pr_tx_args_s *args) {}

int HeTM_init(HeTM_init_s init)
{
  HeTM_gshared_data.isCPUEnabled   = 1;
  HeTM_gshared_data.isGPUEnabled   = 1;
  HeTM_gshared_data.nbCPUThreads   = init.nbCPUThreads;
  HeTM_gshared_data.nbGPUBlocks    = init.nbGPUBlocks;
  HeTM_gshared_data.nbGPUThreads   = init.nbGPUThreads;
  HeTM_gshared_data.timeBudget     = init.timeBudget;
  HeTM_gshared_data.policy         = init.policy;
  if (!init.isCPUEnabled) {
    // disable CPU usage
    HeTM_gshared_data.nbCPUThreads = 0;
    HeTM_gshared_data.isCPUEnabled = 0;
    HeTM_gshared_data.nbThreads    = 1;
  } else if (!init.isGPUEnabled) {
    // disable GPU usage
    HeTM_gshared_data.isGPUEnabled = 0;
    HeTM_gshared_data.nbThreads    = init.nbCPUThreads;
  } else {
    // both on
    HeTM_gshared_data.nbThreads    = init.nbCPUThreads + 1;
  }

  barrier_init(HeTM_gshared_data.CPUBarrier, init.nbCPUThreads); // only the controller thread
  barrier_init(HeTM_gshared_data.nextBatchBarrier, HeTM_gshared_data.nbThreads); // only the controller thread
  barrier_init(HeTM_gshared_data.BBsyncBarrier, HeTM_gshared_data.nbThreads); // only the controller thread
  // TODO: check if we need 1 barrier per GPU

  int nbGPUs = Config::GetInstance(HETM_NB_DEVICES, CHUNK_GRAN*sizeof(PR_GRANULE_T))->NbGPUs();
  for (int j = 0; j < nbGPUs; ++j)
  {
    Config::GetInstance(j)->SelDev(j);
    HeTM_set_GPU_status(j, HETM_BATCH_RUN);
    if (!init.isCPUEnabled) {
      // disable CPU usage
      barrier_init(HeTM_shared_data[j].GPUBarrier, 1); // only the controller thread
    } else if (!init.isGPUEnabled) {
      // disable GPU usage
      HeTM_set_GPU_status(j, HETM_IS_EXIT);
      barrier_init(HeTM_shared_data[j].GPUBarrier, init.nbCPUThreads); // 0 controller
    } else {
      // both on
      barrier_init(HeTM_shared_data[j].GPUBarrier, init.nbCPUThreads+1); // 1 controller
    }

    // TODO: duplicated variables
    size_t sizeOfThreads = HeTM_gshared_data.nbThreads*sizeof(HeTM_thread_s);
    malloc_or_die(HeTM_shared_data[j].threadsInfo, HeTM_gshared_data.nbThreads);
    memset(HeTM_shared_data[j].threadsInfo, 0, sizeOfThreads);
    assert(HeTM_shared_data[j].threadsInfo != NULL);

    MemObjBuilder b_interGPUConflDetect;
    MemObjBuilder b_CPUGPUConflDetect;
    MemObjBuilder b_CPUrsGPUwsConflDetect;
    
    HeTM_interGPUConflDetect_entryObj.AddMemObj(new MemObj(b_interGPUConflDetect
      .SetOptions(0)
      ->SetSize(sizeof(HeTM_cmp_s))
      ->AllocDevPtr()
      ->AllocHostPtr(),
      j
    ));
    HeTM_CPUGPUConflDetect_entryObj.AddMemObj(new MemObj(b_CPUGPUConflDetect
      .SetOptions(0)
      ->SetSize(sizeof(HeTM_cmp_s))
      ->AllocDevPtr()
      ->AllocHostPtr(),
      j
    ));
    HeTM_CPUrsGPUwsConflDetect_entryObj.AddMemObj(new MemObj(b_CPUrsGPUwsConflDetect
      .SetOptions(0)
      ->SetSize(sizeof(HeTM_cmp_s))
      ->AllocDevPtr()
      ->AllocHostPtr(),
      j
    ));
  }

  KnlObjBuilder b_interGPUConflDetect;
  KnlObjBuilder b_CPUGPUConflDetect;
  KnlObjBuilder b_CPUrsGPUwsConflDetect;

  HeTM_interGPUConflDetect
    = new KnlObj(b_interGPUConflDetect
      .SetCallback(run_interGPUConflDetect)
      ->SetEntryObj(&HeTM_interGPUConflDetect_entryObj)
  );
  HeTM_CPUGPUConflDetect // TODO: add HeTM_cmp_s entryObj
    = new KnlObj(b_CPUGPUConflDetect
      .SetCallback(run_CPUGPUConflDetect)
      ->SetEntryObj(&HeTM_CPUGPUConflDetect_entryObj)
  );
  HeTM_CPUrsGPUwsConflDetect
    = new KnlObj(b_CPUrsGPUwsConflDetect
      .SetCallback(run_CPUrsGPUwsConflDetect)
      ->SetEntryObj(&HeTM_CPUrsGPUwsConflDetect_entryObj)
  );

  HeTM_offload_pc = hetm_pc_init(HETM_PC_BUFFER_SIZE);

  pr_clbk_before_run_ext = hetm_impl_pr_clbk_before_run_ext;
  pr_clbk_after_run_ext = hetm_impl_pr_clbk_after_run_ext;

  HeTM_mempool_init(init.mempool_size, init.mempool_opts);

  hetm_batchCount = &HeTM_gshared_data.batchCount;
  
  for (int i = 0; i < nbGPUs; ++i)
  {
    HeTM_set_is_stop(i, 0);
    PR_curr_dev = i;
    // TODO: init here STM too
    PR_init({
      .nbStreams = 2,
      // .nbStreams = 1,
      .lockTableSize = PR_LOCK_TABLE_SIZE
    }); // inits PR-STM mutex array

    Config::GetInstance()->SelDev(i);

    HeTM_memStream[i] = PR_global[PR_curr_dev].PR_streams[0];
    HeTM_memStream2[i] = PR_global[PR_curr_dev].PR_streams[1];

    pr_tx_args_s *pr_args = getPrSTMmetaData(PR_curr_dev);
    PR_global[PR_curr_dev].PR_blockNum = init.nbGPUBlocks;
    PR_global[PR_curr_dev].PR_threadNum = init.nbGPUThreads;
    PR_createStatistics(pr_args);
    // TODO: each thread creates a local cuda event
  }

  stm_init(); //(HeTM_shared_data[0].mempool_hostptr, HeTM_shared_data[0].sizeMemPool);

  // init FVS library
  unsigned char dummy[(HETM_NB_DEVICES+1)*(HETM_NB_DEVICES+1)];
  memset(dummy, 0, sizeof(dummy));
  graph G = fromSquareMat(HETM_NB_DEVICES+1, (unsigned char*)dummy);
  BB_parameters_s params = {
    .n = 1<<20,
    .repOps = 1<<20
  };
  BB_init_G(params, G);
  return 0;
}

int HeTM_destroy()
{
  delete HeTM_interGPUConflDetect;
  // knlmanObjDestroy(&HeTM_checkTxCompressed);
  // knlmanObjDestroy(&HeTM_earlyCheckTxCompressed);
  // knlmanObjDestroy(&HeTM_checkTxExplicit);
  for (int i = 0; i < Config::GetInstance()->NbGPUs(); ++i)
  {
    Config::GetInstance()->SelDev(i);

    barrier_destroy(HeTM_shared_data[i].GPUBarrier);
    free(HeTM_shared_data[i].threadsInfo);
    HeTM_set_is_stop(i, 1);
    HeTM_async_set_is_stop(i, 1);
    HeTM_mempool_destroy(i);
  }
  hetm_pc_destroy(HeTM_offload_pc);
  BB_destroy();
  return 0;
}

int HeTM_sync_barrier(int devId)
{
  if (!HeTM_is_stop(0)) {
    // printf("[%s] barrier dev %i\n", __FUNCTION__, devId);
    barrier_cross(HeTM_shared_data[devId].GPUBarrier);
  } else {
    // printf("[%s] HeTM_flush_barrier\n", __FUNCTION__);
    HeTM_flush_barrier(devId);
  }
  return 0;
}

int HeTM_sync_CPU_barrier()
{
  if (!HeTM_is_stop(0)) {
    barrier_cross(HeTM_gshared_data.CPUBarrier);
  } else {
    barrier_reset(HeTM_gshared_data.CPUBarrier);
  }
  return 0;
}

int HeTM_sync_next_batch()
{
  if (!HeTM_is_stop(0)) {
    barrier_cross(HeTM_gshared_data.nextBatchBarrier);
  } else {
    barrier_reset(HeTM_gshared_data.nextBatchBarrier);
  }
  return 0;
}

int HeTM_sync_BB()
{
  if (!HeTM_is_stop(0)) {
    barrier_cross(HeTM_gshared_data.BBsyncBarrier);
  } else {
    barrier_reset(HeTM_gshared_data.BBsyncBarrier);
  }
  return 0;
}

int HeTM_flush_barrier(int devId)
{
  barrier_reset(HeTM_shared_data[devId].GPUBarrier);
  return 0;
}

// TODO: only works for a "sufficiently" large buffer
void HeTM_async_request(HeTM_async_req_s req)
{
  HeTM_async_req_s *m_req;
  long idx = HETM_PC_ATOMIC_INC_PTR(freeNodesPtr, MAX_FREE_NODES);
  reqsBuffer[idx].fn   = req.fn;
  reqsBuffer[idx].args = req.args;
  m_req = &reqsBuffer[idx];
    // malloc_or_die(m_req, 1);
  // hetm_pc_consume(free_nodes, (void**)&m_req);
  // printf("[%i] got %p\n", HeTM_thread_data->id, m_req);
  hetm_pc_produce(HeTM_offload_pc, m_req);
}

void HeTM_free_async_request(HeTM_async_req_s *req)
{
  // TODO: is not releasing anything --> assume enough space
  // hetm_pc_produce(free_nodes, req);
  // free(req);
}

static void
run_interGPUConflDetect(
  knlman_callback_params_s params
) {
  dim3 blocks(params.blocks.x, params.blocks.y, params.blocks.z);
  dim3 threads(params.threads.x, params.threads.y, params.threads.z);
  cudaStream_t stream = (cudaStream_t)params.stream;
  MemObj *m = params.entryObj;
  HeTM_cmp_s *data = (HeTM_cmp_s*)m->host;
  // HeTM_thread_s *threadData = (HeTM_thread_s*)data->clbkArgs;

  // TODO: for each chunk -> try cache, if conflict -> go for the full kernel

  // CUDA_EVENT_RECORD(threadData->cmpStartEvent, stream);
  interGPUConflDetect<<<blocks, threads, 0, stream>>>(data->knlArgs, 0);
  // CUDA_EVENT_RECORD(threadData->cmpStopEvent, stream);
  // HETM_DEB_THRD_CPU("\033[0;36m" "Thread %i inter confl detect GPU %i started" "\033[0m",
  //     threadData->id, threadData->devId);
  // TODO: move events somewhere else
  CUDA_CHECK_ERROR(cudaStreamAddCallback(
    stream, interGPUconflCallback, data->clbkArgs, 0
  ), "");
}

static void
run_CPUGPUConflDetect(
  knlman_callback_params_s params
) {
  cudaStream_t stream = (cudaStream_t)params.stream;
  MemObj *m = params.entryObj;
  HeTM_cmp_s *data = (HeTM_cmp_s*)m->host;
  HeTM_knl_cmp_args_s knl_cmp_args = data->knlArgs;
  HeTM_knl_global_s knl_global = knl_cmp_args.knlGlobal;
  size_t cacheSize = knl_global.hostWSetCacheSize;
  unsigned char batchCount = data->knlArgs.batchCount;
  unsigned char *cpuWSetCache = (unsigned char *)data->knlArgs.knlGlobal.hostWSetCache_hostptr;
  unsigned char *rsetGPUcache = (unsigned char *)data->knlArgs.knlGlobal.devRSetCache_hostptr;

  for (int i = 0; i < cacheSize; ++i)
  {
    int threadsX = (params.blocks.x + cacheSize - 1) / cacheSize;
    if (cpuWSetCache[i] == batchCount && rsetGPUcache[i] == batchCount)
    {
      // printf("run_CPUGPUConflDetect pos %i/%zu\n", i, cacheSize);
      dim3 blocks(threadsX, params.blocks.y, params.blocks.z);
      dim3 threads(params.threads.x, params.threads.y, params.threads.z);
      CPUGPUConflDetect<<<blocks, threads, 0, stream>>>(data->knlArgs, i*cacheSize);
    }
  }
}

static void
run_CPUrsGPUwsConflDetect(
  knlman_callback_params_s params
) {
  cudaStream_t stream = (cudaStream_t)params.stream;
  MemObj *m = params.entryObj;
  HeTM_cmp_s *data = (HeTM_cmp_s*)m->host;
  size_t cacheSize = data->knlArgs.knlGlobal.hostWSetCacheSize;
  size_t cacheChunk = (HeTM_gshared_data.sizeMemPool/sizeof(PR_GRANULE_T) + (cacheSize-1)) / cacheSize;
  unsigned char batchCount = data->knlArgs.batchCount;
  unsigned char *cpuRSetCache = (unsigned char *)data->knlArgs.knlGlobal.hostRSetCache_hostptr;
  unsigned char *wsetGPUcache = (unsigned char *)data->knlArgs.knlGlobal.devWSetCache_hostptr;

// printf("run_CPUrsGPUwsConflDetect cacheSize = %zu cacheChunk = %zu batchCount = %i\n", cacheSize, cacheChunk, (int)batchCount);
  for (int i = 0; i < cacheSize; ++i)
  {
    int threadsX = (cacheChunk + params.threads.x - 1) / params.threads.x;
// printf("run_CPUrsGPUwsConflDetect pos %i/%zu cpuRSetCache = %i wsetGPUcache = %i\n", i, cacheSize, (int)cpuRSetCache[i], (int)wsetGPUcache[i]);
    if (cpuRSetCache[i] == batchCount && wsetGPUcache[i] == batchCount)
    {
      dim3 blocks(threadsX, params.blocks.y, params.blocks.z);
      dim3 threads(params.threads.x, params.threads.y, params.threads.z);
// printf("run_CPUrsGPUwsConflDetect<<<%i,%i>>> pos %i/%zu\n", blocks.x, threads.x, i, cacheSize);
      CPUrsGPUwsConflDetect<<<blocks, threads, 0, stream>>>(data->knlArgs, i*cacheChunk);
    }
  }
}

// TODO: callback is probably not needed
static void CUDART_CB
interGPUconflCallback(
  cudaStream_t event,
  cudaError_t status,
  void *data
) {
  HeTM_thread_s *threadData = (HeTM_thread_s*)data;

  CUDA_CHECK_ERROR(status, "Inter confl failed!");

  // TODO: statistics
  // TIMER_T now;
  // TIMER_READ(now);
  // double timeTaken = TIMER_DIFF_SECONDS(threadData->beforeApplyKernel, now);
  // // printf(" --- send to %i wait cmp delay=%fms\n", threadData->id, timeTaken*1000);

  HETM_DEB_THRD_CPU("\033[0;32m" "Thread %i inter confl detect GPU %i is done" "\033[0m",
      threadData->id, threadData->devId);

  // threadData->timeApplyKernels += timeTaken;
  __atomic_store_n(&(HeTM_shared_data[threadData->devId].isInterGPUConflDone), 1, __ATOMIC_RELEASE);
  // __sync_synchronize(); // cmpCallback is called from a different thread
}
