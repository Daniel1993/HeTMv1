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

#define EARLY_CHECK_NB_ENTRIES 8192

using namespace memman;

static std::list<HeTM_callback> beforeCPU, afterCPU;
std::mutex HeTM_statsMutex; // extern in hetm-threading-gpu

#define MAXIMUM_THREADS 1024
#define SWAP(_src, _dst, _tmp) (_tmp) = (_src); (_src) = (_dst); (_dst) = (_tmp);

#define MEASURE_CPY_TIME_END(_evntS, _evntE, _storeDiff, _accDiff) \
  CUDA_EVENT_SYNCHRONIZE(_evntS); \
  CUDA_EVENT_SYNCHRONIZE(_evntE); \
  CUDA_EVENT_ELAPSED_TIME(&(_storeDiff), _evntS, _evntE); \
  if (_storeDiff > 0) { _accDiff += _storeDiff; } else { printf("Error measuring event time (%f) in %s\n", _storeDiff, __PRETTY_FUNCTION__); } \
//
#define MEASURE_CPY_TIME(_evntS, _evntE, _storeDiff, _accDiff, _fn) \
  CUDA_EVENT_RECORD(_evntS, NULL); \
  _fn; \
  CUDA_EVENT_RECORD(_evntE, NULL); \
  MEASURE_CPY_TIME_END(_evntS, _evntE, _storeDiff, _accDiff) \
//

// static int consecutiveFlagCpy = 0; // avoid consecutive copies

// thread_local static int inBackoff[HETM_NB_DEVICES];
// thread_local static int nbCpyRounds[HETM_NB_DEVICES];
// thread_local static int doneWithCPUlog[HETM_NB_DEVICES];
// thread_local static int awakeGPU[HETM_NB_DEVICES];

static void dealWithDatasetSync(int isNonBlock);
static void cpyBMAPtoGPU(int devId);
// static void broadcastCPUdataset(int devId);
static void cpyGPUmodificationsToCPU(int devId);
// static void overrideCPUmodificationsWithGPUdataset(int devId);

// static int launchCmpKernel(HeTM_thread_s*, size_t wsetSize, int devId);
// static int launchApplyKernel(HeTM_thread_s *threadData, size_t wsetSize);
// static void wakeUpGPU(int devId);

static void compareCPUrsAgainstAllGPUws();

int HeTM_before_cpu_start(HeTM_callback req)
{
  beforeCPU.push_back(req);
  return 0;
}
  
int HeTM_after_cpu_finish(HeTM_callback req)
{
  afterCPU.push_back(req);
  return 0;
}

void runBeforeCPU(int id, void *data)
{
  // for (int i = 0; i < HETM_NB_DEVICES; ++i) {
  //   awakeGPU[i] = 0;
  // }
  for (auto it = beforeCPU.begin(); it != beforeCPU.end(); ++it) {
    HeTM_callback clbk = *it;
    clbk(id, data);
  }
}

void runAfterCPU(int id, void *data)
{
  for (auto it = afterCPU.begin(); it != afterCPU.end(); ++it) {
    HeTM_callback clbk = *it;
    clbk(id, data);
  }
}

static int launchCPUGPUConflDetectKernel(int devId)
{
  // TODO: add early validation kernel
  // if (!doApply) { return 0; }

  // for (int j = 0; j < HETM_NB_DEVICES; ++j) {
  long nbGrans = HeTM_gshared_data.sizeMemPool / PR_LOCK_GRANULARITY;
  Config::GetInstance()->SelDev(devId);
  int nbThreadsX = 256;
  int bo = (nbGrans + nbThreadsX-1) / (nbThreadsX);

  // Memory region of the entry object
  // printf("dev = %i batchCount = %li\n", j, HeTM_shared_data[j].batchCount);
  MemObj *m = HeTM_CPUGPUConflDetect->entryObj->GetMemObj(devId);
  HeTM_cmp_s *checkTxCompressed_args         = (HeTM_cmp_s*)m->host;
  checkTxCompressed_args->knlArgs.devId      = devId;
  checkTxCompressed_args->knlArgs.knlGlobal  = *(HeTM_get_global_arg(devId));
  checkTxCompressed_args->knlArgs.otherDevId = HETM_NB_DEVICES; // CPU
  checkTxCompressed_args->knlArgs.nbOfGPUs   = HETM_NB_DEVICES;
  checkTxCompressed_args->knlArgs.sizeWSet   = (int)nbGrans;
  checkTxCompressed_args->knlArgs.sizeRSet   = (int)HeTM_gshared_data.rsetLogSize;
  // checkTxCompressed_args->knlArgs.idCPUThr   = (int)threadData->id;
  checkTxCompressed_args->knlArgs.batchCount = (unsigned char) HeTM_gshared_data.batchCount;

  // if (wsetSize & 1) {
  //   printf("invalid wsetSize=%i\n", wsetSize);
  // }

  HETM_DEB_THRD_GPU("\033[0;36m" "GPU %i HeTM_CPUGPUConflDetect (to check against CPU)" "\033[0m", devId);
  
  // HeTM_CPUGPUConflDetect.select(&HeTM_CPUGPUConflDetect);
  // HeTM_CPUGPUConflDetect.setNbBlocks(bo, 1, 1);
  // HeTM_CPUGPUConflDetect.setThrsPerBlock(nbThreadsX, 1, 1);
  // HeTM_CPUGPUConflDetect.setDevice(devId);
  // HeTM_CPUGPUConflDetect.setStream(HeTM_memStream[devId]); // same stream as in the kernel
  // HeTM_CPUGPUConflDetect.setArgs(&checkTxCompressed_args);
  // HeTM_CPUGPUConflDetect.run();

  m->CpyHtD(HeTM_memStream[devId]);
  HeTM_CPUGPUConflDetect->blocks  = (knlman_dim3_s){ .x = bo,         .y = 1, .z = 1 };
  HeTM_CPUGPUConflDetect->threads = (knlman_dim3_s){ .x = nbThreadsX, .y = 1, .z = 1 };
  HeTM_CPUGPUConflDetect->Run(devId, HeTM_memStream[devId]);

  return 0;
}

static int
launchCPUrsGPUwsConflDetectKernel(
  int devId
) {
  long nbGrans = HeTM_gshared_data.sizeMemPool / PR_LOCK_GRANULARITY;
  Config::GetInstance()->SelDev(devId);
  int nbThreadsX = 256;
  int bo = (nbGrans + (nbThreadsX-1)) / (nbThreadsX);

  // Memory region of the entry object
  // printf("dev = %i batchCount = %li\n", j, HeTM_shared_data[j].batchCount);
  MemObj *m = HeTM_CPUrsGPUwsConflDetect->entryObj->GetMemObj(devId);
  HeTM_cmp_s *checkTxCompressed_args = (HeTM_cmp_s*)m->host;
  // thread_local static HeTM_cmp_s checkTxCompressed_args;
  checkTxCompressed_args->knlArgs.devId      = devId;
  checkTxCompressed_args->knlArgs.knlGlobal  = *(HeTM_get_global_arg(devId));
  checkTxCompressed_args->knlArgs.otherDevId = HETM_NB_DEVICES; // CPU
  checkTxCompressed_args->knlArgs.nbOfGPUs   = HETM_NB_DEVICES;
  checkTxCompressed_args->knlArgs.sizeWSet   = (int)nbGrans;
  checkTxCompressed_args->knlArgs.sizeRSet   = (int)HeTM_gshared_data.rsetLogSize;
  // checkTxCompressed_args->knlArgs.idCPUThr   = (int)threadData->id;
  checkTxCompressed_args->knlArgs.batchCount = (unsigned char) HeTM_gshared_data.batchCount;

  HETM_DEB_THRD_GPU("\033[0;36m" "GPU %i HeTM_CPUrsGPUwsConflDetect (to check against CPU)" "\033[0m", devId);

  m->CpyHtD(HeTM_memStream[devId]);
  HeTM_CPUrsGPUwsConflDetect->blocks  = (knlman_dim3_s){ .x = bo,         .y = 1, .z = 1 };
  HeTM_CPUrsGPUwsConflDetect->threads = (knlman_dim3_s){ .x = nbThreadsX, .y = 1, .z = 1 };
  cudaStreamSynchronize((cudaStream_t)HeTM_memStream2[devId]);
  HeTM_CPUrsGPUwsConflDetect->Run(devId, HeTM_memStream[devId]);

  return 0;
}

void waitNextBatch(int nonBlock) // synchs with notifyCPUNextBatch in threading-gpu
{
  // __sync_synchronize();
  // printf("."); // TODO: TinySTM crashes if this is not here... why?!
  // fflush(stdout);
  NVTX_PUSH_RANGE("wait in CPU", NVTX_PROF_CPU_WAITS_NEXT_BATCH);
  // HETM_DEB_THRD_CPU(" -0- thread %i waits next batch\n", HeTM_thread_data[0]->id);
  if (!nonBlock)
    HeTM_sync_next_batch();

  if (!HeTM_is_stop())
    dealWithDatasetSync(nonBlock);

  // HETM_DEB_THRD_CPU(" -1- thread %i waits next batch\n", HeTM_thread_data[0]->id);
  if (!nonBlock)
    HeTM_sync_next_batch();
  // GPU resets BMAPs
  if (!nonBlock)
    HeTM_sync_next_batch();
  // HETM_DEB_THRD_CPU(" -2- thread %i starts next batch\n", HeTM_thread_data[0]->id);
  NVTX_POP_RANGE(); // NVTX_PROF_CPU_WAITS_NEXT_BATCH
}


// #include <unistd.h> // sleep()
void pollIsRoundComplete(int nonBlock)
{
  int nbDoneGPUs = 0;
  for (int j = 0; j < HETM_NB_DEVICES; ++j) { 
    if (HETM_BATCH_DONE == __atomic_load_n(&(HeTM_get_GPU_status(j)), __ATOMIC_ACQUIRE)) {
      nbDoneGPUs++;
    }
  }
  if (nbDoneGPUs == HETM_NB_DEVICES) {
    cpyCPUwrtsetToGPU(nonBlock);
    // sleep(2);
    waitNextBatch(nonBlock);
  }
}

void resetInterGPUConflFlag() 
{
  if (HeTM_thread_data[0]->id == 0) {
    for (int j = 0; j < HETM_NB_DEVICES; j++) {
      // HETM_DEB_THRD_CPU("\033[0;32m" "Thread 0 reset inter confl detect GPU %i flag" "\033[0m", j);
      __atomic_store_n(&(HeTM_shared_data[j].isInterGPUConflDone), 0, __ATOMIC_RELEASE);
    }
  }
}

static void compareCPUrsAgainstAllGPUws()
{
  int myId = HeTM_thread_data[0]->id;
  int nbGPUs = Config::GetInstance()->NbGPUs();

  for (int j = 0; j < nbGPUs; ++j)
  {
    if (j % HeTM_gshared_data.nbCPUThreads == myId)
      launchCPUrsGPUwsConflDetectKernel(j);
  }
}

/* 
static void overrideCPUmodificationsWithGPUdataset(int devId)
{
  char *CPUwsPtr = (char*)HeTM_shared_data[devId].bmap_wset_CPU_hostptr;
  char *CPUwsPtrCache = (char*)stm_wsetCPUCache;
  unsigned char batch = (unsigned char) HeTM_gshared_data.batchCount;
  // long nbGranules = HeTM_gshared_data.sizeMemPool / PR_LOCK_GRANULARITY;

  Config::GetInstance()->SelDev(devId);
  void *hostMempool = HeTM_mempool.GetMemObj(devId)->host;
  void *devMempool = HeTM_mempool.GetMemObj(devId)->dev;
  
  // for (long g = minRange; g < maxRange; ++g) { // TODO: use the cache
  PR_curr_dev = devId;
  size_t cpySize = HeTM_gshared_data.sizeMemPool;
  // TODO: cannot override the entire thing
  // CUDA_CPY_TO_HOST(hostMempool, devMempool, cpySize);
  NVTX_PUSH_RANGE("overrideCPUwrites", 11);

  if (cpySize < MAXSIZE_BEFORE_SMART_KERNEL)
  {
    HeTM_mempool.GetMemObj(devId)->CpyDtH(HeTM_memStream2[devId]);
    cpySize += HeTM_mempool.GetMemObj(devId)->size;
  }
  else
  {
    cpySize = memman_bmap_cpy_DtH(
      0,
      CPUwsPtrCache,
      CPUwsPtr,
      batch,
      BMAP_GRAN,
      sizeof(PR_GRANULE_T),
      hostMempool,
      devMempool,
      HeTM_gshared_data.sizeMemPool/sizeof(PR_GRANULE_T),
      HeTM_memStream[devId],
      HeTM_memStream2[devId]
    );
  }
  __atomic_add_fetch(&HeTM_stats_data.sizeCpyDataset, cpySize, __ATOMIC_ACQ_REL);
  // cpySize = memman_smart_cpy(CPUwsPtr, batch, PR_LOCK_GRANULARITY, hostMempool, devMempool,
  //   HeTM_gshared_data.sizeMemPool, memman_cpy_to_cpu_fn);
  // printf("[%s] Cpy %zu\n", __FUNCTION__, cpySize);
  NVTX_POP_RANGE();
}
*/

static void
cpyGPUmodificationsToCPU(
  int devId
) {
  PR_curr_dev = devId;
  Config::GetInstance()->SelDev(devId);

  MemObj *m_gpu_wset_cache = HeTM_gpu_wset_cache.GetMemObj(devId);
  MemObj *m_cpu_wset_cache = HeTM_cpu_wset_cache.GetMemObj(devId);
  MemObj *m_gpu_wset       = HeTM_gpu_wset.GetMemObj(devId);
  MemObj *m_mempool        = HeTM_mempool.GetMemObj(devId);

  MemObjCpyBuilder b;
  MemObjCpyDtH m(b
    .SetGranFilter(8) // TODO
    ->SetGranApply(sizeof(PR_GRANULE_T))
    ->SetForceFilter(MEMMAN_FILTER|MEMMAN_ON_COLLISION)
    ->SetFilterVal(HeTM_gshared_data.batchCount)
    ->SetCache(m_gpu_wset_cache)
    ->SetCache2(m_cpu_wset_cache)
    ->SetFilter(m_gpu_wset)
    ->SetDst(m_mempool)
    ->SetSrc(m_mempool)
    ->SetSizeChunk(BMAP_GRAN)
    ->SetStrm1(HeTM_memStream[devId])
    ->SetStrm2(HeTM_memStream[devId])
  );
  size_t cpySize = m.Cpy();
  __atomic_add_fetch(&HeTM_stats_data.sizeCpyDataset, cpySize, __ATOMIC_ACQ_REL);
}

static void
cpyCPUmodificationsToGPU(
  int devId
) {
  Config::GetInstance()->SelDev(devId);
  PR_curr_dev = devId;

  HETM_DEB_THRD_GPU("Copy CPU modifications to GPU%i\n", devId);

  MemObj *m_cpu_wset_cache = HeTM_cpu_wset_cache.GetMemObj(devId);
  MemObj *m_gpu_wset_cache = HeTM_gpu_wset_cache.GetMemObj(devId);
  MemObj *m_cpu_wset       = HeTM_cpu_wset.GetMemObj(devId);
  MemObj *m_mempool        = HeTM_mempool.GetMemObj(devId);

  MemObjCpyBuilder b;
  MemObjCpyHtD m(b
    .SetGranFilter(8) // TODO
    ->SetGranApply(sizeof(PR_GRANULE_T))
    ->SetForceFilter(MEMMAN_FILTER|MEMMAN_ON_COLLISION)
    ->SetFilterVal(HeTM_gshared_data.batchCount)
    ->SetCache(m_cpu_wset_cache)
    ->SetCache2(m_gpu_wset_cache)
    ->SetFilter(m_cpu_wset)
    ->SetDst(m_mempool)
    ->SetSrc(m_mempool)
    ->SetSizeChunk(BMAP_GRAN)
    ->SetStrm1(HeTM_memStream2[devId])
    ->SetStrm2(HeTM_memStream2[devId])
  );
  size_t cpySize = m.Cpy();

  __atomic_add_fetch(&HeTM_stats_data.sizeCpyDataset, cpySize, __ATOMIC_ACQ_REL);
}

static void
cpyModifications(
  int devDst,
  int devSrc,
  int idOverride
) {
  int nbGPUs = Config::GetInstance()->NbGPUs();
  int CPUid = nbGPUs;
  if (devDst == CPUid)
  {
    cpyGPUmodificationsToCPU(devSrc);
    return;
  }
  if (devSrc == CPUid)
  {
    cpyCPUmodificationsToGPU(devDst);
    return;
  }

  HETM_DEB_THRD_GPU("Copy GPU%i modifications to GPU%i\n", devSrc, devDst);
  MemObj *m_dst_mempool = HeTM_mempool.GetMemObj(devDst);
  MemObj *m_src_mempool = HeTM_mempool.GetMemObj(devSrc);
  MemObj *m_src_wset_cache = HeTM_gpu_wset_cache.GetMemObj(devSrc);
  MemObj *m_dst_wset_cache = HeTM_gpu_wset_cache.GetMemObj(devDst);
  MemObj *m_src_wset_filter = HeTM_gpu_wset_ext[devSrc].GetMemObj(devDst);
  int fltr = idOverride ? MEMMAN_NONE : MEMMAN_FILTER|MEMMAN_ON_COLLISION;

  MemObjCpyBuilder b;
  MemObjCpyDtD m(b
    .SetGranFilter(8) // TODO
    ->SetGranApply(sizeof(PR_GRANULE_T))
    ->SetForceFilter(fltr)
    ->SetFilterVal(HeTM_gshared_data.batchCount)
    ->SetCache(m_src_wset_cache)
    ->SetCache2(m_dst_wset_cache)
    ->SetFilter(m_src_wset_filter)
    ->SetDst(m_dst_mempool)
    ->SetSrc(m_src_mempool)
    ->SetSizeChunk(BMAP_GRAN)
    ->SetStrm1(HeTM_memStream[devDst])
    ->SetStrm2(HeTM_memStream2[devDst])
  );

  size_t cpySize = m.Cpy();
  // printf("Copied %zuB from GPU%i (%p) to GPU%i (%p)\n", cpySize, devSrc, m_src_mempool->dev, devDst, m_dst_mempool->dev);
  
  __atomic_add_fetch(&HeTM_stats_data.sizeCpyDataset, cpySize, __ATOMIC_ACQ_REL);
}

/* static void broadcastCPUdataset(int devId)
{
  // const int CPUid = HETM_NB_DEVICES;
  size_t totCpySize = 0;
  size_t cpySize = HeTM_gshared_data.sizeMemPool;

  PR_curr_dev = devId;
  Config::GetInstance()->SelDev(devId);
  void *hostMempool = HeTM_mempool.GetMemObj(devId)->host;
  void *devMempool = HeTM_mempool.GetMemObj(devId)->dev;

  // TODO: merge the bitmaps and copy only the right locations
  CUDA_CPY_TO_DEV_ASYNC(devMempool, hostMempool, cpySize, (cudaStream_t)HeTM_memStream[devId]);
  totCpySize += cpySize;

  __atomic_add_fetch(&HeTM_stats_data.sizeCpyDataset, totCpySize, __ATOMIC_ACQ_REL);
// #ifdef USE_NVTX
//     float elapsedTime;
//     CUDA_EVENT_SYNCHRONIZE(threadData->cpyWSetStartEvent);
//     CUDA_EVENT_SYNCHRONIZE(threadData->cpyWSetStopEvent);
//     CUDA_EVENT_ELAPSED_TIME(&elapsedTime, threadData->cpyWSetStartEvent, threadData->cpyWSetStopEvent);
//     HeTM_stats_data.timeDtD += elapsedTime;
// #endif
} */

static void
cpyBMAPtoGPU(
  int devId
) {
  size_t cpyrdsetBMAP = 0;
  // size_t size = HeTM_gshared_data.sizeMemPool / sizeof(PR_GRANULE_T);

  Config::GetInstance()->SelDev(devId);
  PR_curr_dev = devId;

  MemObj *m_cpu_rset_cache = HeTM_cpu_rset_cache.GetMemObj(devId);
  MemObj *m_gpu_wset_cache = HeTM_gpu_wset_cache.GetMemObj(devId);
  MemObj *m_gpu_rset_cache = HeTM_gpu_rset_cache.GetMemObj(devId);
  MemObj *m_cpu_rset       = HeTM_cpu_rset.GetMemObj(devId);
  MemObj *m_cpu_wset_cache = HeTM_cpu_wset_cache.GetMemObj(devId);
  MemObj *m_cpu_wset       = HeTM_cpu_wset.GetMemObj(devId);

  // TODO: these copies are probably duplicated
  m_gpu_wset_cache->CpyDtH(HeTM_memStream[devId]);
  m_gpu_rset_cache->CpyDtH(HeTM_memStream2[devId]);

  MemObjCpyBuilder b;
  MemObjCpyHtD mRSET(b
    .SetGranFilter(8) // TODO
    ->SetGranApply(1)
    ->SetForceFilter(MEMMAN_ONLY_COLLISION)
    ->SetFilterVal(HeTM_gshared_data.batchCount)
    ->SetCache(m_cpu_rset_cache)
    ->SetCache2(m_gpu_wset_cache)
    ->SetFilter(m_cpu_rset)
    ->SetDst(m_cpu_rset)
    ->SetSrc(m_cpu_rset)
#ifdef BMAP_ENC_1BIT
    ->SetSizeChunk((BMAP_GRAN >> LOG2_32BITS)*sizeof(unsigned))
#else
    ->SetSizeChunk(BMAP_GRAN)
#endif /* BMAP_ENC_1BIT */
    ->SetStrm1(HeTM_memStream[devId])
    ->SetStrm2(HeTM_memStream[devId])
  );
  cpyrdsetBMAP = mRSET.Cpy();

  // TODO: can we put this copy in background somehow?
  MemObjCpyHtD mWSET(b
    .SetGranFilter(8) // TODO
    ->SetGranApply(1)
    ->SetForceFilter(MEMMAN_NONE)
    ->SetFilterVal(HeTM_gshared_data.batchCount)
    ->SetCache(m_cpu_wset_cache)
    ->SetFilter(m_cpu_wset)
    ->SetDst(m_cpu_wset)
    ->SetSrc(m_cpu_wset)
#ifdef BMAP_ENC_1BIT
    ->SetSizeChunk((BMAP_GRAN >> LOG2_32BITS)*sizeof(unsigned))
#else
    ->SetSizeChunk(BMAP_GRAN)
#endif /* BMAP_ENC_1BIT */
    ->SetStrm1(HeTM_memStream2[devId])
    ->SetStrm2(HeTM_memStream2[devId])
  );
  cpyrdsetBMAP += mWSET.Cpy();

  __atomic_add_fetch(&HeTM_stats_data.sizeCpyWSetHtD, cpyrdsetBMAP, __ATOMIC_ACQ_REL);

  // TODO: compare with CPU comparison

  // Copy only the portions that changed
  launchCPUGPUConflDetectKernel(devId);
}

void
cpyCPUwrtsetToGPU(
  int nonBlock
) {
  int nbGPUs = Config::GetInstance()->NbGPUs();
  // First loop: launch kernels
  for (int j = 0; j < nbGPUs; ++j)
  { 
    Config::GetInstance()->SelDev(j);
    NVTX_PUSH_RANGE("cpy CPU wset to GPU", NVTX_PROF_CPY_CPU_TO_1GPU);
    __sync_add_and_fetch(&HeTM_shared_data[j].threadsWaitingSync, 1);
    
    HeTM_thread_data[j]->statusCMP = HETM_CMP_BLOCK;

    // TODO: is this correct? First CPU thread handle this
    if (HeTM_thread_data[j]->id == j % HeTM_gshared_data.nbCPUThreads)
      cpyBMAPtoGPU(j); // copies bitmaps and start the cmp kernel
    
    NVTX_POP_RANGE(); // NVTX_PROF_CPY_CPU_TO_1GPU
  }

  // NVTX_PUSH_RANGE("validate in CPU", NVTX_PROF_VALIDATE_CPU);
  compareCPUrsAgainstAllGPUws();
  // NVTX_POP_RANGE(); // NVTX_PROF_VALIDATE_CPU

  // Second loop: unlock the GPU controller
  for (int j = 0; j < nbGPUs; ++j)
  { 
    Config::GetInstance()->SelDev(j);
    // CUDA_CHECK_ERROR(cudaStreamSynchronize((cudaStream_t)HeTM_memStream[j]), "");
    // CUDA_CHECK_ERROR(cudaStreamSynchronize((cudaStream_t)HeTM_memStream2[j]), "");

    if (!nonBlock)
      HeTM_sync_barrier(j); // unblock GPU for the next round
    // HETM_DEB_THRD_CPU("thread %i decreases threadsWaitingSync (%i)\n", HeTM_thread_data[j]->id, HeTM_shared_data[j].threadsWaitingSync);
    __sync_add_and_fetch(&HeTM_shared_data[j].threadsWaitingSync, -1);

    // if (!nonBlock)
    //   HeTM_sync_barrier(j); // blocks and waits the GPU to do the FVS and dataset apply
  }
  // if (!nonBlock)
  // {
  //   HeTM_sync_next_batch(); // allows GPU side to merge the matrices
  // }
}

static void
dealWithDatasetSync(int isNonBlock)
{
  int tid = HeTM_thread_data[0]->id;
  const int nbGPUs = Config::GetInstance()->NbGPUs();
  const int CPUid = nbGPUs;

  // TODO: requires GPU side to call HeTM_sync_next_batch
  // HETM_DEB_THRD_CPU("thread %i waits for GPU BB\n", tid);
  if (!isNonBlock)
    HeTM_sync_BB(); // wait GPU BB
  // HETM_DEB_THRD_CPU("thread %i after wait for GPU BB\n", tid);

  int *p, *q;
  // int isCPUabort = 1;
  int someCommittingDevice = -1;
  int *sol = BB_getBestSolution();
  int *abortingDevs = BB_getFVS();

  // TODO: is sol sorted?
  p = sol;
  while (*p != -1)
  {
    if (someCommittingDevice == -1 || someCommittingDevice == CPUid)
      someCommittingDevice = *p;
    if (*p == CPUid)
    {
      // isCPUabort = 0;
      if (someCommittingDevice != -1 && someCommittingDevice != CPUid)
        break;
    }
    p++;
  }

  // NVTX_PUSH_RANGE("cpy wrts to CPU", NVTX_PROF_CPY_GPUS_TO_CPU);
  p = sol;
  while (*p != -1)
  {
    q = p + 1;
    while (*q != -1)
    {
      // modifications are disjoint AND only the modified regions are actually replayed
      if (*q != *p)
      {
        if (*p % HeTM_gshared_data.nbCPUThreads == tid)
        {
#ifdef USE_NVTX
          char prof_msg[128];
          sprintf(prof_msg, "cpy dataset from dev%i to dev%i\n", *q, *p);
#endif /* USE_NVTX */
          NVTX_PUSH_RANGE(prof_msg, NVTX_PROF_CPY_GPUS_TO_CPU);
          // printf("cpy dataset from dev%i to dev%i\n", *q, *p);
          cpyModifications(*p, *q, /* not override */0);
          NVTX_POP_RANGE();
        }
        if (*q % HeTM_gshared_data.nbCPUThreads == tid)
        {
#ifdef USE_NVTX
          char prof_msg[128];
          sprintf(prof_msg, "cpy dataset from dev%i to dev%i\n", *p, *q);
#endif /* USE_NVTX */
          NVTX_PUSH_RANGE(prof_msg, NVTX_PROF_CPY_GPUS_TO_CPU);
          // printf("cpy dataset from dev%i to dev%i\n", *p, *q);
          cpyModifications(*q, *p, /* not override */0);
          NVTX_POP_RANGE();
        }
      }
      // printf("   dev%i committed\n", *p);
      /* if (*p != CPUid && (*p % HeTM_gshared_data.nbCPUThreads == tid)) {
        cpyGPUmodificationsToCPU(*p); // TODO: keep in some cache the GPU modifications
      } */
      q++;
    }
    p++;
  }

  while (*abortingDevs != -1)
  {
    // TODO: need to either augment the write-set of someCommittingDevice or copy all
    if (*abortingDevs % HeTM_gshared_data.nbCPUThreads == tid)
    {
      // printf("DEV%i ABORTED! handled by tid = %i\n", *abortingDevs, tid);
#ifdef USE_NVTX
      char prof_msg[128];
      sprintf(prof_msg, "rollback dev%i with dev%i\n", *abortingDevs, someCommittingDevice);
#endif /* USE_NVTX */
      NVTX_PUSH_RANGE(prof_msg, NVTX_PROF_CPY_GPUS_TO_CPU);
      // overrideCPUmodificationsWithGPUdataset(someCommittingDevice);
      cpyModifications(*abortingDevs, someCommittingDevice, /* override */1);
      NVTX_POP_RANGE();
    }
    abortingDevs++;
  }


  // CPU has correct image
  // for (int i = 0; i < HETM_NB_DEVICES; ++i)
  // {
  //   if (i % HeTM_gshared_data.nbCPUThreads == tid)
  //     broadcastCPUdataset(i);
  // }

  p = sol;
  while (*p != -1)
  {
    if (*p != CPUid && *p % HeTM_gshared_data.nbCPUThreads == tid)
    {
      // HETM_DEB_THRD_CPU("Wait dataset cpy of dev%i", *p);
      Config::GetInstance()->SelDev(*p);
      CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "");
    }
    p++;
  }

  if (tid == 0)
  {
    // Reset the conflict matrix
    for (int k = 0; k < HETM_NB_DEVICES; ++k) {
      memset((void*)HeTM_shared_data[k].mat_confl_GPU_unif, 0, sizeof(char)*((HETM_NB_DEVICES+1)*(HETM_NB_DEVICES+1)));
    }
    memset((void*)HeTM_gshared_data.mat_confl_CPU_unif, 0, sizeof(char)*((HETM_NB_DEVICES+1)*(HETM_NB_DEVICES+1)));
  }

  // accumulateStatistics();
  // NVTX_POP_RANGE(); // NVTX_PROF_CPY_GPUS_TO_CPU
}
