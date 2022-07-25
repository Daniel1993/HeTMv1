#include "hetm-log.h"
#include "hetm.cuh"
#include "pr-stm-wrapper.cuh"
#include "hetm-timer.h"
#include "hetm-cmp-kernels.cuh"
#include "knlman.hpp"
#include "arch.h"

#include "graph.h"
// #include "gsort_lib.h"
#include "bb_lib.h"

#include <list>

using namespace memman;

#define MINIMUM_DtD_CPY 4194304
#define MINIMUM_DtH_CPY 262144
#define MINIMUM_HtD_CPY 262144

pr_tx_args_s HeTM_pr_args[HETM_NB_DEVICES]; // used only by the CUDA-control thread (TODO: PR-STM only)

int isAfterCmpDone = 0;
int isGetStatsDone = 0;
int isGetPRStatsDone = 0;
int isDatasetSyncDone = 0;
int isGPUResetDone = 0;
long roundCountAfterBatch = 0;
int GPU_merger_id = 0;

static int isGPUDetectInterConfl = 0; // used in the main-loop --> TODO: get rid of this

static std::list<HeTM_callback> beforeGPU;
static std::list<HeTM_callback> afterGPU;
static std::list<HeTM_callback> beforeBatch;
static std::list<HeTM_callback> afterBatch;
static std::list<HeTM_callback> beforeKernel;
static std::list<HeTM_callback> afterKernel;
static HeTM_callback choose_policy;

static long lastRoundTXs[HETM_NB_DEVICES+1];

// static inline void waitDatasetEnd();
static void startGPUtoGPUconflictCompare();
static void waitGPUtoGPUconflictCompare();
static int launchInterGPUConflDetectKernel(HeTM_thread_s *threadData, int locGPU, int remGPU);

// executed in other thread
// static void offloadResetGPUState(void*);

// static thread_local HeTM_thread_s *tmp_threadData; // TODO: HETM_OVERLAP_CPY_BACK

static int peerCpyAvailable[HETM_NB_DEVICES*HETM_NB_DEVICES];

void initGPUPeerCpy()
{
  int nGPUs = Config::GetInstance()->NbGPUs();
  int nbOfGPUs = Config::GetInstance()->GetNbPhysicalGPUs();
  CUDA_CHECK_ERROR(cudaGetDeviceCount(&nbOfGPUs), "");
  for (int j = 0; j < nGPUs; ++j) {
    peerCpyAvailable[j*nGPUs + j] = 1;
    for (int k = j+1; k < nGPUs; ++k) {
      int coord1 = j*nGPUs + k, coord2 = k*nGPUs + j;
      int coord1Real = (j%nbOfGPUs)*nGPUs + (k%nbOfGPUs);
      int coord2Real = (k%nbOfGPUs)*nGPUs + (j%nbOfGPUs);
      if (coord1 == coord1Real && coord2 == coord2Real) {
        CUDA_CHECK_ERROR(cudaDeviceCanAccessPeer(&(peerCpyAvailable[coord1]), j, k), "");
        CUDA_CHECK_ERROR(cudaDeviceCanAccessPeer(&(peerCpyAvailable[coord2]), k, j), "");
        if (peerCpyAvailable[coord1]) {
          CUDA_CHECK_ERROR(cudaSetDevice(j), "");
          CUDA_CHECK_ERROR(cudaDeviceEnablePeerAccess(k, 0), "");
        }
        if (peerCpyAvailable[coord2]) {
          CUDA_CHECK_ERROR(cudaSetDevice(k), "");
          CUDA_CHECK_ERROR(cudaDeviceEnablePeerAccess(j, 0), "");
        }
      } else {
        peerCpyAvailable[coord1] = peerCpyAvailable[coord1Real];
        peerCpyAvailable[coord2] = peerCpyAvailable[coord2Real];
      }
    }
  }
}

void destroyGPUPeerCpy()
{
  int nGPUs = Config::GetInstance()->NbGPUs();
  int nbOfGPUs = Config::GetInstance()->GetNbPhysicalGPUs();
  
  for (int j = 0; j < nGPUs; ++j)
  {
    peerCpyAvailable[j*nGPUs + j] = 1;
    for (int k = j+1; k < nGPUs; ++k)
    {
      int coord1 = j*nGPUs + k, coord2 = k*nGPUs + j;
      int coord1Real = (j%nbOfGPUs)*nGPUs + (k%nbOfGPUs);
      int coord2Real = (k%nbOfGPUs)*nGPUs + (j%nbOfGPUs);
      if (coord1 == coord1Real && coord2 == coord2Real)
      {
        if (peerCpyAvailable[coord1])
        {
          CUDA_CHECK_ERROR(cudaSetDevice(j), "");
          CUDA_CHECK_ERROR(cudaDeviceDisablePeerAccess(k), "");
        }
        if (peerCpyAvailable[coord2])
        {
          CUDA_CHECK_ERROR(cudaSetDevice(k), "");
          CUDA_CHECK_ERROR(cudaDeviceDisablePeerAccess(j), "");
        }
      }
    }
  }
}

struct runMultiGPUbatch_
{
  int threadId;
  int devId;
  HeTM_callback callback;
  void *clbkArgs;
};

// TODO: use multiple threads to launch the kernels
static void runMultiGPUbatch(struct runMultiGPUbatch_ *a)
{
  // printf("         <<<<<<<<<< runMultiGPUbatch >>>>>>>>>>\n");
  Config::GetInstance()->SelDev(a->devId);
  a->callback(a->threadId, a->clbkArgs);
}

static struct runMultiGPUbatch_ runMultiGPUbatch_args[HETM_NB_DEVICES];

void runGPUBatch()
{
  // printf("                 <<<<<<<<<< runGPUBatch >>>>>>>>>>\n");
  int threadId = HeTM_thread_data[0]->id;
  HeTM_callback callback = HeTM_thread_data[0]->callback;
  void *clbkArgs = HeTM_thread_data[0]->args;
  if (HeTM_get_GPU_status(0) != HETM_IS_EXIT) {
    for (int j = 0; j < Config::GetInstance()->NbGPUs(); ++j)
    {
      // while (__atomic_load_n(&HeTM_gpu_has_work[j], __ATOMIC_ACQUIRE)
      //   && !HeTM_async_is_stop(0)); // previous work

      runMultiGPUbatch_args[j].threadId = threadId;
      runMultiGPUbatch_args[j].devId = j;
      runMultiGPUbatch_args[j].callback = callback;
      runMultiGPUbatch_args[j].clbkArgs = clbkArgs;

      // TODO: multi thread is not working
      runMultiGPUbatch(&runMultiGPUbatch_args[j]);
    }
  }

  // for (int j = 0; j < HETM_NB_DEVICES; ++j)
  // {
  //   while (__atomic_load_n(&HeTM_gpu_has_work[j], __ATOMIC_ACQUIRE)
  //     && !HeTM_async_is_stop(0));
  // }
}

void runBeforeGPU(int id, void *data)
{
  HeTM_gshared_data.batchCount = 1; // TODO: first batch, this value is used to mark the bitmaps
  for (auto it = beforeGPU.begin(); it != beforeGPU.end(); ++it) {
    HeTM_callback clbk = *it;
    clbk(id, data);
  }
}

void runAfterGPU(int id, void *data)
{
  for (auto it = afterGPU.begin(); it != afterGPU.end(); ++it) {
    HeTM_callback clbk = *it;
    clbk(id, data);
  }
}

void runGPUBeforeBatch(int id, void *data)
{
  for (int devId = 0; devId < Config::GetInstance()->NbGPUs(); ++devId) {
    for (auto it = beforeBatch.begin(); it != beforeBatch.end(); ++it) {
      HeTM_callback clbk = *it;
      clbk(id, data);
    }
  }
}

void runGPUAfterBatch(int id, void *data)
{
  HeTM_gshared_data.batchCount++;
  if ((HeTM_gshared_data.batchCount & 0xff) == 0) {
    HeTM_gshared_data.batchCount++;
  }
  __sync_synchronize();
  for (auto it = afterBatch.begin(); it != afterBatch.end(); ++it) {
    HeTM_callback clbk = *it;
    clbk(id, data);
  }
}

void runGPUBeforeKernel(int id, void *data)
{
  for (auto it = beforeKernel.begin(); it != beforeKernel.end(); ++it) {
    HeTM_callback clbk = *it;
    clbk(id, data);
  }
}

void runGPUAfterKernel(int id, void *data)
{
  for (auto it = afterKernel.begin(); it != afterKernel.end(); ++it)
  {
    HeTM_callback clbk = *it;
    clbk(id, data);
  }

  for (int j = 0; j < Config::GetInstance()->NbGPUs(); ++j)
  {
    Config::GetInstance()->SelDev(j);
    PR_curr_dev = j;
    __atomic_or_fetch(&isGPUDetectInterConfl, HeTM_is_interconflict(j), __ATOMIC_RELEASE);
  }
}

pr_tx_args_s *getPrSTMmetaData(int devId)
{
  return &(HeTM_pr_args[devId]);
}

void waitGPUBatchEnd()
{
  int j;
  for (j = 0; j < Config::GetInstance()->NbGPUs(); ++j)
  {
    Config::GetInstance()->SelDev(j);
    PR_curr_dev = j;
    // auto strm = PR_getCurrentStream();
    if (!HeTM_async_is_stop())
    {
      PR_waitKernel(&HeTM_pr_args[PR_curr_dev]);
      PR_useNextStream(&HeTM_pr_args[PR_curr_dev]);
    }
    // CUDA_CHECK_ERROR(cudaStreamSynchronize(strm), "");
  }
  // Removed deviceSync from here
}

void notifyBatchIsDone()
{
  int j;
  int nGPUs = Config::GetInstance()->NbGPUs();

  uint64_t *cpuRSet_hostptr = (uint64_t*)HeTM_cpu_rset.GetMemObj(0)->host;
  uint64_t *cpuWSet_hostptr = (uint64_t*)HeTM_cpu_wset.GetMemObj(0)->host;
  uint64_t *gpuRSet_devptr  = (uint64_t*)HeTM_gpu_rset.GetMemObj(0)->dev;
  uint64_t *gpuWSet_devptr  = (uint64_t*)HeTM_gpu_wset.GetMemObj(0)->dev;

  /* printf("cpuRset=%lu cpuWset=%lu gpuRset=%p gpuWset=%p\n", *(cpuRSet_hostptr+1), *(cpuWSet_hostptr+1), gpuRSet_devptr, gpuWSet_devptr); */

  for (j = 0; j < nGPUs; j++)
  {
    // size_t cpyrdsetBMAPcache = 0;
    size_t cpywrsetBMAPcache = 0;
    Config::GetInstance()->SelDev(j);
    MemObj *m_wset = HeTM_gpu_wset_cache.GetMemObj(j);
    m_wset->CpyDtH(HeTM_memStream2[j]);
    cpywrsetBMAPcache += m_wset->size;
    MemObj *m_rset = HeTM_gpu_rset_cache.GetMemObj(j);
    m_rset->CpyDtH(HeTM_memStream2[j]);
    cpywrsetBMAPcache += m_rset->size;
    
    __atomic_add_fetch(&HeTM_stats_data.sizeCpyWSetHtD, cpywrsetBMAPcache, __ATOMIC_ACQ_REL);
  }

  // wait the previous copies
  for (j = 0; j < nGPUs; j++)
    CUDA_CHECK_ERROR(cudaStreamSynchronize((cudaStream_t)HeTM_memStream2[j]), "");
  
  for (j = 0; j < nGPUs; j++)
  {
    HeTM_set_GPU_status(j, HETM_BATCH_DONE); // notifies
  }
  if (HeTM_gshared_data.isCPUEnabled)
    startGPUtoGPUconflictCompare();
  __sync_synchronize();
  if (HeTM_gshared_data.isCPUEnabled)
    waitGPUtoGPUconflictCompare();
}

void waitCPUlogValidation(int nonBlock)
{
  int j;
  // waits threads to stop doing validation (VERS)
  // if (!nonBlock)
  //   HeTM_sync_next_batch(); // wait CPU side comparisons
  for (j = 0; j < Config::GetInstance()->NbGPUs(); j++)
  {
    Config::GetInstance()->SelDev(j);
    HETM_DEB_THRD_GPU("GPU %i waits CPU log validation", j);
    do {
      COMPILER_FENCE();
      CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "waiting CPU validation");
    } while ( !nonBlock &&
      __atomic_load_n(&(HeTM_shared_data[j].threadsWaitingSync), __ATOMIC_ACQUIRE)
      < HeTM_gshared_data.nbCPUThreads && !HeTM_is_stop()
    );
  }
}

void syncGPUtoCPUbarrier(int nonBlock)
{
  int j;
  // waits threads to stop doing validation (VERS)

  // TODO: isInterGPUConflDone was waited and set before by the GPU in notifyBatchIsDone
  for (j = 0; j < Config::GetInstance()->NbGPUs(); j++)
  {
    if (!nonBlock)
      HeTM_sync_barrier(j);
  }
}


void waitGPUCMPEnd(int nonBlock)
{
  waitCPUlogValidation(nonBlock);
  syncGPUtoCPUbarrier(nonBlock);
}

void mergeGPUDataset()
{
  // TODO: this function is not merging anything
  RUN_ASYNC(getGPUStatistics, NULL);

  checkIsExit();
  // ---------------------
  // TIMER_READ(t1WCpy);
  RUN_ASYNC(syncGPUdataset, HeTM_thread_data[0]);
  RUN_ASYNC(waitGPUdataset, HeTM_thread_data[0]);
}

void doGPUStateReset()
{
  RUN_ASYNC(offloadResetGPUState, NULL);
}

int HeTM_choose_policy(HeTM_callback req)
{
  choose_policy = req;
  return 0;
}

int HeTM_before_gpu_start(HeTM_callback req)
{
  beforeGPU.push_back(req);
  return 0;
}

int HeTM_after_gpu_finish(HeTM_callback req)
{
  afterGPU.push_back(req);
  return 0;
}

int HeTM_before_batch(HeTM_callback req)
{
  beforeBatch.push_back(req);
  return 0;
}

int HeTM_after_batch(HeTM_callback req)
{
  afterBatch.push_back(req);
  return 0;
}

int HeTM_before_kernel(HeTM_callback req)
{
  beforeKernel.push_back(req);
  return 0;
}

int HeTM_after_kernel(HeTM_callback req)
{
  afterKernel.push_back(req);
  return 0;
}

void hetm_memcpyDeviceToCPU(int devId, HeTM_thread_s *threadData)
{
  size_t datasetCpySize = 0;
  void *devWinner;

  if (devId < Config::GetInstance()->NbGPUs())
  { // some GPU won
    Config::GetInstance()->SelDev(devId);
    MemObj *m = HeTM_mempool.GetMemObj(devId);
    devWinner = m->dev;
    auto strm = PR_getCurrentStream();
    void *host =  m->host;
    datasetCpySize = m->size;

    // copy to CPU
    CUDA_CHECK_ERROR(
      cudaMemcpyAsync(host, devWinner, datasetCpySize, cudaMemcpyDeviceToHost, strm),
      "");

    HeTM_stats_data.sizeCpyDataset += datasetCpySize;
  }
}

static void mergeAllMatrices()
{
  int nGPUs = Config::GetInstance()->NbGPUs();

  for (int j = 0; j < (nGPUs+1); ++j)
  {
    for (int i = 0; i < (nGPUs+1); ++i) // CPU validation is on GPUs
    {
      if (i == j) continue;
      int coord_l = (nGPUs+1)*i + j; // row
      int coord_c = (nGPUs+1)*j + i; // column
      int coord;
      unsigned char GPUval;
      if (j == nGPUs)
        GPUval = __atomic_load_n(&(HeTM_shared_data[i].mat_confl_GPU_unif[coord = coord_c]), __ATOMIC_ACQUIRE);
      else if (i == nGPUs)
        GPUval = __atomic_load_n(&(HeTM_shared_data[j].mat_confl_GPU_unif[coord = coord_c]), __ATOMIC_ACQUIRE);
      else
        GPUval = __atomic_load_n(&(HeTM_shared_data[i].mat_confl_GPU_unif[coord = coord_l]), __ATOMIC_ACQUIRE);
      HeTM_gshared_data.mat_confl_CPU_unif[coord] |= GPUval;
      // printf("mergeAllMatrices[%i][%i] = %i\n", i, j, (int)HeTM_gshared_data.mat_confl_CPU_unif[coord_l]);
    }
  }
}

#ifdef HETM_DEB
static void printMatrixPerGPU()
{
  int nGPUs = Config::GetInstance()->NbGPUs();

  HETM_DEB_THRD_GPU("\n Conflict matrices:");
  for (int j = 0; j < nGPUs+1; ++j) {
    if (j == nGPUs) {
      printf("       >>> CPU  \n   \\ ");
    }
    else {
      printf("       >>> GPU %i \n   \\ ", j);
    }
    for (int k = 0; k < nGPUs+1; ++k) {
      if (k == nGPUs) {
        printf(" CPU ");
      } 
      else {
        printf(" GPU%i", k);
      }
    }
    for (int k = 0; k < nGPUs+1; ++k) {
      if (k == nGPUs) { printf("\n CPU "); }
      else { printf("\n GPU%i", k); }
      if (j == nGPUs)
      {
        for (int l = 0; l < nGPUs+1; ++l) {
          int coord1 = (nGPUs+1)*k + l;
          printf("   %i ", (int)HeTM_gshared_data.mat_confl_CPU_unif[coord1]);
        }
      }
      else 
      {
        for (int l = 0; l < nGPUs+1; ++l) {
          int coord1 = (nGPUs+1)*k + l;
          printf("   %i ", (int)HeTM_shared_data[j].mat_confl_GPU_unif[coord1]);
        }
      }
    }
    printf("\n");
  }
  printf("\n");
}
#endif

static void
startGPUtoGPUconflictCompare()
{
  // this is done serially
  // static unsigned char last_batch = -1;
  // unsigned char batch = (unsigned char) HeTM_gshared_data.batchCount;
  int nGPUs = Config::GetInstance()->NbGPUs();

  // if (last_batch == batch)
  //   return;
  // last_batch = batch;

  // HETM_DEB_THRD_GPU("\033[0;31m" "Thread %i batch(%i) != lastBatch(%i)" "\033[0m",
  //   devId, batch, lastBatch);

  if (!(nGPUs > 1)) // only 1 GPU does not do inter-GPU conflict detection
    __atomic_store_n(&(HeTM_shared_data[0].isInterGPUConflDone), 1, __ATOMIC_RELEASE);

  for (int i = 0; i < nGPUs; ++i)
  {
    for (int j = i+1; j < nGPUs; ++j)
    {
      size_t cpywrsetBMAP = 0;
      auto threadDataj = &(HeTM_shared_data[j].threadsInfo[0]);
      auto threadDatak = &(HeTM_shared_data[i].threadsInfo[0]);

      int devSrc = i;
      int devDst = j;

      MemObj *m_dst_remote_wset = HeTM_gpu_wset_ext[devSrc].GetMemObj(devDst);
      MemObj *m_src_remote_wset = HeTM_gpu_wset_ext[devDst].GetMemObj(devSrc);
      MemObj *m_src_wset = HeTM_gpu_wset.GetMemObj(devSrc);
      MemObj *m_src_wset_cache = HeTM_gpu_wset_cache.GetMemObj(devSrc);
      MemObj *m_src_wset_filter = m_src_wset;
      MemObj *m_dst_wset = HeTM_gpu_wset.GetMemObj(devDst);
      MemObj *m_dst_wset_cache = HeTM_gpu_wset_cache.GetMemObj(devDst);
      MemObj *m_dst_wset_filter = m_dst_wset;


      MemObjCpyBuilder bSRC;
      MemObjCpyDtD mSRC(bSRC // know how to do PtP also
        .SetGranFilter(8) // TODO
        ->SetGranApply(sizeof(char))
        ->SetForceFilter(MEMMAN_ONLY_COLLISION)
        ->SetFilterVal(HeTM_gshared_data.batchCount)
        ->SetCache(m_src_wset_cache)
        ->SetCache2(m_dst_wset_cache)
        ->SetFilter(m_src_wset_filter)
        ->SetDst(m_dst_remote_wset)
        ->SetSrc(m_src_wset)
#ifdef BMAP_ENC_1BIT
        ->SetSizeChunk((BMAP_GRAN >> LOG2_32BITS)*sizeof(unsigned))
#else
        ->SetSizeChunk(BMAP_GRAN)
#endif /* BMAP_ENC_1BIT */
        ->SetStrm1(HeTM_memStream[devDst])
        ->SetStrm2(HeTM_memStream[devDst])
      );

      MemObjCpyBuilder bDST;
      MemObjCpyDtD mDST(bDST // know how to do PtP also
        .SetGranFilter(8) // TODO
        ->SetGranApply(sizeof(char))
        ->SetForceFilter(MEMMAN_ONLY_COLLISION)
        ->SetFilterVal(HeTM_gshared_data.batchCount)
        ->SetCache(m_dst_wset_cache)
        ->SetCache2(m_src_wset_cache)
        ->SetFilter(m_dst_wset_filter)
        ->SetDst(m_src_remote_wset)
        ->SetSrc(m_dst_wset)
#ifdef BMAP_ENC_1BIT
        ->SetSizeChunk((BMAP_GRAN >> LOG2_32BITS)*sizeof(unsigned))
#else
        ->SetSizeChunk(BMAP_GRAN)
#endif /* BMAP_ENC_1BIT */
        ->SetStrm1(HeTM_memStream2[devSrc])
        ->SetStrm2(HeTM_memStream2[devSrc])
      );

      cpywrsetBMAP += mSRC.Cpy();
      cpywrsetBMAP += mDST.Cpy();

      __atomic_add_fetch(&HeTM_stats_data.sizeCpyWSetHtD, cpywrsetBMAP, __ATOMIC_ACQ_REL);

      launchInterGPUConflDetectKernel(threadDataj, j, i);
      launchInterGPUConflDetectKernel(threadDatak, i, j);
    }
  }
}

static void waitGPUtoGPUconflictCompare()
{
  for (int j = 0; j < Config::GetInstance()->NbGPUs(); ++j)
  {
    Config::GetInstance()->SelDev(j);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "waiting cross GPU validation");
    HETM_DEB_THRD_GPU("isInterGPUConflDone[%d] = 1\n", j);
    __atomic_store_n(&(HeTM_shared_data[j].isInterGPUConflDone), 1, __ATOMIC_RELEASE);
  }
}

static int
launchInterGPUConflDetectKernel(
  HeTM_thread_s *threadData,
  int locGPU,
  int remGPU
) {
  // TODO: add early validation kernel
  // if (!doApply) { return 0; }

  // for (int j = 0; j < HETM_NB_DEVICES; ++j) {
  long nbGrans = HeTM_gshared_data.sizeMemPool / PR_LOCK_GRANULARITY;
  Config::GetInstance()->SelDev(locGPU);
  int nbThreadsX = 256;
  int bo = (nbGrans + nbThreadsX-1) / (nbThreadsX);

  // Memory region of the entry object
  // printf("dev = %i otherDev = %i batchCount = %li\n", locGPU, remGPU, HeTM_gshared_data.batchCount);
  // thread_local static HeTM_cmp_s checkTxCompressed_args;
  MemObj *m = HeTM_interGPUConflDetect->entryObj->GetMemObj(locGPU);
  HeTM_cmp_s *checkTxCompressed_args = (HeTM_cmp_s*)m->host;
  checkTxCompressed_args->knlArgs.knlGlobal  = *(HeTM_get_global_arg(locGPU));
  checkTxCompressed_args->knlArgs.nbOfGPUs   = Config::GetInstance()->NbGPUs();
  checkTxCompressed_args->knlArgs.sizeWSet   = (int)nbGrans;
  checkTxCompressed_args->knlArgs.sizeRSet   = (int)HeTM_gshared_data.rsetLogSize;
  // checkTxCompressed_args->knlArgs.idCPUThr   = (int)threadData->id;
  checkTxCompressed_args->knlArgs.batchCount = (unsigned char) HeTM_gshared_data.batchCount;
  checkTxCompressed_args->clbkArgs = threadData; // TODO: pass some structure with the CUDA events for GPU<->GPU comparison
  checkTxCompressed_args->knlArgs.devId = locGPU;
  checkTxCompressed_args->knlArgs.otherDevId = remGPU;

  HETM_DEB_THRD_GPU("\033[0;36m" "GPU %i HeTM_interGPUConflDetect (to check against GPU %i)" "\033[0m" ,
    locGPU, remGPU);

  m->CpyHtD(HeTM_memStream[locGPU]);
  HeTM_interGPUConflDetect->blocks  = (knlman_dim3_s){ .x = bo,         .y = 1, .z = 1 };
  HeTM_interGPUConflDetect->threads = (knlman_dim3_s){ .x = nbThreadsX, .y = 1, .z = 1 };
  CUDA_CHECK_ERROR(cudaStreamSynchronize((cudaStream_t)HeTM_memStream2[locGPU]), "");
  HeTM_interGPUConflDetect->Run(locGPU, HeTM_memStream[locGPU]);
  // /* REMOVE ME */CUDA_CHECK_ERROR(cudaStreamSynchronize((cudaStream_t)HeTM_memStream[locGPU]), "");

  return 0;
}

void
mergeMatricesAndRunFVS(
  int nonBlock
) {
  int nbGPUs = Config::GetInstance()->NbGPUs();
  // TODO: matrices use Unified Memory (1B * (NB_GPU+1*CPU)**2), 9B for 2 GPUs (is it worth the extra code?)
  // HETM_DEB_THRD_GPU(" -0- GPU waits CPU side comparisons \n");
  if (!nonBlock)
    HeTM_sync_next_batch(); // wait CPU side comparisons
  mergeAllMatrices();

#ifdef HETM_DEB
  printMatrixPerGPU();
#endif

  // Get if any conflicts (compute feedback vertex set)
  graph G = fromSquareMat(nbGPUs+1,
    (unsigned char*)HeTM_gshared_data.mat_confl_CPU_unif);
  BB_reset(G);

  // set weights here
  for (int i = 0; i < nbGPUs+1; i++)
  {
    if (i == nbGPUs)
      printf("CPU did %li TXs\n", lastRoundTXs[i]);
    else
      printf("GPU did %li TXs\n", lastRoundTXs[i]);
  }

  for (int i = 0; i < nbGPUs+1; i++)
    HeTM_gshared_data.dev_weights[i] = lastRoundTXs[i];
  BB_setWeights(HeTM_gshared_data.dev_weights);

  BB_run();
  // HETM_DEB_THRD_GPU(" -1- GPU notifies CPU of BB done \n");
  if (!nonBlock)
    HeTM_sync_BB(); // notify CPU to proceed
}
/* 
static void memman_cpy_to_cpu_fn(void *hostPtr, void *devPtr, size_t cpySize)
{
  CUDA_CPY_TO_HOST_ASYNC(hostPtr, devPtr, cpySize, (cudaStream_t)HeTM_memStream[PR_curr_dev]);
}

static void memman_cpy_to_gpu_fn(void *hostPtr, void *devPtr, size_t cpySize)
{
  CUDA_CPY_TO_DEV_ASYNC(devPtr, hostPtr, cpySize, (cudaStream_t)HeTM_memStream[PR_curr_dev]);
}
 */
void accumulateStatistics()
{
  // WAIT_ON_FLAG(isGetStatsDone); // TODO
  int anyAbort = 0;
  int nGPUs = Config::GetInstance()->NbGPUs();

  if (HeTM_gshared_data.isCPUEnabled && HeTM_gshared_data.isGPUEnabled)
  {
    int *sol = BB_getBestSolution();
    int *toRemove = BB_getFVS();
    const int CPUid = nGPUs;

    int *p = sol;
    while (*p != -1)
    {
      if (*p == CPUid)
        HeTM_stats_data.nbCommittedTxsCPU += lastRoundTXs[*p];
      else // TODO: transform this stat in an array
        HeTM_stats_data.nbCommittedTxsGPU += lastRoundTXs[*p];
      p++;
    }

    p = toRemove;
    while (*p != -1)
    {
      anyAbort = 1;
      if (*p == CPUid)
        HeTM_stats_data.nbDroppedTxsCPU += lastRoundTXs[*p];
      else // TODO: transform this stat in an array
        HeTM_stats_data.nbDroppedTxsGPU += lastRoundTXs[*p];
      p++;
    }
  }
  else
  {
    for (int j = 0; j < nGPUs; ++j)
      HeTM_stats_data.nbCommittedTxsGPU += lastRoundTXs[j];
  }

  HeTM_stats_data.nbBatches++;
  if (anyAbort) {
    HeTM_stats_data.nbBatchesFail++;
  } else {
    // printf("HeTM_stats_data.nbBatchesSuccess = %d\n", HeTM_stats_data.nbBatchesSuccess);
    HeTM_stats_data.nbBatchesSuccess++;
  }

  for (int j = 0; j < nGPUs; ++j)
  {
    PR_curr_dev = j;
    PR_checkpointAbortsCommits();
  }
}

void syncGPUdataset(void *args)
{
  HETM_DEB_THRD_GPU("Syncing dataset ...");

  NVTX_PUSH_RANGE("wait in GPU", NVTX_PROF_GPU_WAITS_NEXT_BATCH);
  // TODO: need to wait CPU validation
  mergeMatricesAndRunFVS(0);
  NVTX_POP_RANGE(); // NVTX_PROF_GPU_WAITS_NEXT_BATCH

  // printf("conf mat:\n");
  // for (int j = 0; j < HETM_NB_DEVICES+1; ++j) {
  //   for (int k = 0; k < HETM_NB_DEVICES+1; ++k) {
  //     printf("%d", (int) HeTM_gshared_data.mat_confl_CPU_unif[(HETM_NB_DEVICES+1)*j+k]);
  //   }
  //   printf("\n");
  // }
}

void waitGPUdataset(void *args)
{
  // HeTM_thread_s *threadData = (HeTM_thread_s*)args;

  for (int i = 0; i < Config::GetInstance()->NbGPUs(); ++i)
  {
    Config::GetInstance()->SelDev(i);
    
    // cudaStreamSynchronize((cudaStream_t)HeTM_memStream[i]);
    // cudaStreamSynchronize((cudaStream_t)HeTM_memStream2[i]);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize(), ""); // TODO: now I'm waiting this to complete
  }

  // TODO: now it is crashing
  // CUDA_EVENT_SYNCHRONIZE(threadData->cpyDatasetStartEvent);
  // CUDA_EVENT_SYNCHRONIZE(threadData->cpyDatasetStopEvent);

  // CUDA_EVENT_ELAPSED_TIME(&threadData->timeCpyDataset, threadData->cpyDatasetStartEvent,
  //   threadData->cpyDatasetStopEvent);
  // threadData->timeCpyDatasetSum += threadData->timeCpyDataset;
  __atomic_store_n(&isDatasetSyncDone, 1, __ATOMIC_RELEASE);
}

void getGPUPRSTMStats(void *argPtr)
{
  // TODO: this is empty!!!
  // HeTM_thread_s *threadData = (HeTM_thread_s*)argPtr;

  // PR_retrieveIO(&HeTM_pr_args);
  // threadData->curNbTxs = PR_nbCommits;
  // HeTM_stats_data.nbAbortsGPU += PR_nbAborts;
  // PR_resetStatistics(&HeTM_pr_args);
  __atomic_store_n(&isGetPRStatsDone, 1, __ATOMIC_RELEASE);
}

void getGPUStatistics(void *arg)
{
  int nGPUs = Config::GetInstance()->NbGPUs();
  long committedTxsCPUBatch = 0;
  long txsNonBlocking = 0;
  // long droppedTxsCPUBatch = 0; // TODO: in apply dataset;
  // int idGPUThread = HeTM_shared_data.nbCPUThreads; // the last one

  for (int i = 0; i < HeTM_gshared_data.nbCPUThreads; ++i) {
    committedTxsCPUBatch += __atomic_load_n(&(HeTM_shared_data[0].threadsInfo[i].curNbTxs), __ATOMIC_ACQUIRE);
    txsNonBlocking += __atomic_load_n(&(HeTM_shared_data[0].threadsInfo[i].curNbTxsNonBlocking), __ATOMIC_ACQUIRE);
    __atomic_store_n(&(HeTM_shared_data[0].threadsInfo[i].curNbTxs), 0, __ATOMIC_RELEASE);
    __atomic_store_n(&(HeTM_shared_data[0].threadsInfo[i].curNbTxsNonBlocking), 0, __ATOMIC_RELEASE);
  }

  lastRoundTXs[nGPUs] = committedTxsCPUBatch + txsNonBlocking;

  // TODO: not doing anythings
  choose_policy(0, arg); // choose the policy for the next batch

  HeTM_stats_data.nbTxsCPU += /*droppedTxsCPUBatch + */committedTxsCPUBatch;
  HeTM_stats_data.txsNonBlocking += txsNonBlocking; // assert == 0

  // TODO: add committed and/or dropped TXs (lastRoundTXs[nGPUs]) after resolving conflicts 
  // HeTM_stats_data.nbCommittedTxsCPU += committedTxsCPUBatch;
  // HeTM_stats_data.nbDroppedTxsCPU   += droppedTxsCPUBatch;

  PR_global_data_s *d;
  for (int j = 0; j < nGPUs; j++)
  {
    PR_curr_dev = j;
    d = &(PR_global[j]);
    HeTM_stats_data.nbTxsGPU += d->PR_nbCommitsSinceCheckpoint;
    HeTM_stats_data.nbTxsPerGPU[j] += d->PR_nbCommitsSinceCheckpoint;
    // printf("    dev%i: +%li TXs (%li total)\n", j, d->PR_nbCommitsSinceCheckpoint, HeTM_stats_data.nbTxsPerGPU[j]);
    HeTM_stats_data.nbAbortsGPU += d->PR_nbAbortsSinceCheckpoint;
    HeTM_stats_data.nbAbortsPerGPU[j] += d->PR_nbAbortsSinceCheckpoint;
    lastRoundTXs[j] = d->PR_nbCommitsSinceCheckpoint;
  }

  // printf(" >>> Update stats <<<\n");
  __atomic_store_n(&isGetStatsDone, 1, __ATOMIC_RELEASE);
}

void offloadResetGPUState(void*)
{
  HeTM_reset_CPU_state(roundCountAfterBatch); // TODO: move to CPU (handle multi threads)
  HeTM_reset_GPU_state(roundCountAfterBatch); // flags/locks
  __atomic_store_n(&isGPUResetDone, 1, __ATOMIC_RELEASE);
}

void setCountAfterBatch()
{
  roundCountAfterBatch = *hetm_batchCount;
}

void runPrSTMCallback(int nbBlcks, int nbThrsPerBlck, void(*callback)(PR_globalKernelArgs), void* inPtr, int sizeIn, void* outPtr, int sizeOut)
{

  for (int j = 0; j < Config::GetInstance()->NbGPUs(); ++j)
  {
		pr_buffer_s inBuf, outBuf;

    // TODO: change PR-STM to use knlman
    Config::GetInstance(j)->SelDev(j);
		PR_curr_dev = j;

    PR_global_data_s *d = &(PR_global[PR_curr_dev]);
    
    d->PR_blockNum = nbBlcks;
    d->PR_threadNum = nbThrsPerBlck;

    inBuf.buf = inPtr;
    inBuf.size = sizeIn;
    outBuf.buf = outPtr;
    outBuf.size = sizeOut;
    PR_prepareIO(&HeTM_pr_args[j], inBuf, outBuf);

    PR_run(callback, &HeTM_pr_args[j]);

    // PR_i_cudaPrepare((&HeTM_pr_args[j]), bankTx);
    // PR_BEFORE_RUN_EXT((&HeTM_pr_args[j]));
    // // PR_i_run(pr_args);
    // bankTx<<<PR_blockNum,PR_threadNum,0,(cudaStream_t)PR_getCurrentStream()>>>(HeTM_pr_args[j].dev);
    // PR_AFTER_RUN_EXT((&HeTM_pr_args[j]));

    // CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "");
  }
}
