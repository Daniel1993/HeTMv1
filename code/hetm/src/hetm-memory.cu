
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <map>
#include <curand_kernel.h>

#include "hetm-log.h"
#include "memman.hpp"
#include "knlman.hpp"
#include "pr-stm-wrapper.cuh"
#include "hetm-cmp-kernels.cuh"

#include "rdtsc.h"
#include "hetm.cuh"


using namespace std;
using namespace memman;

static map<void*, size_t> alloced;
static map<void*, size_t> freed;
static size_t curSize[HETM_NB_DEVICES];

// static size_t bitmapGran = BMAP_GRAN;
// static size_t bitmapGranBits = BMAP_GRAN_BITS;

// static void CUDART_CB cpyCallback(cudaStream_t event, cudaError_t status, void *data);

MemObjOnDev HeTM_mempool;

MemObjOnDev HeTM_gpu_rset;
MemObjOnDev HeTM_gpu_wset;
MemObjOnDev HeTM_gpu_rset_cache;
MemObjOnDev HeTM_gpu_wset_cache;

MemObjOnDev HeTM_cpu_rset;
MemObjOnDev HeTM_cpu_wset;
MemObjOnDev HeTM_cpu_rset_cache;
MemObjOnDev HeTM_cpu_wset_cache;

MemObjOnDev *HeTM_gpu_wset_ext = nullptr;

MemObjOnDev HeTM_curand_state;

MemObjOnDev HeTM_gpu_confl_mat;
MemObjOnDev HeTM_gpu_confl_mat_merge;


static void init_mempool(size_t pool_size, int opts);
static void init_RSetWSet(int devId, size_t pool_size);
static void init_multi_GPU(int devId, size_t pool_size);
static void init_bmap(int devId, size_t pool_size);

int HeTM_mempool_init(size_t pool_size, int opts)
{
  size_t nbGranules = pool_size / PR_LOCK_GRANULARITY;
  size_t nbChunks;
  int nbGPUs = HETM_NB_DEVICES;
  Config::GetInstance(nbGPUs, CHUNK_GRAN);

  for (int j = 0; j < nbGPUs; ++j)
  {
    Config::GetInstance()->SelDev(j);
    init_multi_GPU(j, pool_size);
  }

  init_mempool(pool_size, opts);

  for (int j = 0; j < nbGPUs; ++j)
  {
    Config::GetInstance()->SelDev(j);
    init_bmap(j, pool_size);
  }

  for (int j = 0; j < nbGPUs; ++j)
  {
    Config::GetInstance()->SelDev(j);
    init_RSetWSet(j, pool_size);

    nbChunks = pool_size / BMAP_GRAN;
    if (nbGranules % BMAP_GRAN > 0) nbChunks++;
    HeTM_gshared_data.nbChunks = nbChunks;

    // printf(" <<<<<<< HeTM_shared_data.devMemPoolBackupBmap = %p\n",  HeTM_shared_data.devMemPoolBackupBmap);
    // memman_select("HeTM_mempool");
    // memman_bmap_s *mainBMap = (memman_bmap_s*) memman_get_bmap(NULL);
    HeTM_knl_global_s args;
    args.devMemPoolBasePtr  = HeTM_shared_data[j].mempool_devptr;
    // args.devMemPoolBackupBasePtr = HeTM_shared_data[j].devMemPoolBackup;
    // args.devMemPoolBackupBmap = ((memman_bmap_s*)HeTM_shared_data[j].devMemPoolBackupBmap)->dev;
    args.hostMemPoolBasePtr  = HeTM_shared_data[j].mempool_hostptr;
    args.isInterConfl        = HeTM_shared_data[j].devInterConflFlag;
    args.explicitLogBlock    = HeTM_get_explicit_log_block_size();
    args.nbGranules          = nbGranules; // TODO: granules
    args.devRSet             = HeTM_shared_data[j].bmap_rset_GPU_devptr;
    args.devRSetCache_hostptr = HeTM_shared_data[j].rsetGPUcache_hostptr;
    args.devWSetCache         = HeTM_shared_data[j].wsetGPUcache;
    args.devWSetCache_hostptr = HeTM_gpu_wset_cache.GetMemObj(j)->host;
    for (int k = 0; k < nbGPUs; ++k)
    {
      args.devWSet[k]        = HeTM_shared_data[j].bmap_wset_GPU_devptr[k];
      // TODO: this must go into GPU memory   
      args.localConflMatrix[k] = (char*)(HeTM_shared_data[k].mat_confl_GPU_unif);
    }
    args.hostRSet            = HeTM_shared_data[j].bmap_rset_CPU_devptr;
    args.hostRSetCache_hostptr = HeTM_shared_data[j].bmap_cache_rset_CPU_hostptr;

    // TODO: add merged WSet to reduce copies
    args.mergedWSet          = HeTM_shared_data[j].bmap_merged_wset_devptr;
    args.mergedWSetCache     = HeTM_shared_data[j].bmap_cache_merged_wset_devptr;

    args.hostWSet            = HeTM_shared_data[j].bmap_wset_CPU_hostptr;
    args.hostWSetCache       = HeTM_shared_data[j].bmap_cache_wset_CPU_devptr;
    args.hostWSetCache_hostptr = HeTM_shared_data[j].bmap_cache_wset_CPU_hostptr;
    args.hostWSetCacheConfl  = HeTM_shared_data[j].bmap_cache_wset_CPUConfl; // TODO: rename host to dev
    // args.hostWSetCacheConfl2 = HeTM_shared_data[j].bmap_cache_wset_CPUConfl2;
    args.hostWSetCacheConfl3 = HeTM_shared_data[j].bmap_cache_wset_CPUConfl3;
    args.hostWSetCacheSize   = nbChunks;
    args.hostWSetCacheBits   = HeTM_shared_data[j].bmap_cache_wset_CPUBits;
    args.hostWSetChunks      = nbChunks;
    args.randState           = HeTM_shared_data[j].devCurandState;
    args.isGPUOnly           = (HeTM_gshared_data.isCPUEnabled == 0);
    // args.GPUwsBmap           = mainBMap->dev;

    HeTM_set_global_arg(j, args);
    curSize[j] = 0;
  }

  return 0; // TODO: check for error
}

int HeTM_mempool_destroy(int devId)
{ // TODO: free's are bogus
  // cudaFree(HeTM_mempool.GetMemObj(devId)->dev);
  // // if (devId == 0)
  // //   free(HeTM_mempool.GetMemObj(devId)->host);
  // // delete HeTM_mempool.GetMemObj(devId);

  // cudaFree(HeTM_gpuLog.GetMemObj(devId)->dev);
  // // free(HeTM_gpuLog.GetMemObj(devId)->host);
  // // delete HeTM_gpuLog.GetMemObj(devId);

  // cudaFree(HeTM_gpu_rset.GetMemObj(devId)->dev);
  // // cudaFree(HeTM_gpu_rset.GetMemObj(devId)->host);
  // // delete HeTM_gpu_rset.GetMemObj(devId);

  // cudaFree(HeTM_gpu_rset_cache.GetMemObj(devId)->dev);
  // // free(HeTM_gpu_rset_cache.GetMemObj(devId)->host);
  // // delete HeTM_gpu_rset_cache.GetMemObj(devId);

  // cudaFree(HeTM_cpu_rset.GetMemObj(devId)->dev);
  // // free(HeTM_cpu_rset.GetMemObj(devId)->host);
  // // delete HeTM_cpu_rset.GetMemObj(devId);

  // cudaFree(HeTM_cpu_wset.GetMemObj(devId)->dev);
  // // free(HeTM_cpu_wset.GetMemObj(devId)->host);
  // // delete HeTM_cpu_wset.GetMemObj(devId);

  // // cudaFree(HeTM_cpu_wset_cache.GetMemObj(devId)->dev);
  // // free(HeTM_cpu_wset_cache.GetMemObj(devId)->host);
  // // delete HeTM_cpu_wset_cache.GetMemObj(devId);

  // for (int i = 0; i < HETM_NB_DEVICES; ++i) {
  //   if (devId != i && HeTM_shared_data[devId].bmap_wset_GPU_devptr[i]) {
  //     cudaFree(HeTM_shared_data[devId].bmap_wset_GPU_devptr[i]);
  //     HeTM_shared_data[devId].bmap_wset_GPU_devptr[i] = NULL;
  //   }
  // }
  return 0;
}

void HeTM_initCurandState(int devId)
{
  Config::GetInstance()->SelDev(devId);
  int nbThreads = HeTM_gshared_data.nbGPUThreads;
  int nbBlocks = HeTM_gshared_data.nbGPUBlocks;
  size_t size = nbThreads * nbBlocks * sizeof(long); // TODO: from sizeof(curandState)

  MemObjBuilder b;
  MemObj *m = new MemObj(
    b.SetSize(size)
    ->SetOptions(0)
    ->AllocDevPtr(),
    devId
  );
  HeTM_curand_state.AddMemObj(m);

  HeTM_shared_data[devId].devCurandState = m->dev;
  HeTM_setupCurand<<<nbBlocks, nbThreads>>>(HeTM_shared_data[devId].devCurandState);
  CUDA_CHECK_ERROR(cudaDeviceSynchronize(), ""); // TODO: blocks
}

void HeTM_destroyCurandState(int devId)
{
  Config::GetInstance()->SelDev(devId);
  MemObj *m = HeTM_curand_state.GetMemObj(devId);
  cudaFree(m->dev);
  delete m;
}

int HeTM_mempool_cpy_to_cpu(int devId, size_t *copiedSize, long batchCount)
{
  Config::GetInstance()->SelDev(devId);
#ifndef USE_UNIF_MEM
  HeTM_mempool.GetMemObj(devId)->CpyDtH();
#endif /* USE_UNIF_MEM */
  return 0; // TODO: error code
}

int HeTM_mempool_cpy_to_gpu(int devId, size_t *copiedSize, long batchCount)
{
  Config::GetInstance()->SelDev(devId);
#ifndef USE_UNIF_MEM
  HeTM_mempool.GetMemObj(devId)->CpyHtD();
#endif /* USE_UNIF_MEM */
  return 0; // TODO: error code
}

int HeTM_alloc(int devId, void **cpu_ptr, void **gpu_ptr, size_t size)
{
  Config::GetInstance()->SelDev(devId);
  size_t newSize = curSize[devId] + size;
  if (newSize > HeTM_gshared_data.sizeMemPool)
    return -1;

  // there is still space left
  char *curPtrHost = (char*)HeTM_shared_data[devId].mempool_hostptr;
  char *curPtrDev  = (char*)HeTM_shared_data[devId].mempool_devptr;
  curPtrHost += curSize[devId];
  curPtrDev  += curSize[devId];
  if (cpu_ptr) *cpu_ptr = (void*)curPtrHost;
  if (gpu_ptr) *gpu_ptr = (void*)curPtrDev;
  curSize[devId] = newSize;
  alloced.insert(make_pair(*cpu_ptr, size));

  return 0;
}

int HeTM_free(int devId, void **cpu_ptr)
{
  // TODO:
  auto it = alloced.find(*cpu_ptr);
  if (it == alloced.end()) {
    return -1; // not found
  }
  freed.insert(make_pair(*cpu_ptr, it->second));
  alloced.erase(it);
  return 0;
}

void* HeTM_map_addr_to_gpu(int devId, void *origin)
{
  int hostDev = 0; // always use the one mapped to device 0 as default
  uintptr_t o = (uintptr_t)origin;
  uintptr_t host = (uintptr_t)HeTM_shared_data[hostDev].mempool_hostptr;
  uintptr_t dev  = (uintptr_t)HeTM_shared_data[devId].mempool_devptr;
  return (void*)(o - host + dev);
}

void* HeTM_map_cpu_to_cpu(int devId, void *origin)
{
  uintptr_t o = (uintptr_t)origin;
  uintptr_t host = (uintptr_t)HeTM_shared_data[devId].mempool_hostptr;
  uintptr_t dev  = (uintptr_t)HeTM_shared_data[devId].mempool_devptr;
  return (void*)(o - dev + host);
}

int HeTM_reset_GPU_state(long batchCount)
{
  // size_t s;
  // CUDA_CHECK_ERROR(cudaMemset(PR_lockTableDev, 0, PR_LOCK_TABLE_SIZE*sizeof(int)), "");
  for (int i = 0; i < HETM_NB_DEVICES; ++i)
  {
    if ((batchCount & 0xff) == 0xff)
    { // at some round --> reset
      Config::GetInstance()->SelDev(i);
      HeTM_gpu_rset.GetMemObj(i)->ZeroDev(HeTM_memStream[i]);
      HeTM_gpu_rset_cache.GetMemObj(i)->ZeroDev(HeTM_memStream[i]);
      HeTM_gpu_wset.GetMemObj(i)->ZeroDev(HeTM_memStream[i]);
      HeTM_gpu_wset_cache.GetMemObj(i)->ZeroDev(HeTM_memStream[i]);
    }
  }
  for (int i = 0; i < HETM_NB_DEVICES; ++i)
  {
    Config::GetInstance()->SelDev(i);
    CUDA_CHECK_ERROR(cudaStreamSynchronize((cudaStream_t)HeTM_memStream[i]), "");
  }
  return 0;
}
/* 
static void CUDART_CB cpyCallback(cudaStream_t stream, cudaError_t status, void *data)
{
  HeTM_thread_s *threadData = (HeTM_thread_s*)data;

  if(status != cudaSuccess) { // TODO: Handle error
    printf("CMP failed! >>> %s <<<\n", cudaGetErrorString(status));
    // TODO: exit application
  }
  __atomic_store_n(&(threadData->isCpyDone), 1, __ATOMIC_RELEASE);
} */

static void init_mempool(size_t pool_size, int memOpts)
{
  // size_t granBmap;
  // memman_bmap_s *mainBMap[HETM_NB_DEVICES];
  // memman_bmap_s *backupBMap[HETM_NB_DEVICES];

  int mempoolOpts = 
#ifdef USE_UNIF_MEM
      MEMMAN_UNIF
#else /* !USE_UNIF_MEM */
      memOpts
#endif
  ;

  for (int j = 0; j < Config::GetInstance()->NbGPUs(); ++j)
  {
    Config::GetInstance()->SelDev(j);

    MemObjBuilder b;
    b.SetSize(pool_size)
    ->SetOptions(mempoolOpts)
    ->AllocDevPtr();

    if (j == 0)
      b.AllocHostPtr();
    else
      b.SetHostPtr(HeTM_shared_data[0].mempool_hostptr);

    MemObj *m = new MemObj(&b, j);
    HeTM_mempool.AddMemObj(m);

    HeTM_shared_data[j].mempool_devptr  = m->dev;
    HeTM_shared_data[j].mempool_hostptr = m->host;

    stm_baseMemPool[j] = HeTM_shared_data[j].mempool_hostptr;
    // stm_devMemPoolBmap[j] = mainBMap[j];
    // stm_devMemPoolBackupBmap[j] = backupBMap[j];
  }
}

static void init_RSetWSet(int devId, size_t pool_size)
{
  size_t sizeRSetLog = 0;
  Config::GetInstance()->SelDev(devId);

  sizeRSetLog = pool_size / PR_LOCK_GRANULARITY;
  size_t cacheSize = (sizeRSetLog + (CACHE_GRANULE_SIZE-1)) / CACHE_GRANULE_SIZE;

  MemObjBuilder b_cpu_rset;
  MemObjBuilder b_cpu_rset_cache;
  MemObjBuilder b_gpu_rset;

  MemObj *m_cpu_rset;
  MemObj *m_cpu_rset_cache;
  MemObj *m_gpu_rset;

  b_cpu_rset
  .SetSize(sizeRSetLog)
  ->SetOptions(0)
  ->AllocDevPtr();
  b_cpu_rset_cache
  .SetSize(cacheSize)
  ->SetOptions(0)
  ->AllocDevPtr();

  if (0 == devId)
  {
    b_cpu_rset.AllocHostPtr();
    b_cpu_rset_cache.AllocHostPtr();
  }
  else
  {
    b_cpu_rset.SetHostPtr(HeTM_cpu_rset.GetMemObj()->host);
    b_cpu_rset_cache.SetHostPtr(HeTM_cpu_rset_cache.GetMemObj()->host);
  }

  m_cpu_rset = new MemObj(&b_cpu_rset, devId);
  m_cpu_rset_cache = new MemObj(&b_cpu_rset_cache, devId);

  HeTM_gshared_data.bmap_rset_CPU_hostptr = m_cpu_rset->host;
  stm_rsetCPU = HeTM_gshared_data.bmap_rset_CPU_hostptr;
  HeTM_gshared_data.bmap_cache_rset_CPU_hostptr = m_cpu_rset_cache->host;
  stm_rsetCPUCache = HeTM_gshared_data.bmap_cache_rset_CPU_hostptr;
  stm_rsetCPUCacheBits = CACHE_GRANULE_BITS;

  HeTM_cpu_rset.AddMemObj(m_cpu_rset);
  HeTM_cpu_rset_cache.AddMemObj(m_cpu_rset_cache);

  m_gpu_rset = new MemObj(b_gpu_rset
    .SetSize(sizeRSetLog)
    ->SetOptions(0)
    ->AllocDevPtr(),
    devId);
  m_gpu_rset->ZeroDev();
  HeTM_gpu_rset.AddMemObj(m_gpu_rset);

  HeTM_shared_data[devId].bmap_rset_CPU_hostptr = HeTM_cpu_rset.GetMemObj(devId)->host;
  HeTM_shared_data[devId].bmap_rset_CPU_devptr = HeTM_cpu_rset.GetMemObj(devId)->dev;

  HeTM_shared_data[devId].bmap_cache_rset_CPU_hostptr = HeTM_cpu_rset_cache.GetMemObj(devId)->host;
  HeTM_shared_data[devId].bmap_cache_rset_CPU_devptr = HeTM_cpu_rset_cache.GetMemObj(devId)->dev;

  HeTM_shared_data[devId].bmap_rset_GPU_devptr = HeTM_gpu_rset.GetMemObj(devId)->dev;
  HeTM_gshared_data.rsetLogSize = sizeRSetLog;
}

// TODO: As a first step put each GPU WS as unified memory
static void init_multi_GPU(int devId, size_t pool_size)
{
  // TODO: there is some bug with the memory allocation
  size_t nbGranules = pool_size / PR_LOCK_GRANULARITY;
  size_t cacheSize = (nbGranules + (CACHE_GRANULE_SIZE-1)) / CACHE_GRANULE_SIZE;
  int nbOfGPUs = 0;

  Config::GetInstance()->SelDev(devId);
  nbOfGPUs = HeTM_gshared_data.nbOfGPUs = Config::GetInstance()->NbGPUs();

  MemObjBuilder b_gpu_rset_cache;
  MemObjBuilder b_gpu_wset_cache;
  MemObjBuilder b_gpu_wset_DIAG;

  // TODO: move the caches out of unified memory eventually
  b_gpu_rset_cache
    .SetSize(cacheSize)
    ->SetOptions(0)
    ->AllocUnifMem();
  // b_gpu_rset_cache
  //   .SetSize(cacheSize)
  //   ->SetOptions(0)
  //   ->AllocHostPtr()
  //   ->AllocDevPtr();
  b_gpu_wset_cache
    .SetSize(cacheSize)
    ->SetOptions(0)
    ->AllocUnifMem();
  // b_gpu_wset_cache
  //   .SetSize(cacheSize)
  //   ->SetOptions(0)
  //   ->AllocHostPtr()
  //   ->AllocDevPtr();

  MemObj *m_gpu_rset_cache = new MemObj(&b_gpu_rset_cache, devId);
  MemObj *m_gpu_wset_cache = new MemObj(&b_gpu_wset_cache, devId);
  HeTM_gpu_rset_cache.AddMemObj(m_gpu_rset_cache);
  HeTM_gpu_wset_cache.AddMemObj(m_gpu_wset_cache);

  HeTM_shared_data[devId].rsetGPUcache = m_gpu_rset_cache->host;
  HeTM_shared_data[devId].rsetGPUcache_hostptr = m_gpu_rset_cache->host;

  if (nullptr == HeTM_gpu_wset_ext)
  {
    HeTM_gpu_wset_ext = new MemObjOnDev[nbOfGPUs*nbOfGPUs];
  }

  b_gpu_wset_DIAG
    .SetSize(nbGranules)
    ->SetOptions(0)
    ->AllocDevPtr()
    ->AllocHostPtr();
  MemObj *m_gpu_wset_DIAG = new MemObj(&b_gpu_wset_DIAG, devId);
  HeTM_gpu_wset_ext[devId*nbOfGPUs+devId].AddMemObj(m_gpu_wset_DIAG);
  HeTM_gpu_wset.AddMemObj(m_gpu_wset_DIAG);

  for (int j = 0; j < nbOfGPUs; ++j)
  {
    if (devId == j)
      continue;

    MemObjBuilder b_gpu_wset;
    MemObj *m_gpu_wset;

    m_gpu_wset = new MemObj(b_gpu_wset
      .SetSize(nbGranules)
      ->SetOptions(0)
      ->AllocDevPtr()
      ->SetHostPtr(m_gpu_wset_DIAG->host),
      devId
    );

    HeTM_shared_data[devId].bmap_wset_GPU_devptr[j] = m_gpu_wset->dev;
    HeTM_gpu_wset_ext[devId*nbOfGPUs+j].AddMemObj(m_gpu_wset);
  }

  MemObjBuilder b_gpu_confl_mat;
  MemObj *m_gpu_confl_mat = new MemObj(b_gpu_confl_mat
    .SetSize(((nbOfGPUs+1) * (nbOfGPUs+1)) * sizeof(char))
    ->SetOptions(0)
    ->AllocUnifMem(),
    devId);
  HeTM_gpu_confl_mat.AddMemObj(m_gpu_confl_mat);
  m_gpu_confl_mat->ZeroDev();
  HeTM_shared_data[devId].mat_confl_GPU_unif = (char*)m_gpu_confl_mat->host;

  MemObjBuilder b_gpu_confl_mat_merge;
  MemObj *m_gpu_confl_mat_merge = new MemObj(b_gpu_confl_mat
    .SetSize(((nbOfGPUs+1) * (nbOfGPUs+1)) * sizeof(char))
    ->SetOptions(0)
    ->AllocUnifMem(),
    devId);
  HeTM_gpu_confl_mat_merge.AddMemObj(m_gpu_confl_mat_merge);
  m_gpu_confl_mat_merge->ZeroDev();
  HeTM_gshared_data.mat_confl_CPU_unif = (char*)m_gpu_confl_mat_merge->host;
}

static void init_bmap(int devId, size_t pool_size)
{
  size_t nbGranules = pool_size / PR_LOCK_GRANULARITY;
  // size_t granBmap;
  int nbOfGPUs = Config::GetInstance()->NbGPUs();
  size_t cacheSize = (nbGranules + (CACHE_GRANULE_SIZE-1)) / CACHE_GRANULE_SIZE;
  
  Config::GetInstance()->SelDev(devId);

  // TODO: this is repeated for each GPU
  HeTM_gshared_data.wsetLogSize   = nbGranules;
  HeTM_gshared_data.bmap_cache_wset_CPUSize = CACHE_GRANULE_SIZE;
  HeTM_gshared_data.bmap_cache_wset_CPUBits = CACHE_GRANULE_BITS;
  HeTM_gshared_data.sizeMemPool   = pool_size;

  MemObjBuilder b_cpu_wset;
  MemObj *m_cpu_wset;
  MemObjBuilder b_cpu_wset_cache;
  MemObj *m_cpu_wset_cache;

  if (devId == 0)
  { // TODO: assumes devId==0 is called first
    b_cpu_wset_cache
      .SetSize(cacheSize)
      ->SetOptions(0)
      ->AllocHostPtr()
      ->AllocDevPtr();
    m_cpu_wset_cache = new MemObj(&b_cpu_wset_cache, 0);
    HeTM_cpu_wset_cache.AddMemObj(m_cpu_wset_cache);
    m_cpu_wset_cache->ZeroDev();
    m_cpu_wset_cache->ZeroHost();
    HeTM_shared_data[devId].bmap_cache_wset_CPU_devptr = m_cpu_wset_cache->dev;
    stm_wsetCPUCache = m_cpu_wset_cache->host;
    HeTM_shared_data[devId].bmap_cache_wset_CPU_hostptr = m_cpu_wset_cache->host;

    b_cpu_wset
      .SetSize(nbGranules)
      ->SetOptions(0)
      ->AllocHostPtr()
      ->AllocDevPtr();
    m_cpu_wset = new MemObj(&b_cpu_wset, 0);
    HeTM_cpu_wset.AddMemObj(m_cpu_wset);
    m_cpu_wset->ZeroDev();
    m_cpu_wset->ZeroHost();

    HeTM_shared_data[devId].bmap_wset_CPU_devptr = m_cpu_wset->dev;
    HeTM_shared_data[devId].bmap_wset_CPU_hostptr = m_cpu_wset->host;
    stm_wsetCPU = m_cpu_wset->host;
  }
  else
  {
    b_cpu_wset_cache
      .SetSize(cacheSize)
      ->SetOptions(0)
      ->SetHostPtr(HeTM_cpu_wset_cache.GetMemObj(0)->host)
      ->AllocDevPtr();
    m_cpu_wset_cache = new MemObj(&b_cpu_wset_cache, devId);
    HeTM_cpu_wset_cache.AddMemObj(m_cpu_wset_cache);
    m_cpu_wset_cache->ZeroDev();
    HeTM_shared_data[devId].bmap_cache_wset_CPU_devptr = m_cpu_wset_cache->dev;
    HeTM_shared_data[devId].bmap_cache_wset_CPU_hostptr = stm_wsetCPUCache;

    b_cpu_wset
      .SetSize(nbGranules)
      ->SetOptions(0)
      ->SetHostPtr(HeTM_cpu_wset.GetMemObj(0)->host)
      ->AllocDevPtr();
    m_cpu_wset = new MemObj(&b_cpu_wset, devId);
    HeTM_cpu_wset.AddMemObj(m_cpu_wset);
    m_cpu_wset->ZeroDev();
    HeTM_shared_data[devId].bmap_wset_CPU_devptr = m_cpu_wset->dev;
    HeTM_shared_data[devId].bmap_wset_CPU_hostptr = m_cpu_wset->host;
  }

  stm_wsetCPUCacheBits = CACHE_GRANULE_BITS;

  stm_baseMemPool[devId] = HeTM_mempool.GetMemObj(devId)->host;
  HeTM_shared_data[devId].mempool_hostptr = stm_baseMemPool[devId];
}
