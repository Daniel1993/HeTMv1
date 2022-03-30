#ifndef HETM_H_GUARD_
#define HETM_H_GUARD_

#include <cuda.h>
#include <cuda_profiler_api.h>

#include "hetm-log.h"
#include "hetm-utils.h"
#include "hetm-producer-consumer.h"
#include "hetm-timer.h"
#include "memman.hpp"

#define HETM_MAX_CMP_KERNEL_LAUNCHES 4
#define HETM_MAX_CPU_THREADS         64
#define HETM_PC_BUFFER_SIZE          0x100
#define CONTINUE_COND (!HeTM_is_stop(0) || HeTM_get_GPU_status(0) != HETM_IS_EXIT)

#define HETM_CMP_DISABLED   0
#define HETM_CMP_COMPRESSED 1
#define HETM_CMP_EXPLICIT   2

#ifndef HETM_CMP_TYPE
#define HETM_CMP_TYPE HETM_CMP_DISABLED
#endif

#ifndef HETM_NB_DEVICES
#define HETM_NB_DEVICES 2
#endif

static const size_t MAXSIZE_BEFORE_SMART_KERNEL = 1000000;

typedef enum {
  HETM_BATCH_RUN  = 0,
  HETM_BATCH_DONE = 1,
  HETM_IS_EXIT    = 2,
  HETM_GPU_IDLE   = 3
} HETM_GPU_STATE;

typedef enum {
  HETM_CMP_OFF     = 0, // during normal execution
  HETM_CMP_STARTED = 1, // execution ended --> start comparing
  HETM_CPY_ASYNC   = 2, // copy while running TXs
  HETM_CMP_ASYNC   = 3, // CMP while running TXs
  HETM_DONE_ASYNC  = 4, // after cpy and cmp
  HETM_CMP_BLOCK   = 5  // block CPU while the compare completes
} HETM_CMP_STATE;

typedef enum {
  HETM_CPU_INV,
  HETM_GPU_INV
} HETM_INVALIDATE_POLICY;

// ------------- callbacks
// the GPU callback launches the kernel inside of it
typedef void(*HeTM_callback)(int, void*);
typedef void(*HeTM_request)(void*);
typedef int(*HeTM_map_tid_to_core)(int);
// -------------

// use this to run some function in a side thread
#define RUN_ASYNC(_func, _args) \
  if (!HeTM_run_sync) { \
    HeTM_async_request((HeTM_async_req_s){ \
      .args = (void*)_args, \
      .fn = _func, \
    }); \
  } else { \
    _func((void*)_args); \
  } \
//

typedef struct HeTM_thread_
{
  // configuration
  int id;
  int devId;
  HeTM_callback callback;
  void *args;
  pthread_t thread;

  // algorithm specific
  HETM_LOG_T *wSetLog;
  volatile int isCmpDone;
  volatile int isCmpVoid;
  int countCpy;
  int nbCpyDone;
  volatile int isCpyDone;
  volatile int isApplyDone;

  cudaEvent_t cmpStartEvent, cmpStopEvent;
  float timeCmp;
  double timeCmpSum;

  cudaEvent_t cpyWSetStartEvent, cpyWSetStopEvent;
  cudaEvent_t cpyLogChunkStartEvent[STM_LOG_BUFFER_SIZE];
  cudaEvent_t cpyLogChunkStopEvent[STM_LOG_BUFFER_SIZE];
  unsigned long logChunkEventCounter, logChunkEventStore;
  float timeCpy;
  double timeCpySum;
  double timeMemCpySum;

  cudaEvent_t cpyDatasetStartEvent, cpyDatasetStopEvent;
  float timeCpyDataset;
  double timeCpyDatasetSum;

  size_t curWSetSize;
  HETM_CMP_STATE statusCMP;
  int nbCmpLaunches;
  long curNbTxs /* count all */, curNbTxsNonBlocking /* ADDR|VERS */;
  void *stream; // cudaStream_t

  size_t emptySpaceFirstChunk;

  TIMER_T backoffBegTimer;
  TIMER_T backoffEndTimer;
  TIMER_T blockingEndTimer;
  TIMER_T beforeCpyLogs;
  TIMER_T beforeCmpKernel;
  TIMER_T beforeApplyKernel;

  double timeLogs;
  double timeCmpKernels;
  double timeApplyKernels;
  double timeBackoff;
  double timeBlocked;

  int targetCopyNb;
  int curCopyNb;
  int doHardLogCpy;
  int isCopying;
} HeTM_thread_s;

// sigleton
typedef struct HeTM_shared_
{
  // algorithm specific
  HETM_GPU_STATE statusGPU;
  volatile int stopFlag;
  volatile int stopAsyncFlag;
  barrier_t GPUBarrier;
  volatile int isInterconflict; // set to 1 when a CPU-GPU conflict is found
  int *hostInterConflFlag;
  int *devInterConflFlag;

  // memory pool
  void *mempool_hostptr;
  void *mempool_devptr;
  void *bmap_mempool_devptr;

  void *mempool_backup_hostptr;
  void *mempool_backup_devptr;
  void *bmap_mempool_backup_devptr;
  void *bmap_mempool_backup; // memman_bmap_s* in the host

  void *bmap_rset_CPU_devptr;
  void *bmap_cache_rset_CPU_devptr;

  // wset in the CPU, rset in the GPU (bitmap or explict)
  void *bmap_wset_CPU_devptr;
  void *bmap_wset_CPU_hostptr;
  void *bmap_rset_GPU_devptr; // size in gshared

  void *bmap_rset_CPU_hostptr; // TODO: currently using unified memory double check if need to copy to GPU
  void *bmap_cache_rset_CPU_hostptr;

  // TODO: is this in use? come up with better naming
  void *bmap_cache_wset_CPU_devptr;
  void *bmap_cache_wset_CPU_hostptr;
  void *bmap_cache_wset_CPUConfl; // CPU WS | GPU WS
  void *bmap_cache_merged_wset_devptr;
  void *bmap_cache_merged_wset_hostptr;

  // TODO: add the merged contents in a BMAP
  void *bmap_merged_wset_devptr;
  void *bmap_merged_wset_hostptr;
  
  void *wsetGPUcache;
  void *wsetGPUcache_hostptr;
  void *rsetGPUcache;
  void *rsetGPUcache_hostptr;
  void *bmap_cache_wset_CPUConfl3;
  size_t bmap_cache_wset_CPUSize, bmap_cache_wset_CPUBits;

  void *bmap_wset_GPU_devptr[HETM_NB_DEVICES];
  char *mat_confl_GPU_unif;

  // TODO: remove this is benchmark specific
  void *devCurandState; // type is curandState*
  size_t devCurandStateSize;

  HeTM_thread_s *threadsInfo;
  pthread_t asyncThread;

  cudaEvent_t batchStartEvent, batchStopEvent;

  volatile int isInterGPUConflDone;
  volatile long threadsWaitingSync; /* waits target GPU to validate CPU log */
  float timeBudget;

} HeTM_shared_s;

typedef struct HeTM_gshared_
{
  // configuration
  volatile HETM_INVALIDATE_POLICY policy;
  int nbCPUThreads, nbGPUBlocks, nbGPUThreads;
  int isCPUEnabled, isGPUEnabled, nbThreads;

  // algorithm specific
  HETM_GPU_STATE statusGPU;
  int stopFlag, stopAsyncFlag;
  barrier_t CPUBarrier;
  barrier_t nextBatchBarrier;
  barrier_t BBsyncBarrier;

  // memory pool
  void *mempool_hostptr;
  void *mempool_devptr;
  void *mempool_backup_hostptr;
  void *mempool_backup_devptr;
  void *bmap_mempool_backup_devptr;
  void *devMemPoolBackupBmap;
  size_t sizeMemPool;

  // wset in the CPU, rset in the GPU (bitmap or explict)
  size_t wsetLogSize, rsetLogSize; // pointers in shared

  void *bmap_rset_CPU_hostptr; // TODO: currently using unified memory double check if need to copy to GPU
  void *bmap_cache_rset_CPU_hostptr;

  // TODO: is this in use? come up with better naming
  size_t bmap_cache_wset_CPUSize, bmap_cache_wset_CPUBits, nbChunks;

  int nbOfGPUs;
  void *bmap_wset_GPU_hostptr[HETM_NB_DEVICES];
  char *mat_confl_CPU_unif; // final merged matrix, CPU pre-fill this with its own conflicts

  // TODO: remove this is benchmark specific
  void *devCurandState; // type is curandState*
  size_t devCurandStateSize;

  HeTM_thread_s *threadsInfo;
  pthread_t asyncThread;

  cudaEvent_t batchStartEvent, batchStopEvent;

  int isInterGPUConflDone;

  long batchCount;
  float timeBudget;

} HeTM_gshared_s;

typedef struct HeTM_statistics_ {
  long nbBatches, nbBatchesSuccess, nbBatchesFail;
  long nbTxsGPU, nbCommittedTxsGPU, nbDroppedTxsGPU, nbAbortsGPU;
  long nbTxsCPU, nbCommittedTxsCPU, nbDroppedTxsCPU, nbAbortsCPU;
  long nbTxsPerGPU[HETM_NB_DEVICES], nbCommittedTxsPerGPU[HETM_NB_DEVICES], nbDroppedTxsPerGPU[HETM_NB_DEVICES], nbAbortsPerGPU[HETM_NB_DEVICES];
  long nbEarlyValAborts;
  size_t sizeCpyDataset, sizeCpyWSet /* ADDR|BMAP */, sizeCpyLogs /* ADDR|VERS */;
  size_t sizeCpyWSetHtD, sizeCpyWSetDtH;
  size_t sizeCpyWSetCPUData;
  long nbBitmapConflicts;
  long nbCPUdataConflicts;
  long txsNonBlocking;
  double timeNonBlocking, timeBlocking;
  double timeCMP, timeAfterCMP;
  unsigned long timeCPUcmp;
  double timeGPU, timePRSTM;
  double timeAbortedBatches;
  double timeDtD;
  double timeCPU;
  double timeMemCpySum;
  double totalTimeCpyWSet, totalTimeCmp, totalTimeCpyDataset;
} HeTM_statistics_s;

typedef struct HeTM_init_ {
  HETM_INVALIDATE_POLICY policy;
  int nbCPUThreads, nbGPUBlocks, nbGPUThreads;
  float timeBudget;
  int isCPUEnabled, isGPUEnabled;
  long mempool_size;
  int mempool_opts;
} HeTM_init_s;

typedef struct HeTM_async_req_ {
  void *args;
  HeTM_request fn;
} HeTM_async_req_s;

extern HeTM_shared_s HeTM_shared_data[HETM_NB_DEVICES];
extern HeTM_gshared_s HeTM_gshared_data;
extern HeTM_statistics_s HeTM_stats_data;
extern hetm_pc_s *HeTM_offload_pc;

extern HeTM_thread_s HeTM_thread_data_all[HETM_NB_DEVICES][HETM_MAX_CPU_THREADS];
extern __thread HeTM_thread_s *HeTM_thread_data[HETM_NB_DEVICES];
extern void *HeTM_memStream[HETM_NB_DEVICES];
extern void *HeTM_memStream2[HETM_NB_DEVICES];

const static int CACHE_GRANULE_SIZE = BMAP_GRAN;// 16384;
const static int CACHE_GRANULE_BITS = BMAP_GRAN_BITS;// 14;

// here it is defined some methods that are restricted to the implementation
#include "hetm-aux.cuh"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


extern int HeTM_run_sync;

int HeTM_init(HeTM_init_s);
int HeTM_destroy();

//---------------------- Memory Pool
// Sets the memory pool size for HeTM, users can then
// allocate memory from this pool using HeTM_alloc(...)
int HeTM_mempool_init(size_t pool_size, int opts);
int HeTM_mempool_destroy(int devId);
int HeTM_mempool_cpy_to_cpu(int devId, size_t *copiedSize, long batchCount);
int HeTM_mempool_cpy_to_gpu(int devId, size_t *copiedSize, long batchCount);
int HeTM_alloc(int devId, void **cpu_ptr, void **gpu_ptr, size_t);
int HeTM_free(int devId, void **cpu_ptr);
void* HeTM_map_addr_to_gpu(int devId, void *origin);
void* HeTM_map_addr_to_cpu(int devId, void *origin);

void HeTM_initCurandState(int devId);
void HeTM_destroyCurandState(int devId);

// returns 1 if out of space in the buffer (and does not copy)
//----------------------

//---------------------- Threading
// Call this before HeTM_start for a custom core mapping function
int HeTM_set_thread_mapping_fn(HeTM_map_tid_to_core);

void HeTM_cpu_thread(); // Do not call this
void HeTM_gpu_thread(); // Do not call this

// Registers two callbacks, first for the CPU, second for the GPU,
// and the global arguments (GPU thread will have the last id),
//  --- IMPORTANT:
// GPU callback must guarantee the kernel is launched before returning
int HeTM_start(HeTM_callback, HeTM_callback, void *args);
int HeTM_before_cpu_start(HeTM_callback); // uses these to collect statistics
int HeTM_after_cpu_finish(HeTM_callback);
int HeTM_before_gpu_start(HeTM_callback);
int HeTM_after_gpu_finish(HeTM_callback);

// choose the policy for the next batch
int HeTM_choose_policy(HeTM_callback);

// neither CPU nor GPU are running (executes after HeTM_after_batch)
int HeTM_before_batch(HeTM_callback);

// neither CPU nor GPU are running (executes before HeTM_after_batch)
int HeTM_after_batch(HeTM_callback);

int HeTM_before_kernel(HeTM_callback); // use it, e.g., to send input
int HeTM_after_kernel(HeTM_callback); // TODO: batch is not complete

// Registers a callback to be later executed (serially in a side thread)
void HeTM_async_request(HeTM_async_req_s req);
void HeTM_free_async_request(HeTM_async_req_s *req); // Do not call this

int HeTM_sync_barrier(int devId); // Do not call this
int HeTM_sync_CPU_barrier();
int HeTM_sync_next_batch();
int HeTM_sync_BB();
int HeTM_flush_barrier(int devId); // Do not call this

// Waits the threads. Note: HeTM_set_is_stop(1) must be called
// before so the threads know it is time to stop
int HeTM_join_CPU_threads();
//----------------------

//---------------------- getters/setters
#define HeTM_set_is_stop(_devId, isStop)       (HeTM_shared_data[_devId].stopFlag = isStop)
#define HeTM_is_stop(_devId)                   (HeTM_shared_data[_devId].stopFlag)
#define HeTM_async_set_is_stop(_devId, isStop) (__atomic_store_n(&HeTM_shared_data[_devId].stopAsyncFlag, isStop, __ATOMIC_RELEASE))
#define HeTM_async_is_stop(_devId)             (__atomic_load_n(&HeTM_shared_data[_devId].stopAsyncFlag, __ATOMIC_ACQUIRE))
#define HeTM_set_is_interconflict(_devId, val) (HeTM_shared_data[_devId].isInterconflict = val)
#define HeTM_is_interconflict(_devId)          (HeTM_shared_data[_devId].isInterconflict)
#define HeTM_set_GPU_status(_devId, status)    (HeTM_shared_data[_devId].statusGPU = status)
#define HeTM_get_GPU_status(_devId)            (HeTM_shared_data[_devId].statusGPU)

// resets PR-STM lock table, flags, etc
int HeTM_reset_GPU_state(long batchCount);
//----------------------

#ifdef __cplusplus
}
#endif /* __cplusplus */

// ###############################
// ### Debuging ##################
// ###############################
#ifdef HETM_DEB
#define HETM_PRINT(...) printf("[%s]: ", __func__); printf(__VA_ARGS__); printf("\n");

// TODO: use flags to enable/disable each module prints
// #define HETM_DEB_THREADING(...) printf("[THR]"); HETM_PRINT(__VA_ARGS__)
#define HETM_DEB_THREADING(...) /* empty */
#define HETM_DEB_THRD_CPU(...)  {printf("\033[1;36m" "[CPU]" "\033[0m"); HETM_PRINT(__VA_ARGS__);}
#define HETM_DEB_THRD_GPU(...)  {printf("\033[1;33m" "[GPU]" "\033[0m"); HETM_PRINT(__VA_ARGS__);}

#else /* !HETM_DEB */
#define HETM_PRINT(...) /* empty */

// TODO: use flags to enable/disable each module prints
#define HETM_DEB_THREADING(...) /* empty */
#define HETM_DEB_THRD_CPU(...)  /* empty */
#define HETM_DEB_THRD_GPU(...)  /* empty */
#endif
// ###############################

#endif /* HETM_H_GUARD_ */
