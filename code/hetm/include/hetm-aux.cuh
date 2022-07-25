#ifndef HETM_AUX_H_GUARD_
#define HETM_AUX_H_GUARD_

#include "pr-stm.cuh"
#include "memman.hpp"
#include "knlman.hpp"

#define NVTX_PROF_BLOCK           0
#define NVTX_PROF_BACKOFF         1
#define NVTX_PROF_SYNC_PHASE      2
#define NVTX_PROF_CPY_CPU_TO_GPUS 3
#define NVTX_PROF_CPY_CPU_TO_1GPU 4
#define NVTX_PROF_CPY_GPUS_TO_CPU 5
#define NVTX_PROF_VALIDATE_GPU    6
#define NVTX_PROF_VALIDATE_CPU    7
#define NVTX_PROF_CPU_WAITS_NEXT_BATCH 8
#define NVTX_PROF_GPU_WAITS_NEXT_BATCH 9
#define NVTX_PROF_ROLLBACK_CPU    10
#define NVTX_RUN_TX_BATCH         11



/* MEMORY REGIONS DOCUMENTATION
 *  - "HeTM_mempool": main memory pool accessed by both CPUs and GPUs
 *  - "HeTM_mempool_bmap": 1B per 4B, tracks if it was written written 
 *  - "HeTM_mempool_backup": [DEPRECATED] used to rollback the GPU
 *  - "HeTM_mempool_backup_bmap": [DEPRECATED] same as HeTM_mempool_bmap for the backup
 *  - "HeTM_versions": per granule CPU version, (CPU has logs with different granule version)
 *  - "HeTM_gpuLog": aggregates metadata sent in the GPU (e.g., read and write-sets)
 *  - "HeTM_gpu_rset": bitmap containing the GPU readset (blind writes avoided by placing writes also here)
 *  - "HeTM_gpu_wset": bitmap containing the GPU writeset
 *  - "HeTM_cpu_wset": buffer for the CPU logs
 *  - "HeTM_interConflFlag": [DEPRECATED] flag set to one if CPU_WS -> GPU_RS
 *  - "HeTM_gpu_confl_mat": conflict matrix
 *  - "HeTM_gpu_confl_mat_cpu": completed conflict matrix, first updated with the CPU conflicts
 * 
 * 
 */

extern memman::MemObjOnDev HeTM_gpuLog;
extern memman::MemObjOnDev HeTM_mempool;
extern memman::MemObjOnDev HeTM_mempool_bmap;

extern memman::MemObjOnDev HeTM_gpu_rset;
extern memman::MemObjOnDev HeTM_gpu_wset;

extern memman::MemObjOnDev *HeTM_gpu_wset_ext;
extern memman::MemObjOnDev HeTM_gpu_rset_cache;
extern memman::MemObjOnDev HeTM_gpu_wset_cache;

extern memman::MemObjOnDev HeTM_cpu_rset;
extern memman::MemObjOnDev HeTM_cpu_wset;
extern memman::MemObjOnDev HeTM_cpu_rset_cache;
extern memman::MemObjOnDev HeTM_cpu_wset_cache;

extern memman::MemObjOnDev HeTM_gpu_confl_mat;
extern memman::MemObjOnDev HeTM_gpu_confl_mat_merge;

extern memman::MemObjOnDev HeTM_curand_state;

// ----------------------- Kernels used
extern knlman::KnlObj *HeTM_checkTxCompressed;
extern knlman::KnlObj *HeTM_earlyCheckTxCompressed;
extern knlman::KnlObj *HeTM_applyTxCompressed;
extern knlman::KnlObj *HeTM_checkTxExplicit;
extern knlman::KnlObj *HeTM_interGPUConflDetect;
extern knlman::KnlObj *HeTM_CPUGPUConflDetect;
extern knlman::KnlObj *HeTM_CPUrsGPUwsConflDetect;

extern memman::MemObjOnDev HeTM_interGPUConflDetect_entryObj;
extern memman::MemObjOnDev HeTM_CPUGPUConflDetect_entryObj;
extern memman::MemObjOnDev HeTM_CPUrsGPUwsConflDetect_entryObj;
// ------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

// TODO: better naming do not call it outside hetm code
#define WAIT_ON_FLAG(flag) while(!(__atomic_load_n(&flag, __ATOMIC_ACQUIRE)) && !HeTM_is_stop()) PAUSE(); __atomic_store_n(&flag, 0, __ATOMIC_RELAXED);

typedef void*(*HeTM_gpu_work_fn)(void*);
typedef struct offload_gpu_
{
  int GPUid;
  volatile int *hasWork;
  HeTM_gpu_work_fn *fn;
  void **args;
}
offload_gpu_s;

// ----------------------- HeTM CPU
void cpyCPUwrtsetToGPU(int nonBlock);
void waitNextBatch(int nonBlock);
void pollIsRoundComplete(int nonBlock);
void resetInterGPUConflFlag();
void runBeforeCPU(int id, void *data);
void runAfterCPU(int id, void *data);
// ------------ aux functions
void enterCPU();
void exitCPU();
// --------------------------
// --------------------------------

// ----------------------- HeTM GPU
extern int isAfterCmpDone;
extern int isGetStatsDone;
extern int isGetPRStatsDone;
extern int isDatasetSyncDone;
extern int isGPUResetDone;
extern long roundCountAfterBatch;
void runAfterGPU(int id, void *data);
void runBeforeGPU(int id, void *data);
void initGPUPeerCpy();
void destroyGPUPeerCpy();
void doGPUStateReset();
void runGPUBeforeKernel(int id, void *data);
void runGPUBatch();
void runGPUAfterBatch(int id, void *data);
void waitGPUBatchEnd();
void runGPUAfterKernel(int id, void *data);
void notifyBatchIsDone();

void waitGPUCMPEnd(int nonBlock);
// waitCPUlogValidation and syncGPUtoCPUbarrier are called in waitGPUCMPEnd
void waitCPUlogValidation(int nonBlock);
void syncGPUtoCPUbarrier(int nonBlock);

void mergeGPUDataset();
void runAfterBatch(int id, void *data);
int isGPUBatchAbort();
void runGPUBeforeBatch(int id, void *data);
void getGPUStatistics(void*);
void accumulateStatistics();
void getGPUPRSTMStats(void *argPtr);
void syncGPUdataset(void *args);
void waitGPUdataset(void *args);
// ------------ aux functions
void enterGPU();
void exitGPU();
int timeIsOver();
void checkIsExit();
void notifyCPUNextBatch();
void offloadResetGPUState(void*);
void setCountAfterBatch();

// TODO: this is not defined here
// #ifndef PR_globalKernelArgs
// #define PR_globalKernelArgs pr_tx_args_dev_host_s args
// typedef struct pr_tx_args_ pr_tx_args_s;
// #endif 

void runPrSTMCallback(int nbBlcks, int nbThrsPerBlck, void(*callback)(PR_globalKernelArgs), void* inPtr, int sizeIn, void* outPtr, int sizeOut);
pr_tx_args_s *getPrSTMmetaData(int devId);
void mergeMatricesAndRunFVS(int nonBlock);
// --------------------------
// --------------------------------

// ----------------------- TODO: test
void hetm_memcpyDeviceToCPU(int devId, HeTM_thread_s *threadData);
// --------------------------------

#ifdef __cplusplus
}
#endif

#endif /* HETM_AUX_H_GUARD_ */
