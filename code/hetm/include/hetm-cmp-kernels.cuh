#ifndef HETM_CMP_KERNELS_H_GUARD_
#define HETM_CMP_KERNELS_H_GUARD_

#include <stdio.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// #include "cmp_kernels.cuh"
#include "hetm-log.h"
#include "pr-stm-wrapper.cuh"
#include "bitmap.hpp"

#define CMP_EXPLICIT_THRS_PER_RSET 1024 /* divide per CMP_EXPLICIT_THRS_PER_WSET */
#define CMP_EXPLICIT_THRS_PER_WSET 32 /* nb. of entries to compare */

/****************************************
 *	HeTM_knl_checkTxCompressed()
 *
 *	Description: Compare device write-log with host log, when using compressed log
 *
 ****************************************/

typedef struct HeTM_knl_global_ {
  void *devMemPoolBasePtr;
  void *hostMemPoolBasePtr;
	size_t nbGranules;
  void *devRSet;
  void *devRSetCache_hostptr[HETM_NB_DEVICES];
  void *devWSet[HETM_NB_DEVICES];
  void *devWSetCache[HETM_NB_DEVICES];
  void *devWSetCache_hostptr[HETM_NB_DEVICES];
  char *localConflMatrix[HETM_NB_DEVICES];
  void *hostRSet;
  void *hostRSetCache;
  void *hostRSetCache_hostptr;
  void *hostWSet;
  void *hostWSetCache_hostptr;
  void *hostWSetCache;
  size_t hostWSetCacheSize;
  size_t hostWSetCacheBits;
	size_t hostWSetChunks;
	void *randState;
	int isGPUOnly;
	void *GPUwsBmap;
} HeTM_knl_global_s;

typedef struct HeTM_knl_cmp_args_ {
  int devId; 
  int otherDevId; // for inter-GPU
  int nbOfGPUs;
	int sizeWSet; /* Size of the host log */
	int sizeRSet; /* Size of the device log */
	int idCPUThr;
	unsigned char batchCount;
  HeTM_knl_global_s knlGlobal; // before it was a __constant__ HeTM_knl_global
} HeTM_knl_cmp_args_s;

typedef struct HeTM_cmp_ {
	HeTM_knl_cmp_args_s knlArgs;
	HeTM_thread_s *clbkArgs;
} HeTM_cmp_s;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void HeTM_set_global_arg(int devId, HeTM_knl_global_s);
HeTM_knl_global_s *HeTM_get_global_arg(int devId);

// set the size of the explicit log (benchmark dependent), it is equal to
// the number of writes of a thread (all transactions) in a GPU kernel

#ifdef __cplusplus
}
#endif /* __cplusplus */

__global__ void interGPUConflDetect(HeTM_knl_cmp_args_s args, char *remote_wset, char *my_rset, size_t offset);
__global__ void CPUGPUConflDetect(HeTM_knl_cmp_args_s args, size_t offset);
__global__ void CPUrsGPUwsConflDetect(HeTM_knl_cmp_args_s args, size_t offset);

// HETM_BMAP_LOG requires a specific kernel
__global__ void HeTM_knl_checkTxBitmap(HeTM_knl_cmp_args_s args, size_t offset);
__global__ void HeTM_knl_checkTxBitmapCache(HeTM_knl_cmp_args_s args);
__global__ void HeTM_knl_checkTxBitmap_Explicit(HeTM_knl_cmp_args_s args);
__global__ void HeTM_knl_writeTxBitmap(HeTM_knl_cmp_args_s args, size_t offset);

#endif /* HETM_CMP_KERNELS_H_GUARD_ */
