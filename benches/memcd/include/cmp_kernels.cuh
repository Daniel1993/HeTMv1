#ifndef _CMP_KERNEL
#define _CMP_KERNEL

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "pr-stm-wrapper.cuh"
#include "hetm-log.h"
#include "cuda_defines.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <curand_kernel.h>
#include "helper_cuda.h"
#include "cuda_wrapper.h"
#include "helper_timer.h"
#include <time.h>
#include "bitmap.h"

#include "hetm-cmp-kernels.cuh"

//Support for the host log comparison
#define checkThreadMult   2   // Number of warps in each comparison kernel block
#define checkLocalSize    8   // Size of the local memory in the comparison kernel
#define checkSharedSize   (checkThreadMult*32)
#define blockCheckSize    32*checkThreadMult*checkLocalSize
#define blockCheckOffset  (LOG_SIZE+1)/(blockCheckSize)
#define compressHash      10   //How many accounts map to one position of the compressed log

//Support for the lazylog implementation
#define EXPLICIT_LOG_BLOCK     (parsedData.trans * BANK_NB_TRANSFERS)
// b==numBlocks t==numThreads
#define EXPLICIT_LOG_SIZE(b,t) (b * t * EXPLICIT_LOG_BLOCK)	//size of the lazy lock

/****************************************************************************
 *	KERNELS
 ****************************************************************************/

// The namespace for these kernels is HeTM_knl_*

/****************************************
 *	HeTM_knl_checkTx()
 *
 *	Description: Compare device write-log with host log.
 *
 ****************************************/
typedef struct HeTM_knl_checkTx_ {
	int *dev_flag;
	int *stm_log;  /*Host log*/
	int size_stm;  /*Size of the host log*/
	int *devLogR;  /*GPU read log */
} HeTM_knl_checkTx_s;

__global__ void HeTM_knl_checkTx(HeTM_knl_checkTx_s);

// called after stream
void CUDART_CB checkCallback(cudaStream_t event, cudaError_t status, void *data);

/****************************************
 *	HeTM_knl_finalTxLog()
 *
 *	Description: Generate a log of accessed memory positions, when using compressed log
 *
 ****************************************/
typedef struct HeTM_knl_finalTxLog_ {
	int * global_bitmap; /*values in global memory*/
	int * devLogR;       /*GPU read log */
} HeTM_knl_finalTxLog_s;

// finalizeTransactionLog
__global__ void HeTM_knl_finalTxLog(HeTM_knl_finalTxLog_s);

/****************************************
 *	HeTM_knl_finalTxs()
 *
 *	Description: Update mapped memory with the GPU's transactions
 *
 ****************************************/
typedef struct HeTM_knl_finalTxs_ {
	int * devLogR; /*GPU read/write log */
	long * a;      /*Values array*/
	long * b;      /*Bitmap*/
	int size;
} HeTM_knl_finalTxs_s;

__global__ void HeTM_knl_finalTxs(HeTM_knl_finalTxs_s);

/****************************************
 *	HeTM_knl_finalTxLog2()
 *
 *	Description: Generate a log of accessed memory positions, when using compressed log
 *
 ****************************************/
typedef struct HeTM_knl_finalTxLog2_ {
 	int *global_bitmap; /* values in global memory */
	int size;
	int *devLogR;       /* GPU read log */
} HeTM_knl_finalTxLog2_s;

// typedef struct HeTM_finalTxLog2_ {
//  	HeTM_knl_finalTxLog2_s knlArgs;
//  	stream_t *clbkArgs;
// } HeTM_finalTxLog2_s;

__global__ void HeTM_knl_finalTxLog2(HeTM_knl_finalTxLog2_s);

/****************************************************************************
 *	FUNCTIONS
 ****************************************************************************/

#endif
