#include <stdio.h>

#include "hetm-log.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// TODO: code included
#include "kernels/BlockingCmp.cuh"
#include "hetm-cmp-kernels.cuh"

#include "memman.hpp"

/****************************************************************************
 *	GLOBALS
 ****************************************************************************/

/****************************************
 *	HeTM_knl_checkTx()
 *
 *	Description: Compare device write-log with host log.
 *
 ****************************************/

__global__ void HeTM_knl_checkTx(HeTM_knl_checkTx_s args)
{
	int *dev_flag = args.dev_flag; /*Flag in global memory*/
	int *stm_log  = args.stm_log;  /*Host log*/
	int size_stm  = args.size_stm; /*Size of the host log*/
	int *devLogR  = args.devLogR;  /*GPU read log */
	int i;
	int	pos;
	int id = threadIdx.y*32 + threadIdx.x;
	unsigned int logOffset = 0;
	int local_stm[checkLocalSize];

	__shared__ int shared_gpu[checkSharedSize];


	//Save GPU log to shared memory
	shared_gpu[id] = devLogR[blockIdx.x*checkSharedSize+id];

	__syncthreads();						//Is this necessary???

	//Save stm log to local memory
	logOffset = blockIdx.y*blockCheckSize + threadIdx.y*32*checkLocalSize + threadIdx.x;
	#pragma unroll
	for(i= 0; i< checkLocalSize; i++) {
		pos = logOffset + i*checkLocalSize;
		local_stm[i] = pos < size_stm ? stm_log[pos] : -1;			//Acess global memory only if the value is valid
	}

	//Comparison
#if ARCH == FERMI
	checkTransactionKernel_FERMI(dev_flag, shared_gpu, local_stm, id);
#else
	checkTransactionKernel_OTHER(dev_flag, shared_gpu, local_stm, id);
#endif
}

/****************************************
 *	HeTM_knl_finalTxLog()
 *
 *	Description: Generate a log of accessed memory positions, when using compressed log
 *
 ****************************************/
__global__ void HeTM_knl_finalTxLog(HeTM_knl_finalTxLog_s args)
{
	int *global_bitmap = args.global_bitmap; /*values in global memory*/
	int *devLogR = args.devLogR; /*GPU read log */
	int id = threadIdx.y*32 + threadIdx.x;
	//int val;
	int	save;

	//Save GPU log to shared memory
	save = devLogR[blockIdx.x*checkSharedSize+id];
	//Save to compressed log
	global_bitmap[save] = writeMask;
}

/****************************************
 *	HeTM_knl_finalTxs()
 *
 *	Description: Update mapped memory with the GPU's transactions
 *
 ****************************************/
__global__ void HeTM_knl_finalTxs(HeTM_knl_finalTxs_s args)
{
	int *devLogR = args.devLogR; /*GPU read/write log */
	long *a      = args.a;       /*Values array*/
	long *b      = args.b;       /*Bitmap*/
	int size     = args.size;
	int id = blockIdx.x*32 + threadIdx.x;
	int mask;

	if (id < size ) {
		//Get log value
		mask = devLogR[id];

		b[id] = mask != 0 ? b[id] : a[id];

		/*if ( mask & writeMask  != 0) {
			b[id] = a[id];
		}*/
	}
}

/****************************************
 *	HeTM_knl_finalTxLog2()
 *
 *	Description: Generate a log of accessed memory positions, when using compressed log
 *
 ****************************************/
__global__ void HeTM_knl_finalTxLog2(HeTM_knl_finalTxLog2_s args)
{
	int *global_bitmap = args.global_bitmap; /*values in global memory*/
	int size           = args.size;
	int *devLogR       = args.devLogR; /*GPU read log */
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	int pos = id >> BM_HASH;
	int	save;

	if(id < size) {
		save = devLogR[id];

		//Save to compressed log
		if (save > 1)
			global_bitmap[pos] = save;
	}
}

/****************************************************************************
 *	FUNCTIONS
 ****************************************************************************/
