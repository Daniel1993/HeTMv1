#ifndef _MEMCD_KERNEL
#define _MEMCD_KERNEL

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_defines.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <curand_kernel.h>
#include "helper_cuda.h"
#include "helper_timer.h"
#include <time.h>

// TODO: extend PR-STM here in order to have the logs...
#include "cmp_kernels.cuh"
#include "bank.hpp"

/****************************************************************************
 *	STRUCTURES
 ****************************************************************************/

// TODO: repeated struct
typedef struct {
	long size;
	int  TransEachThread;
	int  hashNum;
	int  threadNum;
	int  blockNum;
  int  num_ways;
} cuda_config;

typedef struct {
  GRANULE_T* accounts;
  int *ts_vec;
  int clock_value;
  unsigned int *tx_queue;
} memcd_input_s;

/*********************************
 *	writeKernelTransaction()
 *
 *  memcached GET kernel
 **********************************/
__global__ void HeTM_memcd_read(PR_globalKernelArgs);

  volatile long* a,	/*values in global memory*/
	int * ts_vec,										/*timestamp array*/
	cuda_output_t * output,								/*produced results*/
	volatile uint_64* mymutex,							/*store lock,version,owner in format version*10000+owner*10+lock*/
	int * dev_abortcount,								/*record how many times aborted*/
	unsigned int size,									/*size of the data-set*/
	int clock_value,									/*sync timestamp*/
	unsigned int * tx_queue,							/*transaction queue*/
	int * devLogR,
	int * devLogW);

/*********************************
 *	writeKernelTransaction()
 *
 *  memcached SET kernel
 **********************************/
__global__ void HeTM_memcd_write(PR_globalKernelArgs);

  volatile long* a,	/*values in global memory*/
	int * ts_vec,										/*timestamp array*/
	cuda_output_t * output,								/*produced results*/
	volatile uint_64* mymutex,							/*store lock,version,owner in format version*10000+owner*10+lock*/
	int * dev_abortcount,								/*record how many times aborted*/
	unsigned int size,									/*size of the data-set*/
	int clock_value,									/*sync timestamp*/
	unsigned int * tx_queue,							/*transaction queue*/
	int * devLogR,
	int * devLogW);

#endif /* MEMCD */
