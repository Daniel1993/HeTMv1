#include <stdio.h>
#include <iostream>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand_kernel.h>
#include "helper_cuda.h"
#include "helper_timer.h"

#include "hetm-log.h"
#include "knlman.hpp"
#include "hetm.cuh"
#include "memman.hpp"

#include "setupKernels.cuh"
#include "pr-stm-wrapper.cuh"

#include "cuda_wrapper.h"

#include "bankKernel.cuh"

using namespace memman;
using namespace knlman;

extern KnlObj *HeTM_finalTxLog2;
extern KnlObj *HeTM_bankTx;
extern KnlObj *HeTM_memcdWriteTx;
extern KnlObj *HeTM_memcdReadTx;

//#define	DEBUG_CUDA
//#define	DEBUG_CUDA2

#define ARCH             FERMI
#define DEVICE_ID        0

//Support for the lazylog implementation
// moved to cmp_kernels.cuh
// #define EXPLICIT_LOG_BLOCK     (TransEachThread * BANK_NB_TRANSFERS)
// #define EXPLICIT_LOG_SIZE      (blockNum * threadNum * EXPLICIT_LOG_BLOCK)	//size of the lazy lock

//Support for the compressed log implementation
#define readMask         0b01	//Set entry to write
#define writeMask        0b10	//Set entry to read

//versions in global memory	10 bits(version),0 bits (padding),20 bits(owner threadID),1 bit(LOCKED),1 bit(pre-locked)
#define	offVers	22
#define offOwn	2
#define offLock	1

#define finalIdx          (threadIdx.x+blockIdx.x*blockDim.x)
#define newLock(x,y,z)    ( ((x) << offVers) | ((y) << offOwn) | (z))

#define uint_64				int

typedef struct offload_bank_tx_thread_ {
  cuda_t *d;
  account_t *a;
} offload_bank_tx_thread_s;

// static int kernelLaunched = 0;
static TIMER_T beginTimer;

static void offloadBankTxThread(void *argsPtr); // bank_tx

/* ################################################################################################# *
 * HOST CODE
 * ################################################################################################# */

/****************************************
 *	jobWithCuda_init(size,hostLogMax)
 *
 *	Description:	Initialize the GPU, by allocating all necessary memory,
 *					transferring static data and running the setup kernel.
 *
 *	Args:
 *		int size		: Size (in integers) to allocate for working set data
 *		long ** accounts: Pointer to host array pointer
 *      int hostLogMax	: Maximum number of entries the host transaction log can contain
 *		long * b:		:(Optional) Host log address, for use with zero copy
 *
 *	Returns:
 *		cuda_t: 	Custom structure containing all essential CUDA pointers/data
 *					or null in case of failure
 *
 ****************************************/
 // TODO: put GRANULE_T or account_t
cuda_t *
jobWithCuda_init(
  account_t *accounts,
  int nbCPUThreads,
  int size,
  int trans,
  int hash,
  int tx,
  int bl,
  int hprob,
  float hmult
) {
  //int *a = (int *)malloc(size * sizeof(int));
  cuda_config cuda_info;    //Cuda config info
  cuda_t *c_data;
  //cudaProfilerStop();  //Stop unnecessary profiling

  // char *someStr1 = "passou aqui 1!\n";
  // char *someStr2 = "passou aqui 2!\n";
  // char *someStr3 = "passou aqui 3!\n";
  // char *someStr4 = "passou aqui 4!\n";
  // char *someStr5 = "passou aqui 5!\n";
  // char *someStr6 = "passou aqui 6!\n";
  // auto test1 = std::string(someStr1);
  
  // Choose which GPU to run on, change this on a multi-GPU system.
  CUDA_CHECK_ERROR(cudaSetDevice(0), "Device failed");

  // printf("%s", test1.c_str());
  cuda_info = cuda_configInit(size, trans, hash, tx, bl, hprob, hmult);

  // auto test2 = std::string(someStr2);
  // printf("%s", test2.c_str());
  TIMER_READ(beginTimer);


  // Init check Tx kernel
  // TODO: init EXPLICIT
  // auto test3 = std::string(someStr3);
  // printf("%s", test3.c_str());
  HeTM_setup_finalTxLog2();
  // auto test4 = std::string(someStr4);
  // printf("%s", test4.c_str());
  HeTM_setup_bankTx(cuda_info.blockNum, cuda_info.threadNum);

  // auto test5 = std::string(someStr5);
  // printf("%s", test5.c_str());
  time_t t;
  time(&t);

  // TODO: Do we really need the pointers in this struct?

  // auto test6 = std::string(someStr6);
  // printf("%s", test6.c_str());
  //Save cuda pointers
  c_data = (cuda_t*)malloc( sizeof(cuda_t) * HETM_NB_DEVICES );

  for (int j = 0; j < Config::GetInstance()->NbGPUs(); ++j)
  {
    Config::GetInstance()->SelDev(j);
    HeTM_initCurandState(j);
    cuda_configCpy(cuda_info); // THIS IS PER GPU!
    c_data[j].host_a = accounts;
    c_data[j].dev_a = (account_t*)HeTM_map_addr_to_gpu(j, accounts);
    c_data[j].devStates = HeTM_shared_data[j].devCurandState;
    c_data[j].size      = cuda_info.size;
    c_data[j].dev_zc    = NULL;
    c_data[j].threadNum = cuda_info.threadNum;
    c_data[j].blockNum  = cuda_info.blockNum;
    // c_data->blockNumG = cuda_info.blockNum / 1;
    c_data[j].TransEachThread = cuda_info.TransEachThread;
  }

  return c_data;
}

/****************************************
 *	jobWithCuda_run(d,a)
 *
 *	Description:	Update working set data and run transaction kernel.
 *					Failures are only detected on subsequent calls to jobWithCuda_wait()
 *
 *	Args:
 *		cuda_t * d		: Custom structure containing all essential transaction kernel CUDA pointers/data
 *      long * a		: Working set data
 *
 *	Returns:
 *		int:		1 in case of success, 0 otherwise
 *
 ****************************************/
int
jobWithCuda_run(
  cuda_t *d,
  account_t *a
) {
  static offload_bank_tx_thread_s offload_thread_args;

  offload_thread_args.d = d;
  offload_thread_args.a = a;

  // TODO: if overlap kernel
  offloadBankTxThread((void*)&offload_thread_args);

  // HeTM_async_request((HeTM_async_req_s){
  //   .args = (void*)&offload_thread_args,
  //   .fn = offloadBankTxThread
  // });
  return 1;
}

/****************************************
 *	jobWithCuda_swap(d,a)
 *
 *	Description:	Overwrites devices working set with the hosts
 *
 *	Args:
 *		cuda_t  d		: Custom structure containing all essential transaction kernel CUDA pointers/data
 *
 *	Returns:
 *		long *:			0 in case of failure, a pointer otherwise
 *
 ****************************************/
account_t* jobWithCuda_swap(cuda_t *d)
{
  return d->host_a;
}

/****************************************
 *	jobWithCuda_getStats(cd,ab,com)
 *
 *	Description:	Get cuda stats
 *
 *	Args:
 *		cuda_t * d	: Custom structure containing all essential CUDA pointers/data
 *		int * ab	: (Optional) Pointer to store tx kernel abort counter
 *		int * com	: (Optional) Pointer to store tx kernel commit counter
 *
 *	Returns:
 *		(None)
 *
 ****************************************/
void
jobWithCuda_getStats(
  cuda_t *d,
  long *ab,
  long *com
) {
  cudaError_t cudaStatus;
  int err = 1;

  while(err) {
    err = 0;

    PR_global_data_s *d;
    *ab  = 0;
    *com = 0;
    for (int j = 0; j < Config::GetInstance()->NbGPUs(); j++)
    {
      CHECK_ERROR_CONTINUE(cudaDeviceSynchronize());
      HeTM_bankTx_cpy_IO();
      PR_curr_dev = j;
      d = &(PR_global[PR_curr_dev]);
      //Transfer aborts
      if (ab != NULL) {
        *ab += d->PR_nbAborts;
      }

      //Transfer commits
      if (com != NULL) {
        *com += d->PR_nbCommits;
      }
    }


  }

  if (cudaStatus != cudaSuccess) {
    printf("\nStats: Error code is: %s.\n", cudaGetErrorString(cudaStatus));
    return;
  }
}

/****************************************
 *	jobWithCuda_exit(d)
 *
 *	Description:	Finish Cuda execution, free device memory and reset device.
 *
 *	Args:
 *		cuda_t d	: Custom structure containing all essential CUDA pointers/data
 *
 *	Returns:		(none)
 *
 ****************************************/
void jobWithCuda_exit(cuda_t * d)
{
  cudaError_t cudaStatus;

  cudaStatus = cudaSetDevice(DEVICE_ID);
  if (cudaStatus != cudaSuccess) {
    printf("cudaDeviceSynchronize returned error code: %d\n", cudaStatus);
  }

  // cudaDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.

  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    printf("cudaDeviceSynchronize returned error code: %d\n", cudaStatus);
  }

  if(d != NULL) {
    HeTM_destroy();
    for (int j = 0; j < HETM_NB_DEVICES; ++j) {
      HeTM_destroyCurandState(j);
      PR_curr_dev = j;
      PR_teardown();
    }
  }

  // HeTM_teardown_bankTx();
  // HeTM_teardown_finalTxLog2();

  // cudaDeviceReset(); // This is crashing on CUDA 9.0

  return;
}

static void
offloadBankTxThread(
  void *argsPtr
) {
  offload_bank_tx_thread_s *args = (offload_bank_tx_thread_s*)argsPtr;
  cuda_t *d = args->d; // array per device
  account_t *a = args->a;
  int devId = Config::GetInstance()->SelDev();
  MemObj *m_bankTx = HeTM_bankTx->entryObj->GetMemObj(devId);
  HeTM_bankTx_s *bankTx_args = (HeTM_bankTx_s*)m_bankTx->host;

  HeTM_bankTx->blocks  = (knlman_dim3_s){ .x = d->blockNum,  .y = 1, .z = 1 };
  HeTM_bankTx->threads = (knlman_dim3_s){ .x = d->threadNum, .y = 1, .z = 1 };
  bankTx_args->knlArgs.d = d;
  bankTx_args->knlArgs.a = a;
  bankTx_args->clbkArgs = NULL;
  m_bankTx->CpyHtD(NULL); // TODO: this copy is not necessary

  HeTM_bankTx->Run(devId);
}
