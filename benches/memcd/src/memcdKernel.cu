#include "cuda_wrapper.h"
#include "bankKernel.cuh"
#include "setupKernels.cuh"

#include "hetm.cuh"

#include "memman.hpp"
#include "knlman.hpp"

using namespace memman;
using namespace knlman;

typedef struct offload_bank_tx_thread_ {
  cuda_t *d;
  thread_data_t *thread_data;
  account_t *a;
  int clock;
} offload_memcd_tx_thread_s;

static void offloadMemcdTxThread(void *argsPtr);
static void offloadEmptyTxThread(void *argsPtr);

extern int nbOfGPUSetKernels;

extern KnlObj *HeTM_memcdWriteTx;
extern KnlObj *HeTM_memcdReadTx;

void call_cuda_check_memcd(PR_GRANULE_T* gpuMempool, size_t size)
{
  memcd_check<<<32,4>>>(gpuMempool, size);
}

// inits the remaining stuff for memcd
void jobWithCuda_initMemcd(cuda_t *cd, int ways, int sets, float wr, int sr)
{
  // queue_t *queue;
  // queue = (queue_t*)malloc( sizeof(queue_t) );

  cd->set_percent = wr;
	// cd->clock = 0;

  // TODO: this was in cuda_configInit
  cd->num_ways = ways > 0 ? ways : NUMBER_WAYS;
  cd->num_sets = sets > 0 ? sets : NUMBER_SETS;

	// CUDA_CHECK_ERROR(queue_Init(queue, cd, sr, QUEUE_SIZE, (long*)cd->devStates), "");
  HeTM_setup_memcdReadTx(cd->blockNum, cd->threadNum);
  HeTM_setup_memcdWriteTx(cd->blockNum, cd->threadNum);

  // TODO: use buffer alloc API
  // CUDA_CHECK_ERROR(cudaMalloc((void **)&cd->output, cd->threadNum*cd->blockNum*sizeof(cuda_output_t)), "");
  // cd->q = queue;
  CUDA_CHECK_ERROR(cuda_configCpyMemcd(cd), "");
}


int jobWithCuda_runMemcd(void *thread_data, cuda_t *d, account_t *a, int clock) // TODO: rerun?
{
  bool err = 1;
  cudaError_t cudaStatus;
  static offload_memcd_tx_thread_s offload_thread_args;

  while (err) {
    err = 0;

    CHECK_ERROR_CONTINUE(cudaSetDevice(DEVICE_ID));

    offload_thread_args.thread_data = (thread_data_t*)thread_data;
    offload_thread_args.d = d;
    offload_thread_args.a = a;

    HeTM_async_request((HeTM_async_req_s){
      .args = (void*)&offload_thread_args,
      .fn = offloadMemcdTxThread
    });

    //Check for errors
    cudaStatus = cudaGetLastError();
  }

  if (cudaStatus != cudaSuccess) {
    printf("\nTransaction kernel launch failed. Error code: %s.\n", cudaGetErrorString(cudaStatus));
    return 0;
  }

  return 1;
}


int jobWithCuda_runEmptyKernel(void *thread_data, cuda_t *d, account_t *a, int clock) // TODO: rerun?
{
  bool err = 1;
  cudaError_t cudaStatus;
  static offload_memcd_tx_thread_s offload_thread_args;

  while (err) {
    err = 0;

    CHECK_ERROR_CONTINUE(cudaSetDevice(DEVICE_ID));

    offload_thread_args.thread_data = (thread_data_t*)thread_data;
    offload_thread_args.d = d;
    offload_thread_args.a = a;

    HeTM_async_request((HeTM_async_req_s){
      .args = (void*)&offload_thread_args,
      .fn = offloadEmptyTxThread
    });

    //Check for errors
    cudaStatus = cudaGetLastError();
  }

  if (cudaStatus != cudaSuccess) {
    printf("\nTransaction kernel launch failed. Error code: %s.\n", cudaGetErrorString(cudaStatus));
    return 0;
  }

  return 1;
}


static void offloadMemcdTxThread(void *argsPtr)
{
  offload_memcd_tx_thread_s *args = (offload_memcd_tx_thread_s*)argsPtr;
  // thread_data_t *cd = args->thread_data;
  cuda_t *d = args->d;
  account_t *a = args->a;
  static thread_local unsigned seed = 82913126;

  cudaError_t cudaStatus;

  CUDA_CHECK_ERROR(cudaSetDevice(DEVICE_ID), "");

  // 100% * 1000 --- cd->seed is not working correcly
  unsigned decider = RAND_R_FNC(seed) % 100000;
  // int read_size;

  // TODO
  HeTM_bankTx_s bankTx_args = {
    .knlArgs = {
      .d = d,
      .a = a,
    },
    .clbkArgs = NULL
  };

  // decider = 100; // always GET

  // TODO: check if the entry object is well defined
  // knlman_set_entry_object(&bankTx_args);
  if (decider >= d->set_percent * 1000) {
		//MEMCACHED GET
    HeTM_memcdReadTx->blocks = {.x = d->blockNum, .y = 1, .z = 1};
    HeTM_memcdReadTx->threads = {.x = d->threadNum, .y = 1, .z = 1};
    HeTM_memcdReadTx->Run(Config::GetInstance()->SelDev());
		// d->run_size = d->blockNum*d->threadNum;
		// read_size = d->blockNum*d->threadNum;
	} else {
		//MEMCACHED SET
    HeTM_memcdWriteTx->blocks = {.x = d->blockNum, .y = 1, .z = 1};
    HeTM_memcdWriteTx->threads = {.x = d->threadNum, .y = 1, .z = 1};
    HeTM_memcdWriteTx->Run(Config::GetInstance()->SelDev());
    nbOfGPUSetKernels++;
		// d->run_size = d->blockNumG*d->threadNum;
		// read_size = d->blockNumG*d->threadNum;
	}

  //Check for errors
  cudaStatus = cudaGetLastError();

  if (cudaStatus != cudaSuccess) {
    printf("\nTransaction kernel launch failed. Error code: %s.\n", cudaGetErrorString(cudaStatus));
  }
}

static void offloadEmptyTxThread(void *argsPtr)
{
  pr_tx_args_s *pr_args = getPrSTMmetaData(0); // TODO: is this in use?

  CUDA_CHECK_ERROR(cudaSetDevice(DEVICE_ID), "");
  PR_run(emptyKernelTx, pr_args);
}
