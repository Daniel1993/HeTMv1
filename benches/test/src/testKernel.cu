#include "test.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand_kernel.h>
#include "helper_cuda.h"
#include "helper_timer.h"

#include "pr-stm-wrapper.cuh" // enables the granularity
#include "pr-stm-internal.cuh"

__global__ void test_kernel(PR_globalKernelArgs)
{
	PR_enterKernel();

	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int total = blockDim.x * gridDim.x;
  kernel_input_s *input = (kernel_input_s*)args.inBuf;
  int *mempool = input->mempool;
  int memPos;
  HeTM_GPU_log_s *GPU_log = (HeTM_GPU_log_s*)pr_args.pr_args_ext;

  if (input->roundId & 1) {
    memPos = idx;
  } else {
    memPos = total-idx-1;
  }
  mempool[memPos] = idx;

  memman_access_addr_dev(GPU_log->bmap, &mempool[memPos]); // track GPU writes
  SET_ON_RS_BMAP(&mempool[memPos]); // read-set

	PR_exitKernel();
}
// ----------------------

/****************************************************************************
 *	FUNCTIONS
/****************************************************************************/

void LaunchTestKernel(void *argsPtr)
{
  pr_buffer_s inBuf, outBuf;
  CUDA_CHECK_ERROR(cudaDeviceSynchronize(), ""); // sync the previous run

  cudaFuncSetCacheConfig(test_kernel, cudaFuncCachePreferL1);

  memman_select("test_kernel_input");
  kernel_input_s* input = (kernel_input_s*)memman_get_cpu(NULL);
  kernel_input_s* inputDev = (kernel_input_s*)memman_get_gpu(NULL);

  inBuf.buf = (void*)inputDev;
  inBuf.size = sizeof(kernel_input_s);
  outBuf.buf = NULL;
  outBuf.size = 0;
  pr_tx_args_s *pr_args = getPrSTMmetaData(PR_curr_dev);
  PR_prepareIO(pr_args, inBuf, outBuf);
  PR_run(test_kernel, pr_args);

  cudaError_t cudaStatus;
  if ((cudaStatus = cudaGetLastError()) != cudaSuccess) {
    printf("\nTransaction kernel launch failed. Error code: %s.\n", cudaGetErrorString(cudaStatus));
  }
}
