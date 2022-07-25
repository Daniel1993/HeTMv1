#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <iostream>
#include <memory>
#include <string>
#include <cstdio>
#include <cstdlib>

#include "cuda_util.h"

#define CONVERT_TO_GBPS(dataSize, timeMs) (((float)(dataSize)) / (((float)(timeMs) / 1000.0f) * (1024.0f*1024.0f*1024.0f)))

using namespace std;

int
main()
{
  int deviceCount;
  void **devMemDst;
  void *hostMemDst;
  void *hostMemDst2;
  void **devMemSrc;
  void *hostMemSrc;
  void *hostMemSrc2;
  const size_t COPY_SIZE = 1 * 1024*1024*1024; // 1GB
  cudaEvent_t *cpyEv;
  cudaStream_t s1;
  cudaStream_t s2;
  cudaStream_t s3;
  cudaStream_t s4;
  cudaStream_t s5;
  cudaStream_t s6;
  float time_ms1, time_ms2;

  CUDA_CHECK_ERROR(cudaGetDeviceCount(&deviceCount), "");

  printf("detected %d devices\n", deviceCount);
  CUDA_CHECK_ERROR(cudaMallocHost(&hostMemDst, COPY_SIZE), "");
  CUDA_CHECK_ERROR(cudaMallocHost(&hostMemDst2, COPY_SIZE), "");
  CUDA_CHECK_ERROR(cudaMallocHost(&hostMemSrc, COPY_SIZE), "");
  CUDA_CHECK_ERROR(cudaMallocHost(&hostMemSrc2, COPY_SIZE), "");
  devMemDst = (void**)malloc(deviceCount*sizeof(void*));
  devMemSrc = (void**)malloc(deviceCount*sizeof(void*));
  cpyEv = (cudaEvent_t*)malloc(2*deviceCount*sizeof(cudaEvent_t));
  CUDA_CHECK_ERROR(cudaStreamCreate(&s1), "");
  CUDA_CHECK_ERROR(cudaStreamCreate(&s2), "");
  CUDA_CHECK_ERROR(cudaStreamCreate(&s3), "");
  CUDA_CHECK_ERROR(cudaStreamCreate(&s4), "");
  CUDA_CHECK_ERROR(cudaStreamCreate(&s5), "");
  CUDA_CHECK_ERROR(cudaStreamCreate(&s6), "");
  for (int i = 0; i < deviceCount; ++i)
  {
    CUDA_CHECK_ERROR(cudaSetDevice(i), "");

    CUDA_CHECK_ERROR(cudaEventCreate(&(cpyEv[2*i])), "");
    CUDA_CHECK_ERROR(cudaEventCreate(&(cpyEv[2*i+1])), "");
    CUDA_CHECK_ERROR(cudaMalloc(&(devMemSrc[i]), COPY_SIZE), "");
    CUDA_CHECK_ERROR(cudaMalloc(&(devMemDst[i]), COPY_SIZE), "");
  }

  // TODO: agnostic of the number of devices

  int canDev0AccessDev1, canDev1AccessDev0;
  CUDA_CHECK_ERROR(cudaDeviceCanAccessPeer(&canDev0AccessDev1, 0, 1), "");
  CUDA_CHECK_ERROR(cudaDeviceCanAccessPeer(&canDev1AccessDev0, 1, 0), "");
  if (canDev0AccessDev1) {
    printf("dev0 can access dev1 directly\n");
    CUDA_CHECK_ERROR(cudaSetDevice(0), "");
    CUDA_CHECK_ERROR(cudaDeviceEnablePeerAccess(/*peer_id*/1, 0), "");
  }
  if (canDev1AccessDev0) {
    printf("dev1 can access dev0 directly\n");
    CUDA_CHECK_ERROR(cudaSetDevice(1), "");
    CUDA_CHECK_ERROR(cudaDeviceEnablePeerAccess(/*peer_id*/0, 0), "");
  }

  CUDA_CHECK_ERROR(cudaSetDevice(0), "");
  
  CUDA_CHECK_ERROR(cudaEventRecord(cpyEv[0], s1), "");
  CUDA_CPY_PtP_ASYNC(devMemDst[0], 0, devMemSrc[1], 1, COPY_SIZE, s1);
  CUDA_CHECK_ERROR(cudaEventRecord(cpyEv[1], s1), "");
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s1), "");
  CUDA_CHECK_ERROR(cudaEventSynchronize(cpyEv[0]), "");
  CUDA_CHECK_ERROR(cudaEventSynchronize(cpyEv[1]), "");
  CUDA_CHECK_ERROR(cudaEventElapsedTime(&time_ms1, cpyEv[0], cpyEv[1]), "");
  printf("CPY PtP dev1->dev0 took %f ms (%f GB/s)\n", time_ms1, CONVERT_TO_GBPS(COPY_SIZE, time_ms1));

  CUDA_CHECK_ERROR(cudaSetDevice(1), "");

  CUDA_CHECK_ERROR(cudaEventRecord(cpyEv[0], s1), "");
  CUDA_CPY_PtP_ASYNC(devMemDst[1], 1, devMemSrc[0], 0, COPY_SIZE, s1);
  CUDA_CHECK_ERROR(cudaEventRecord(cpyEv[1], s1), "");
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s1), "");
  CUDA_CHECK_ERROR(cudaEventSynchronize(cpyEv[0]), "");
  CUDA_CHECK_ERROR(cudaEventSynchronize(cpyEv[1]), "");
  CUDA_CHECK_ERROR(cudaEventElapsedTime(&time_ms1, cpyEv[0], cpyEv[1]), "");
  printf("CPY PtP dev0->dev1 took %f ms (%f GB/s)\n", time_ms1, CONVERT_TO_GBPS(COPY_SIZE, time_ms1));

  CUDA_CHECK_ERROR(cudaSetDevice(0), "");

  CUDA_CHECK_ERROR(cudaEventRecord(cpyEv[0], s1), "");
  CUDA_CPY_TO_DEV_ASYNC(devMemDst[0], hostMemSrc, COPY_SIZE, s1);
  CUDA_CHECK_ERROR(cudaEventRecord(cpyEv[1], s1), "");
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s1), "");
  CUDA_CHECK_ERROR(cudaEventSynchronize(cpyEv[0]), "");
  CUDA_CHECK_ERROR(cudaEventSynchronize(cpyEv[1]), "");
  CUDA_CHECK_ERROR(cudaEventElapsedTime(&time_ms1, cpyEv[0], cpyEv[1]), "");
  printf("CPY HtD host->dev0 took %f ms (%f GB/s)\n", time_ms1, CONVERT_TO_GBPS(COPY_SIZE, time_ms1));

  CUDA_CHECK_ERROR(cudaSetDevice(1), "");

  CUDA_CHECK_ERROR(cudaEventRecord(cpyEv[0], s1), "");
  CUDA_CPY_TO_DEV_ASYNC(devMemDst[1], hostMemSrc, COPY_SIZE, s1);
  CUDA_CHECK_ERROR(cudaEventRecord(cpyEv[1], s1), "");
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s1), "");
  CUDA_CHECK_ERROR(cudaEventSynchronize(cpyEv[0]), "");
  CUDA_CHECK_ERROR(cudaEventSynchronize(cpyEv[1]), "");
  CUDA_CHECK_ERROR(cudaEventElapsedTime(&time_ms1, cpyEv[0], cpyEv[1]), "");
  printf("CPY HtD host->dev1 took %f ms (%f GB/s)\n", time_ms1, CONVERT_TO_GBPS(COPY_SIZE, time_ms1));

  CUDA_CHECK_ERROR(cudaSetDevice(0), "");

  CUDA_CHECK_ERROR(cudaEventRecord(cpyEv[0], s1), "");
  CUDA_CPY_TO_HOST_ASYNC(hostMemDst, devMemSrc[0], COPY_SIZE, s1);
  CUDA_CHECK_ERROR(cudaEventRecord(cpyEv[1], s1), "");
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s1), "");
  CUDA_CHECK_ERROR(cudaEventSynchronize(cpyEv[0]), "");
  CUDA_CHECK_ERROR(cudaEventSynchronize(cpyEv[1]), "");
  CUDA_CHECK_ERROR(cudaEventElapsedTime(&time_ms1, cpyEv[0], cpyEv[1]), "");
  printf("CPY DtH dev0->host took %f ms (%f GB/s)\n", time_ms1, CONVERT_TO_GBPS(COPY_SIZE, time_ms1));

  CUDA_CHECK_ERROR(cudaSetDevice(1), "");

  CUDA_CHECK_ERROR(cudaEventRecord(cpyEv[0], s1), "");
  CUDA_CPY_TO_HOST_ASYNC(hostMemDst, devMemSrc[1], COPY_SIZE, s1);
  CUDA_CHECK_ERROR(cudaEventRecord(cpyEv[1], s1), "");
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s1), "");
  CUDA_CHECK_ERROR(cudaEventSynchronize(cpyEv[0]), "");
  CUDA_CHECK_ERROR(cudaEventSynchronize(cpyEv[1]), "");
  CUDA_CHECK_ERROR(cudaEventElapsedTime(&time_ms1, cpyEv[0], cpyEv[1]), "");
  printf("CPY DtH dev1->host took %f ms (%f GB/s)\n", time_ms1, CONVERT_TO_GBPS(COPY_SIZE, time_ms1));

  // CUDA_CHECK_ERROR(cudaSetDevice(1), "");

  CUDA_CHECK_ERROR(cudaSetDevice(0), "");
  CUDA_CHECK_ERROR(cudaEventRecord(cpyEv[0], s1), "");
  CUDA_CPY_PtP_ASYNC(devMemDst[0], 0, devMemSrc[1], 1, COPY_SIZE, s1);
  CUDA_CHECK_ERROR(cudaEventRecord(cpyEv[1], s1), "");
  CUDA_CHECK_ERROR(cudaSetDevice(1), "");
  // CUDA_CHECK_ERROR(cudaEventRecord(cpyEv[2], s2), "");
  CUDA_CPY_PtP_ASYNC(devMemDst[1], 1, devMemSrc[0], 0, COPY_SIZE, s2);
  // CUDA_CHECK_ERROR(cudaEventRecord(cpyEv[3], s2), "");
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s1), "");
  CUDA_CHECK_ERROR(cudaEventSynchronize(cpyEv[0]), "");
  CUDA_CHECK_ERROR(cudaEventSynchronize(cpyEv[1]), "");
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s2), "");
  CUDA_CHECK_ERROR(cudaEventSynchronize(cpyEv[2]), "");
  CUDA_CHECK_ERROR(cudaEventSynchronize(cpyEv[3]), "");
  CUDA_CHECK_ERROR(cudaEventElapsedTime(&time_ms1, cpyEv[0], cpyEv[1]), "");
  // CUDA_CHECK_ERROR(cudaEventElapsedTime(&time_ms2, cpyEv[2], cpyEv[3]), "");
  printf("CPY concurrent PtP dev0->dev1 took %f ms (%f GB/s)\n", time_ms1, CONVERT_TO_GBPS(COPY_SIZE, time_ms1));
  // printf("CPY concurrent PtP dev1->dev0 took %f ms (%f GB/s)\n", time_ms2, CONVERT_TO_GBPS(COPY_SIZE, time_ms2));


  CUDA_CHECK_ERROR(cudaSetDevice(0), "");
  CUDA_CPY_TO_HOST_ASYNC(hostMemDst, devMemSrc[0], COPY_SIZE, s1);
  CUDA_CHECK_ERROR(cudaSetDevice(1), "");
  CUDA_CPY_TO_HOST_ASYNC(hostMemDst, devMemSrc[1], COPY_SIZE, s2);
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s1), "");
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s2), "");

  CUDA_CHECK_ERROR(cudaSetDevice(0), "");
  CUDA_CPY_TO_DEV_ASYNC(devMemDst[0], hostMemSrc, COPY_SIZE, s1);
  CUDA_CHECK_ERROR(cudaSetDevice(1), "");
  CUDA_CPY_TO_DEV_ASYNC(devMemDst[1], hostMemSrc, COPY_SIZE, s2);
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s1), "");
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s2), "");

  CUDA_CHECK_ERROR(cudaSetDevice(0), "");
  CUDA_CPY_TO_DEV_ASYNC(devMemDst[0], hostMemSrc, COPY_SIZE, s1);
  CUDA_CHECK_ERROR(cudaSetDevice(1), "");
  CUDA_CPY_TO_HOST_ASYNC(hostMemDst, devMemSrc[1], COPY_SIZE, s2);
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s1), "");
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s2), "");

  CUDA_CHECK_ERROR(cudaSetDevice(0), "");
  CUDA_CPY_TO_HOST_ASYNC(hostMemDst, devMemSrc[0], COPY_SIZE, s1);
  CUDA_CHECK_ERROR(cudaSetDevice(1), "");
  CUDA_CPY_TO_DEV_ASYNC(devMemDst[1], hostMemSrc, COPY_SIZE, s2);
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s1), "");
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s2), "");

  CUDA_CHECK_ERROR(cudaSetDevice(0), "");
  CUDA_CPY_TO_DEV_ASYNC(devMemDst[0], hostMemSrc, COPY_SIZE, s1);
  CUDA_CPY_TO_HOST_ASYNC(hostMemDst, devMemSrc[0], COPY_SIZE, s2);
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s1), "");
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s2), "");

  CUDA_CHECK_ERROR(cudaSetDevice(1), "");
  CUDA_CPY_TO_DEV_ASYNC(devMemDst[1], hostMemSrc, COPY_SIZE, s3);
  CUDA_CPY_TO_HOST_ASYNC(hostMemDst, devMemSrc[1], COPY_SIZE, s4);
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s3), "");
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s4), "");

  CUDA_CHECK_ERROR(cudaSetDevice(0), "");
  CUDA_CPY_TO_DEV_ASYNC(devMemDst[0], hostMemSrc, COPY_SIZE, s1);
  CUDA_CPY_TO_HOST_ASYNC(hostMemDst, devMemSrc[0], COPY_SIZE, s2);
  CUDA_CHECK_ERROR(cudaSetDevice(1), "");
  CUDA_CPY_TO_DEV_ASYNC(devMemDst[1], hostMemSrc, COPY_SIZE, s3);
  CUDA_CPY_TO_HOST_ASYNC(hostMemDst, devMemSrc[1], COPY_SIZE, s4);
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s1), "");
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s2), "");
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s3), "");
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s4), "");

  CUDA_CHECK_ERROR(cudaSetDevice(0), "");
  CUDA_CPY_TO_DEV_ASYNC(devMemDst[0], hostMemSrc, COPY_SIZE, s1);
  CUDA_CPY_TO_HOST_ASYNC(hostMemDst, devMemSrc[0], COPY_SIZE, s2);
  CUDA_CHECK_ERROR(cudaSetDevice(1), "");
  CUDA_CPY_TO_DEV_ASYNC(devMemDst[1], hostMemSrc2, COPY_SIZE, s3);
  CUDA_CPY_TO_HOST_ASYNC(hostMemDst2, devMemSrc[1], COPY_SIZE, s4);
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s1), "");
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s2), "");
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s3), "");
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s4), "");

  CUDA_CHECK_ERROR(cudaSetDevice(0), "");
  CUDA_CPY_TO_DEV_ASYNC(devMemDst[0], hostMemSrc, COPY_SIZE, s1);
  CUDA_CPY_TO_HOST_ASYNC(hostMemDst, devMemSrc[0], COPY_SIZE, s2);
  CUDA_CHECK_ERROR(cudaSetDevice(1), "");
  CUDA_CPY_TO_DEV_ASYNC(devMemDst[1], hostMemSrc, COPY_SIZE, s3);
  CUDA_CPY_TO_HOST_ASYNC(hostMemDst, devMemSrc[1], COPY_SIZE, s4);
  CUDA_CPY_PtP_ASYNC(devMemDst[0], 0, devMemSrc[1], 1, COPY_SIZE, s5);
  CUDA_CPY_PtP_ASYNC(devMemDst[1], 1, devMemSrc[0], 0, COPY_SIZE, s6);
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s1), "");
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s2), "");
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s3), "");
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s4), "");
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s5), "");
  CUDA_CHECK_ERROR(cudaStreamSynchronize(s6), "");

  for (int i = 0; i < deviceCount; ++i)
  {
    cudaEventDestroy(cpyEv[i]);
    cudaEventDestroy(cpyEv[i+1]);
    cudaFree(devMemSrc[i]);
    cudaFree(devMemDst[i]);
  }
  free(devMemSrc);
  free(devMemDst);
  cudaFreeHost(hostMemSrc);
  cudaFreeHost(hostMemSrc2);
  cudaFreeHost(hostMemDst);
  cudaFreeHost(hostMemDst2);

  return 0;
}

