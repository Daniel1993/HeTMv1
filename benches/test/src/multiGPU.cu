#include <cuda.h>
#include <stdio.h>

__global__ void someKernel(int devId)
{
  int cudaDevId;
  if (cudaGetDevice(&cudaDevId) != cudaSuccess) {
    printf("Cannot get device\n");
  }

  printf("hello from thread %i dev %i (cudaDev = %i)\n", threadIdx.x+blockDim.x*blockIdx.x, devId, cudaDevId);
  
}

int main(int argc, char **argv)
{
  int *dev0, *dev1; // TODO: array per device
  
  if (cudaSetDevice(0) != cudaSuccess) { printf("Cannot set dev 0\n"); }
  cudaStream_t streamDev0;
  cudaStreamCreate(&streamDev0);
  someKernel<<<1, 1, 0, streamDev0>>>(0);


  if (cudaSetDevice(1) != cudaSuccess) { printf("Cannot set dev 1\n"); }
  cudaStream_t streamDev1;
  cudaStreamCreate(&streamDev1);
  someKernel<<<1, 1, 0, streamDev1>>>(1);


  if (cudaSetDevice(0) != cudaSuccess) { printf("Cannot set dev 0\n"); }
  if (cudaStreamSynchronize(streamDev0) != cudaSuccess) { printf("Error in kernel dev 0\n"); }


  if (cudaSetDevice(1) != cudaSuccess) { printf("Cannot set dev 1\n"); }
  if (cudaStreamSynchronize(streamDev1) != cudaSuccess) { printf("Error in kernel dev 1\n"); }

  return 0;
}