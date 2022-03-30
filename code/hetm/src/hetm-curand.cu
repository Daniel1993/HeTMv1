#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include "helper_cuda.h"
#include "helper_timer.h"
#include <time.h>

#include "hetm-log.h"

#include "bitmap.hpp"
#include "hetm.cuh"
#include "memman.hpp"

// --------------------
__constant__ unsigned PR_seed = 0xA1792F2B; // TODO: set seed
// --------------------

__global__ void HeTM_setupCurand(void *args)
{
	// curandState *state = (curandState*)args;
	// int id = threadIdx.x + blockDim.x*blockIdx.x;
	// curand_init(PR_seed, id, 0, &state[id]);
	long *state = (long*)args;
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	state[id] = id * PR_seed;
	RAND_R_FNC(state[id]);
}

// __device__ int dev_memman_access_addr_gran(void *bmap, void *base, void *addr,
// 	size_t gran, size_t bits, unsigned char valToSet)
// {
//   memman_access_addr_gran(bmap, base, addr, gran, bits, valToSet);
//   return 0;
// }

// __device__ int dev_memman_access_addr(void *bmap, void *addr, unsigned char valToSet)
// {
//   memman_bmap_s *key_bmap = (memman_bmap_s*)bmap;
//   char *bytes = key_bmap->ptr;
//   dev_memman_access_addr_gran(bytes, key_bmap->base, addr, key_bmap->gran, key_bmap->bits, valToSet);
//   return 0;
// }
