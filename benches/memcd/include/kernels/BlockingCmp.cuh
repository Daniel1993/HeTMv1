#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cmp_kernels.cuh"

// não chegou a ser usado

#if ARCH == FERMI
__device__ int checkTransactionKernel_FERMI(
	int * dev_flag, /*Flag in global memory*/
	int * shared_gpu,
	int * local_stm,
	int id
) {
	int	i, n, pos, val;
	int comp = 0;

	//Comparison
	#pragma unroll
	for (i = 0; i < checkSharedSize; i++) {
		pos = id+i & (checkSharedSize-1);
		val = shared_gpu[pos];

		#pragma unroll
		for (n = 0; n < checkLocalSize; n++)
			comp += !(val ^ local_stm[n]);

		if (comp !=0) *dev_flag = 1;

		if (*dev_flag != 0)  break;
	}

	return comp;
}
#else /* ARCH != FERMI */
__device__ int checkTransactionKernel_OTHER(
	int * dev_flag, /*Flag in global memory*/
	int * shared_gpu,
	int * local_stm,
	int id
) {
	int j, cmp;
	int	i, n, pos, val;
	int comp = 0;

	#pragma unroll
	for (i = 0; i < checkSharedSize; i += 32) {
		pos = id+i & (checkSharedSize-1);
		cmp = val = shared_gpu[pos];

		#pragma unroll
		for (j = 0; j < 32; j++) {
			#pragma unroll
			for (n = 0; n < checkLocalSize; n++) {
				//comp += !(cmp^ local_stm[n] );
				comp = cmp == local_stm[n] ? 1 : comp;
			}

			cmp = __shfl(val, (threadIdx.x + j) & 31);
		}

		if (comp != 0) *dev_flag = 1;
		if (*dev_flag != 0) break;
	}

	return comp;
}
#endif /* ARCH */
