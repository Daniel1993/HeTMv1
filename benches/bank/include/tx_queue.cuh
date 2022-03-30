#ifndef GUARD_H_TX_QUEUE_
#define GUARD_H_TX_QUEUE_

#define SHARED_VALUE 100

#include "bankKernel.cuh"

#include "pr-stm-wrapper.cuh"

/****************************************************************************
 *	STRUCTURES
 ****************************************************************************/

//

/****************************************************************************
 *	KERNELS
 ****************************************************************************/

__global__ void queue_generate_kernel(long *state,  unsigned int *queue, int per_thread, int lower, int upper); //to setup random seed

/****************************************************************************
 *	FUNCTIONS
 ****************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t queue_Init(queue_t *q, cuda_t *c, int sr, int q_size, long *states);

#ifdef __cplusplus
}
#endif

// unsigned int* readQueue(queue_t *q, int size, int queue_id);
//
// void queue_Delete(queue_t * q);


#endif /* GUARD_H_TX_QUEUE_ */
