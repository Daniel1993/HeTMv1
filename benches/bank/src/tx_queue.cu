#include <pthread.h>

#include "tx_queue.cuh"


/****************************************************************************
 *	GLOBALS
 ****************************************************************************/

 //

/****************************************************************************
 *	KERNELS
 ****************************************************************************/

// to setup random seed
__global__ void queue_generate_kernel(
  long *state,
  unsigned int *queue,
  int lower,
  int upper
) {
	unsigned int count = upper - lower;
	unsigned int gen = 0;
	int loop = 1;
	int id = 0;
  id = threadIdx.x + blockDim.x*gridDim.y*blockIdx.x + blockIdx.y;
	unsigned int *local = &queue[id*MEMCD_NB_TRANSFERS];

  #pragma unroll
	for (int i = 0; i < MEMCD_NB_TRANSFERS; i++)
	{
    gen = RAND_R_FNC(state[id]) % count;
    gen += lower;
		local[i] = gen;
	}
}

 /****************************************************************************
 *	FUNCTIONS
 ****************************************************************************/

extern "C"
cudaError_t queue_Init(queue_t *q, cuda_t *cd, int shared_rate, int q_size, long *devStates)
{
	cudaError_t cudaStatus = cudaErrorMemoryAllocation;
  unsigned int *temp_q;
	int err = 0, lower, upper;
	int per_thread = 1 + (q_size)/(cd->blockNum * cd->threadNum);
  // rounds the number of requests
	q_size = cd->blockNum * cd->threadNum * per_thread; // WTF?
  cd->queue_size = q_size;
  size_t byte_q_size = q_size*MEMCD_NB_TRANSFERS*sizeof(unsigned int);
  int nbAccounts = cd->size / cd->num_ways;

	q->shared_rate = shared_rate;

	//Host malloc for cpu queue
	q->cpu_queue.size = q_size*MEMCD_NB_TRANSFERS;
  printf("Host/Dev q->cpu_queue.size=%i\n", q->cpu_queue.size);
	q->cpu_queue.current = 0;
	pthread_mutex_init(&q->cpu_queue.lock, NULL);

  // TODO: for multi-device: cudaHostAllocPortable
  memman_alloc_dual("memcd_cpu_queue", byte_q_size, 0); // TODO: only one queue allowed
  q->cpu_queue.values = (unsigned int*)memman_get_cpu(NULL);
  temp_q = (unsigned int*)memman_get_gpu(NULL);

	//Host malloc for gpu queue
	q->gpu_queue.size = q_size*MEMCD_NB_TRANSFERS;
	q->gpu_queue.current = 0;
	pthread_mutex_init(&q->gpu_queue.lock, NULL);

  // TODO: for multi-device: cudaHostAllocPortable
  memman_alloc_cpu("memcd_gpu_queue", byte_q_size, temp_q, 0);
  q->gpu_queue.values = (unsigned int*)memman_get_cpu(NULL);
  printf("Alloc q->gpu_queue.values (%p) with size %zu\n", q->gpu_queue.values, byte_q_size);

	//Host malloc for shared queue
	q->shared_queue.size = q_size*MEMCD_NB_TRANSFERS;
	q->shared_queue.current = 0;
	pthread_mutex_init(&q->shared_queue.lock, NULL);

  // TODO: for multi-device: cudaHostAllocPortable
  memman_alloc_cpu("memcd_shared_queue", byte_q_size, temp_q, 0);
  q->shared_queue.values = (unsigned int*)memman_get_cpu(NULL);

	dim3 blocksQueue(cd->blockNum, per_thread);
	/*
	 * GEN CPU QUEUE
	 */
	//  Kernel Gen
	upper = nbAccounts >> 2;
  lower = 0;
	queue_generate_kernel <<< blocksQueue, cd->threadNum >>>(devStates, temp_q, lower, upper);
	cudaStatus = cudaDeviceSynchronize();	//synchronize threads
	if (cudaStatus != cudaSuccess) {
		printf("queue_generate_kernel failed!");
	}
	//Copy to host
  memman_select("memcd_cpu_queue");
  memman_cpy_to_cpu(NULL, NULL, *hetm_batchCount);
  CUDA_CHECK_ERROR(cudaDeviceSynchronize(), ""); // waits copy back
  cd->gpu_queue = temp_q; // TODO:

	/*
	 * GEN SHARED QUEUE
	 */
	//  Kernel Gen
	upper = nbAccounts >> 1;
  lower = nbAccounts >> 2;
	queue_generate_kernel<<<blocksQueue, cd->threadNum>>>(devStates, temp_q, lower, upper);
	cudaStatus = cudaDeviceSynchronize();	//synchronize threads
	if (cudaStatus != cudaSuccess) {
		printf("queue_generate_kernel failed!");
	}
  memman_select("memcd_shared_queue");
  memman_cpy_to_cpu(NULL, NULL, *hetm_batchCount);
  CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "");

	/*
	 * GEN GPU QUEUE
	 */
	//  Kernel Gen
	upper = nbAccounts;
  lower = nbAccounts >> 1;
	queue_generate_kernel<<<blocksQueue, cd->threadNum>>>(devStates, temp_q, lower, upper);
	cudaStatus = cudaDeviceSynchronize();	//synchronize threads
	if (cudaStatus != cudaSuccess) {
		printf("queue_generate_kernel failed!");
	}
  memman_select("memcd_gpu_queue");
  memman_cpy_to_cpu(NULL, NULL, *hetm_batchCount);
  CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "");

	// cudaFree(temp_q); // TODO: temp_q is freed?

  // TODO: I think some initialization is needed

	return cudaStatus;
}

#define SPIN_CAS_INC(ptr, inc, max) ({ \
  int load, sum, res; \
  while (1) { \
    res = load = *(ptr); \
    sum = load + inc; \
    if (sum >= max) { \
      if(__sync_bool_compare_and_swap(ptr, load, 0)) { \
        res = 0; \
        break; \
      } \
    } else { \
      if(__sync_bool_compare_and_swap(ptr, load, sum)) { \
        break; \
      } \
    } \
  } \
  res; \
})

unsigned int* readQueue_cpu(queue_t *q, int size)
{
  int count;

	if (size > q->cpu_queue.size) {
    return NULL;
  }

  count = SPIN_CAS_INC(&(q->cpu_queue.current), size, q->cpu_queue.size);
	return &q->cpu_queue.values[count];
}

unsigned int* readQueue_gpu(queue_t *q, int size)
{
  if (size > q->gpu_queue.size) {
    return NULL;
  }

  int count;

  count = SPIN_CAS_INC(&(q->gpu_queue.current), size, q->gpu_queue.size);
  return &q->gpu_queue.values[count];
}

extern "C"
unsigned int* readQueue(unsigned int *seed, queue_t *q, int size, int queue_id)
{
  int count;
	int percent = RAND_R_FNC(*seed) % 100;

	if (percent < q->shared_rate) {
		if (size > q->shared_queue.size) {
      return NULL;
    }

    count = SPIN_CAS_INC(&(q->shared_queue.current), size, q->shared_queue.size);
		return &q->shared_queue.values[count];
	}

	if (queue_id == 0) {
    return readQueue_cpu(q, size);
	}

	if (queue_id == 1) {
    return readQueue_gpu(q, size);
	}

	return NULL;
}

extern "C"
void queue_Delete(queue_t *q)
{
	if(q!=NULL) {
		cudaFreeHost(q->shared_queue.values);
		cudaFreeHost(q->cpu_queue.values);
		cudaFreeHost(q->gpu_queue.values);
		free(q);
	}
}
