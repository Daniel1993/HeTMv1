#ifndef CUDA_WRAPPER_H_GUARD_
#define CUDA_WRAPPER_H_GUARD_

#include <cuda.h>
#include "hetm-log.h"
#include "cuda_defines.h"
#include "hetm-types.h"

#include <cuda_runtime.h>
// #include <curand.h>
// #include <curand_kernel.h>
// #include "helper_timer.h"

#include "pr-stm-wrapper.cuh"
#include "shared.h"

#define PADDING              0

// TODO: Number of comparisons sent to the GPU before restarting the kernel
#define NB_CMP_RETRIES       2

/****************************************************************************
 *	STRUCTURES
 ****************************************************************************/

// TODO
// typedef struct HeTM_shared_ {
//
// } HeTM_shared_s;

typedef struct cuda_info {
	account_t *host_a;     //host values array
  account_t *dev_a;      //values array
	account_t *dev_b;      //device copy of host values array
	account_t *dev_bckp;   //device copy of host values array
	account_t *dev_zc;     //zero_copy array

	// TODO: not in use anymore
	int *dev_bm;				   //compressed log array
	int bm_size;

	// TODO: delete
	int *dev_LogR;			   //Read Log array
	int *dev_LogW;			   //Write Log array

	void *devStates;       //randseed array

	int size;
	int threadNum;
	int blockNum;
	int TransEachThread;
	float set_percent;
	int num_ways;
	int num_sets;

	size_t memcd_array_size;
	size_t memcd_nbSets;
	size_t memcd_nbWays;

	// TODO: this is memcd
	// int clock;
	// int *version;
  // queue_t *q;
	// unsigned int *gpu_queue;
	// cuda_output_t *output;
	// size_t queue_size;
	// long run_size;
	// int blockNumG;
} cuda_t;

typedef struct stream_info {
	int id;
	int count;
	int isCmpDone;
	int isThreshold;
	int maxC;
	int isCudaError;
	cudaStream_t st;
	pthread_mutex_t mutex;
} stream_t;

/****************************************************************************
 *	MACROS
 ****************************************************************************/

#define EXPECTED_COMMITS_GPU(d)			d->TransEachThread*d->blockNum*d->threadNum

/****************************************************************************
 *	FUNCTIONS
 ****************************************************************************/

// TODO: put GRANULE_T or account_t
cuda_t * jobWithCuda_init(account_t *base, int nbCPUThreads, int size, int trans, int hash, int tx, int bl, int hprob, float hmult);
void jobWithCuda_initMemcd(cuda_t *cd, int ways, int sets, float wr, int sr); // memcd

int jobWithCuda_run(cuda_t *d, account_t *a);

int jobWithCuda_runMemcd(void *thread_data, cuda_t *d, account_t *a, int clock);
int jobWithCuda_runEmptyKernel(void *thread_data, cuda_t *d, account_t *a, int clock);

account_t* jobWithCuda_swap(cuda_t *d);

int jobWithCuda_cpyDatasetToGPU(cuda_t *d, account_t *b);

void jobWithCuda_getStats(cuda_t *d, long *ab, long *com);

void jobWithCuda_exit(cuda_t * d);

// TODO: memcd
unsigned int* readQueue(unsigned int *seed, queue_t *q, int size, int queue_id);
void queue_Delete(queue_t *q);

#endif /* CUDA_WRAPPER_H_GUARD_ */
