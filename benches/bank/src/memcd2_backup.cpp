#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define __USE_GNU

#include <random>

#include "bank.hpp"
#include "hetm-cmp-kernels.cuh"
#include "bank_aux.h"
#include "CheckAllFlags.h"
#include "zipf_dist.hpp"

/* ################################################################### *
* GLOBALS
* ################################################################### */

// static std::random_device randDev{};
static std::mt19937 generator;
static zipf_distribution<int, double> *zipf_dist = NULL;

// global
thread_data_t parsedData;
int isInterBatch = 0;
size_t accountsSize;
size_t sizePool;
void* gpuMempool;

// -----------------------------------------------------------------------------
const size_t NB_OF_BUFFERS = 2; // 2 good + 2 bad

const static int NB_CPU_TXS_PER_THREAD = 2048;

int *GPUoutputBuffer, *CPUoutputBuffer;
int *GPUInputBuffer, *CPUInputBuffer;

static unsigned long long input_seed = 0x3F12514A3F12514A;

// CPU output --> must fill the buffer until the log ends (just realloc on full)
static size_t currMaxCPUoutputBufferSize, currCPUoutputBufferPtr = 0;
static size_t maxGPUoutputBufferSize;
static size_t size_of_GPU_input_buffer, size_of_CPU_input_buffer;
static int lockOutputBuffer = 0;

FILE *GPU_input_file = NULL;
FILE *CPU_input_file = NULL;

// TODO: break the dataset
// static cudaStream_t *streams;
// -----------------------------------------------------------------------------

// TODO: input <isSET, key, val>
// 	int setKernelPerc = parsedData.set_percent;

// TODO: memory access

static int fill_GPU_input_buffers()
{
	int buffer_last = size_of_GPU_input_buffer/sizeof(int);
	GPU_input_file = fopen(parsedData.GPUInputFile, "r");

	if (zipf_dist == NULL) {
		generator.seed(input_seed);
		zipf_dist = new zipf_distribution<int, double>(parsedData.nb_accounts * parsedData.num_ways);
	}

	memman_select("GPU_input_buffer_good");
	int *cpu_ptr = (int*)memman_get_cpu(NULL);

// #if BANK_PART == 3
	// unsigned rnd = 12345723; //RAND_R_FNC(input_seed);
	unsigned rnd; // = (*zipf_dist)(generator);
	for (int i = 0; i < buffer_last; ++i) {
		fscanf(GPU_input_file, "%i\n", &rnd); // RAND_R_FNC(input_seed);
		cpu_ptr[i] = rnd; // GPU_ACCESS(rnd, parsedData.nb_accounts) + (i * parsedData.nb_accounts); // not the same key
		cpu_ptr[i] &= (unsigned)-2;
	}

	memman_select("GPU_input_buffer_bad");
	cpu_ptr = (int*)memman_get_cpu(NULL);

	// cpu_ptr[0] = 0; // deterministic abort
	for (int i = 0; i < buffer_last; ++i) {
		fscanf(GPU_input_file, "%i\n", &rnd); // RAND_R_FNC(input_seed);
		cpu_ptr[i] = rnd; // INTERSECT_ACCESS_GPU(rnd, parsedData.nb_accounts) + (i * parsedData.nb_accounts);
	}
}

static int fill_CPU_input_buffers()
{
	int good_buffers_last = size_of_CPU_input_buffer/sizeof(int);
	int bad_buffers_last = 2*size_of_CPU_input_buffer/sizeof(int);
	CPU_input_file = fopen(parsedData.CPUInputFile, "r");

	if (zipf_dist == NULL) {
		generator.seed(input_seed);
		zipf_dist = new zipf_distribution<int, double>(parsedData.nb_accounts * parsedData.num_ways);
	}

// #if BANK_PART == 3
	for (int i = 0; i < good_buffers_last; ++i) {
		// unsigned rnd = (*zipf_dist)(generator); // RAND_R_FNC(input_seed);
		unsigned rnd;
		fscanf(CPU_input_file, "%i\n", &rnd); // RAND_R_FNC(input_seed);
		CPUInputBuffer[i] = rnd; // CPU_ACCESS(rnd, parsedData.nb_accounts) + (i * parsedData.nb_accounts);
		CPUInputBuffer[i] |= 1;
	}
	// CPUInputBuffer[good_buffers_last] = 0; // deterministic abort
	for (int i = good_buffers_last; i < bad_buffers_last; ++i) {
		// unsigned rnd = (*zipf_dist)(generator); // RAND_R_FNC(input_seed);
		unsigned rnd; // RAND_R_FNC(input_seed);
		fscanf(CPU_input_file, "%i\n", &rnd); // RAND_R_FNC(input_seed);
		CPUInputBuffer[i] = rnd; // INTERSECT_ACCESS_CPU(rnd, parsedData.nb_accounts) + (i * parsedData.nb_accounts);
		// CPUInputBuffer[i] |= 1;
	}
}

memcd_get_output_t cpu_GET_kernel(memcd_t *memcd, int *input_key, unsigned input_clock)
{
	// const int size_of_hash = 16;
  int z = 0;
  // char key[size_of_hash]; // assert(sizeof(char) == 1)
	// uintptr_t hash[2]; // assert(sizeof(uintptr_t) == 8)
	size_t modHash, setIdx;
  memcd_get_output_t response;
	int key = *input_key;

	// 1) hash key
	modHash = key;
	// TODO: this would be nice, but we lose control of where the key goes to
	// memset(key, 0, size_of_hash);
	// memcpy(key, input_key, sizeof(int));
	// MurmurHash3_x64_128(key, size_of_hash, 0, hash);
	// modHash = hash[0];
	// modHash += hash[1];

	// 2) setIdx <- hash % nbSets
	setIdx = modHash % memcd->nbSets;
	setIdx = setIdx * memcd->nbWays;

  /* Allow overdrafts */
  TM_START(z, RW);

	// 3) find in set the key, if not found write not found in the output
	for (int i = 0; i < memcd->nbWays; ++i)
	{
		size_t newIdx = setIdx + i;
		int readState = TM_LOAD(&memcd->state[newIdx]);
		int readKey   = TM_LOAD(&memcd->key[newIdx]);
		if ((readState & MEMCD_VALID) && readKey == key) {
			// found it!
			int readVal = TM_LOAD(&memcd->val[newIdx]);
			int* volatile ptr_ts = &memcd->ts[newIdx];
			TM_STORE(ptr_ts, input_clock); // TODO: set state to isRead
			response.isFound = 1;
			response.value   = readVal;
			break;
		}
	}
	TM_COMMIT;

  return response;
}

void cpu_SET_kernel(memcd_t *memcd, int *input_key, int *input_value, unsigned input_clock)
{
	// const int size_of_hash = 16;
  int z = 0;
  // char key[size_of_hash]; // assert(sizeof(char) == 1)
	// uintptr_t hash[2]; // assert(sizeof(uintptr_t) == 8)
	size_t modHash, setIdx;
  memcd_get_output_t response;
	volatile int key = *input_key;
	volatile int val = *input_value;

	// 1) hash key
	modHash = key;
	// TODO: this would be nice, but we lose control of where the key goes to
	// memset(key, 0, size_of_hash);
	// memcpy(key, input_key, sizeof(int));
	// MurmurHash3_x64_128(key, size_of_hash, 0, hash);
	// modHash = hash[0];
	// modHash += hash[1];

	// 2) setIdx <- hash % nbSets
	setIdx = modHash % memcd->nbSets;
	setIdx = setIdx * memcd->nbWays;

  /* Allow overdrafts */
  TM_START(z, RW);

	// 3) find in set the key, if not found write not found in the output
	int idxFound = -1;
	int idxEvict = -1;
	int isInCache = 0;
	unsigned TS   = (unsigned)-1; // largest TS
	for (int i = 0; i < memcd->nbWays; ++i)
	{
		size_t newIdx = setIdx + i;
		int readState = TM_LOAD(&memcd->state[newIdx]);
		if (((readState & MEMCD_VALID) == 0) && idxFound == -1) {
			// found empty spot
			idxFound = i;
			continue;
		}
		int readKey = TM_LOAD(&memcd->key[newIdx]);
		if (readKey == key && ((readState & MEMCD_VALID) != 0)) {
			// found the key in the cache --> just use this spot
			isInCache = 1;
			idxFound = i;
			break;
		}
		unsigned readTS = TM_LOAD(&memcd->ts[newIdx]);
		if (readTS < TS) { // look for the older entry
			TS = readTS;
			idxEvict = i;
		}
	}
	size_t newIdx;
	if (idxFound == -1) {
		newIdx = setIdx + idxEvict;
	} else {
		newIdx = setIdx + idxFound;
	}
	int* volatile ptr_ts = &memcd->ts[newIdx]; // TODO: optimizer screws the ptrs
	int* volatile ptr_val = &memcd->val[newIdx];
	TM_STORE(ptr_ts, input_clock);
	TM_STORE(ptr_val, val);
	if (!isInCache) {
		volatile int newState = MEMCD_VALID|MEMCD_WRITTEN;
		int* volatile ptr_key = &memcd->key[newIdx];
		int* volatile ptr_state = &memcd->state[newIdx];
		TM_STORE(ptr_key, key);
		TM_STORE(ptr_state, newState);
	}
	TM_COMMIT;
}

void cpu_SET_kernel_NOTX(memcd_t *memcd, int *input_key, int *input_value, unsigned input_clock)
{
	size_t modHash, setIdx;
	memcd_get_output_t response;
	int key = *input_key;
	int val = *input_value;

	// 1) hash key
	modHash = key;

	// 2) setIdx <- hash % nbSets
	setIdx = modHash % memcd->nbSets;
	setIdx = setIdx * memcd->nbWays;

	// 3) find in set the key, if not found write not found in the output
	int idxFound = -1;
	int idxEvict = -1;
	int isInCache = 0;
	unsigned TS   = (unsigned)-1; // largest TS
	for (int i = 0; i < memcd->nbWays; ++i)
	{
		size_t newIdx = setIdx + i;
		int readState = memcd->state[newIdx];
		if (((readState & MEMCD_VALID) == 0) && idxFound == -1) {
			// found empty spot
			idxFound = i;
			continue;
		}
		int readKey = memcd->key[newIdx];
		if (readKey == key && ((readState & MEMCD_VALID) != 0)) {
			// found the key in the cache --> just use this spot
			isInCache = 1;
			idxFound = i;
			break;
		}
		unsigned readTS = memcd->ts[newIdx];
		if (readTS < TS) { // look for the older entry
			TS = readTS;
			idxEvict = i;
		}
	}
	size_t newIdx;
	if (idxFound == -1) {
		newIdx = setIdx + idxEvict;
	} else {
		newIdx = setIdx + idxFound;
	}
	memcd->ts[newIdx] = input_clock;
	memcd->val[newIdx] = val;
	if (!isInCache) {
		int newState = MEMCD_VALID|MEMCD_WRITTEN;
		memcd->key[newIdx] = key;
		memcd->state[newIdx] = newState;
	}
}

/* ################################################################### *
* TRANSACTION THREADS
* ################################################################### */

static void test(int id, void *data)
{
  thread_data_t *d = &((thread_data_t *)data)[id];
  cuda_t       *cd = d->cd;
  memcd_t   *memcd = d->memcd;

  int input_key;
  int input_val;

  int nbSets = memcd->nbSets;
  int nbWays = memcd->nbWays;
  unsigned rnd;
	int rndOpt = RAND_R_FNC(d->seed);
	static thread_local int curr_tx = 0;

	// TODO: this goes into the input
	float setKernelPerc = parsedData.set_percent * 1000;

	int good_buffers_start = 0;
	int bad_buffers_start = size_of_CPU_input_buffer/sizeof(int);

	int buffers_start = isInterBatch ? bad_buffers_start : good_buffers_start;
	input_key = CPUInputBuffer[buffers_start+id*NB_CPU_TXS_PER_THREAD + curr_tx];
	input_val = input_key;

	/* 100% * 1000*/
	if (rndOpt % 100000 < setKernelPerc) {
		// Set kernel
		cpu_SET_kernel(d->memcd, &input_key, &input_val, *d->memcd->globalTs);
	} else {
		// Get kernel
		memcd_get_output_t res; // TODO: write it in the output buffer
		res = cpu_GET_kernel(d->memcd, &input_key, *d->memcd->globalTs);
	}
	curr_tx += 1;
	curr_tx = curr_tx % NB_CPU_TXS_PER_THREAD;

  d->nb_transfer++;
  d->global_commits++;
}

static void beforeCPU(int id, void *data)
{
  thread_data_t *d = &((thread_data_t *)data)[id];
	// also, setup the memory
	// call_cuda_check_memcd((int*)gpuMempool, accountsSize/sizeof(int));
}

static void afterCPU(int id, void *data)
{
  thread_data_t *d = &((thread_data_t *)data)[id];
  BANK_GET_STATS(d);
}

/* ################################################################### *
* CUDA THREAD
* ################################################################### */

// TODO: add a beforeBatch and afterBatch callbacks

static void before_batch(int id, void *data)
{
	thread_local static unsigned long seed = 0x0012112A3112514A;
	isInterBatch = RAND_R_FNC(seed) % 100 < parsedData.shared_percent;
	__sync_synchronize();

	for (int j = 0; j < HETM_NB_DEVICES; ++j) {
		memman_select_device(j);
		if (isInterBatch) {
			memman_select("GPU_input_buffer_bad");
		} else {
			memman_select("GPU_input_buffer_good");
		}
		memman_cpy_to_gpu(NULL, NULL, *hetm_batchCount[j]);
	}
}

static void after_batch(int id, void *data)
{
	for (int j = 0; j < HETM_NB_DEVICES; ++j) {
		memman_select_device(j);
		memman_select("GPU_output_buffer");
		memman_cpy_to_cpu(NULL, NULL, *hetm_batchCount[j]);
		// TODO: conflict mechanism
	}
}

static void choose_policy(int, void*) {
  // // -----------------------------------------------------------------------
  // int idGPUThread = HeTM_shared_data.nbCPUThreads;
  // long TXsOnCPU = 0,
	// 	TXsOnGPU = parsedData.GPUthreadNum*parsedData.GPUblockNum*parsedData.trans;
  // for (int i = 0; i < HeTM_shared_data.nbCPUThreads; ++i) {
  //   TXsOnCPU += HeTM_shared_data.threadsInfo[i].curNbTxs;
  // }
	//
  // // TODO: this picks the one with higher number of TXs
	// // TODO: GPU only gets the stats in the end --> need to remove the dropped TXs
	// HeTM_stats_data.nbTxsGPU += TXsOnGPU;
  // if (TXsOnCPU > TXsOnGPU) {
  //   HeTM_gshared_data.policy = HETM_GPU_INV;
	// 	if (HeTM_is_interconflict()) {
	// 		HeTM_stats_data.nbDroppedTxsGPU += TXsOnGPU;
	// 	} else {
	// 		HeTM_stats_data.nbCommittedTxsGPU += TXsOnGPU;
	// 	}
  // } else {
  //   HeTM_gshared_data.policy = HETM_CPU_INV;
	// 	HeTM_stats_data.nbCommittedTxsGPU += TXsOnGPU;
  // }
  // // -----------------------------------------------------------------------
}

static void test_cuda(int id, void *data)
{
  thread_data_t    *d = &((thread_data_t *)data)[id];
  cuda_t          *cd = d->cd;
  account_t *base_ptr = d->memcd->key;

	*(d->memcd->globalTs) += 1;
	for (int j = 0; j < HETM_NB_DEVICES; ++j) {
		memman_select_device(j);
		memman_select("memcd_global_ts");
		memman_cpy_to_gpu(NULL, NULL, *hetm_batchCount);
	}

  jobWithCuda_runMemcd(d, cd, base_ptr, *(d->memcd->globalTs));
}

static void afterGPU(int id, void *data)
{
  thread_data_t *d = &((thread_data_t *)data)[id];

  // int ret = bank_sum(d->bank);
  // if (ret != 0) {
    // this gets CPU transactions running
    // printf("error at batch %i, expect %i but got %i\n", HeTM_stats_data.nbBatches, 0, ret);
  // }

  d->nb_transfer               = HeTM_stats_data.nbCommittedTxsGPU;
  d->nb_transfer_gpu_only      = HeTM_stats_data.nbTxsGPU;
  d->nb_aborts                 = HeTM_stats_data.nbAbortsGPU;
  d->nb_aborts_1               = HeTM_stats_data.timeGPU; /* duration; */ // TODO
  d->nb_aborts_2               = HeTM_stats_data.timeCMP * 1000;
  d->nb_aborts_locked_read     = HeTM_stats_data.nbBatches; // TODO: these names are really bad
  d->nb_aborts_locked_write    = HeTM_stats_data.nbBatchesSuccess; /*counter_sync;*/ // TODO: successes?
  d->nb_aborts_validate_read   = HeTM_stats_data.timeAfterCMP * 1000;
  d->nb_aborts_validate_write  = HeTM_stats_data.timePRSTM * 1000; // TODO
  d->nb_aborts_validate_commit = 0; /* trans_cmp; */ // TODO
  d->nb_aborts_invalid_memory  = 0;
  d->nb_aborts_killed          = 0;
  d->locked_reads_ok           = 0;
  d->locked_reads_failed       = 0;
  d->max_retries               = HeTM_stats_data.timeGPU; // TODO:
  printf("nb_transfer=%li\n", d->nb_transfer);
  printf("nb_batches=%li\n", d->nb_aborts_locked_read);

  // leave this one
  printf("CUDA thread terminated after %li(%li successful) run(s). \nTotal cuda execution time: %f ms.\n",
    HeTM_stats_data.nbBatches, HeTM_stats_data.nbBatchesSuccess, HeTM_stats_data.timeGPU);
}

/* ################################################################### *
*
* MAIN
*
* ################################################################### */

int main(int argc, char **argv)
{
  memcd_t *memcd;
  // barrier_t cuda_barrier;
  int i, j, ret = -1;
  thread_data_t *data;
  pthread_t *threads;
  barrier_t barrier;
  sigset_t block_set;

  memset(&parsedData, 0, sizeof(thread_data_t));

  PRINT_FLAGS();

  // ##########################################
  // ### Input management
  bank_parseArgs(argc, argv, &parsedData);

	// ---------------------------------------------------------------------------
	maxGPUoutputBufferSize = parsedData.GPUthreadNum*parsedData.GPUblockNum*parsedData.trans;
	currMaxCPUoutputBufferSize = maxGPUoutputBufferSize; // realloc on full

	// malloc_or_die(streams, parsedData.trans);
	// for (int i = 0; i < parsedData.trans; ++i) {
	// 	cudaStreamCreate(streams + i);
	// }
	// parsedData.streams = streams;

	// malloc_or_die(GPUoutputBuffer, maxGPUoutputBufferSize);
	memman_alloc_dual("GPU_output_buffer", maxGPUoutputBufferSize*sizeof(memcd_get_output_t), 0);
	GPUoutputBuffer = (int*)memman_get_gpu(NULL);

	size_of_GPU_input_buffer = maxGPUoutputBufferSize*sizeof(int);

	// TODO: this parsedData.nb_threads is what comes of the -n input
	size_of_CPU_input_buffer = parsedData.nb_threads*sizeof(int) * NB_CPU_TXS_PER_THREAD;

	// malloc_or_die(GPUInputBuffer, size_of_GPU_input_buffer*NB_OF_BUFFERS);
	memman_alloc_gpu("GPU_input_buffer", size_of_GPU_input_buffer, NULL, 0);
	GPUInputBuffer = (int*)memman_get_gpu(NULL);
	memman_alloc_cpu("GPU_input_buffer_good", size_of_GPU_input_buffer, GPUInputBuffer, 0);
	memman_alloc_cpu("GPU_input_buffer_bad", size_of_GPU_input_buffer, GPUInputBuffer, 0);

	// CPU Buffers
	malloc_or_die(CPUInputBuffer, size_of_CPU_input_buffer * 2); // good and bad
	malloc_or_die(CPUoutputBuffer, currMaxCPUoutputBufferSize); // kinda big

	fill_GPU_input_buffers();
	fill_CPU_input_buffers();
	// ---------------------------------------------------------------------------

  HeTM_init((HeTM_init_s){
#if CPU_INV == 1
    .policy       = HETM_CPU_INV,
#else /* GPU_INV */
    .policy       = HETM_GPU_INV,
#endif /**/
    .nbCPUThreads = parsedData.nb_threads,
    .nbGPUBlocks  = parsedData.GPUblockNum,
    .nbGPUThreads = parsedData.GPUthreadNum,
#if HETM_CPU_EN == 0
    .isCPUEnabled = 0,
    .isGPUEnabled = 1
#elif HETM_GPU_EN == 0
    .isCPUEnabled = 1,
    .isGPUEnabled = 0
#else /* both on */
    .isCPUEnabled = 1,
    .isGPUEnabled = 1
#endif
  });

	memman_alloc_dual("memcd_global_ts", sizeof(unsigned), 0);

	malloc_or_die(memcd, 1);
	memcd->nbSets = parsedData.nb_accounts;
	memcd->nbWays = parsedData.num_ways;
	memcd->globalTs = (unsigned*)memman_get_cpu(NULL);
	accountsSize = memcd->nbSets*memcd->nbWays*sizeof(account_t);
	sizePool = accountsSize * 4;
  HeTM_mempool_init(sizePool); // <K,V,TS,STATE>

  // TODO:
  parsedData.nb_threadsCPU = HeTM_shared_data.nbCPUThreads;
  parsedData.nb_threads    = HeTM_shared_data.nbThreads;
  bank_check_params(&parsedData);

  malloc_or_die(data, parsedData.nb_threads + 1);
  memset(data, 0, (parsedData.nb_threads + 1)*sizeof(thread_data_t)); // safer
  parsedData.dthreads     = data;
  // ##########################################

  jobWithCuda_exit(NULL); // Reset Cuda Device

  malloc_or_die(threads, parsedData.nb_threads);

	// mallocs 4 arrays (accountsSize * NUMBER_WAYS * 4)
  HeTM_alloc((void**)&memcd->key, &gpuMempool, sizePool); // <K,V,TS,STATE>
	memcd->val   = memcd->key + (memcd->nbSets*memcd->nbWays);
	memcd->ts    = memcd->val + (memcd->nbSets*memcd->nbWays);
	memcd->state = memcd->ts  + (memcd->nbSets*memcd->nbWays);

  memset(memcd->key, 0, sizePool);
  parsedData.memcd = memcd;

  DEBUG_PRINT("Initializing GPU.\n");

  cuda_t *cuda_st;
  cuda_st = jobWithCuda_init(memcd->key, parsedData.nb_threadsCPU,
    sizePool, parsedData.trans, 0, parsedData.GPUthreadNum, parsedData.GPUblockNum,
    parsedData.hprob, parsedData.hmult);

  jobWithCuda_initMemcd(cuda_st, parsedData.num_ways, parsedData.nb_accounts,
    parsedData.set_percent, parsedData.shared_percent);
	cuda_st->memcd_array_size = accountsSize;

  parsedData.cd = cuda_st;
  //DEBUG_PRINT("Base: %lu %lu \n", bank->accounts, &bank->accounts);
  if (cuda_st == NULL) {
    printf("CUDA init failed.\n");
    exit(-1);
  }

  /* Init STM */
  printf("Initializing STM\n");

	/* POPULATE the cache */
	for (int i = 0; i < size_of_CPU_input_buffer/sizeof(int); ++i) {
		cpu_SET_kernel_NOTX(memcd, &CPUInputBuffer[i], &CPUInputBuffer[i], 0);
	}
	memman_select("GPU_input_buffer_good");
	int *gpu_buffer_cpu_ptr = (int*)memman_get_cpu(NULL);
	for (int i = 0; i < size_of_GPU_input_buffer/sizeof(int); ++i) {
		cpu_SET_kernel_NOTX(memcd, &gpu_buffer_cpu_ptr[i], &gpu_buffer_cpu_ptr[i], 0);
	}

	// for (int i = 0; i < 32*4; ++i) {
	// 	printf("%i : KEY=%i STATE=%i\n", i, memcd->key[i], memcd->state[i]);
	// }

	CUDA_CPY_TO_DEV(gpuMempool, memcd->key, sizePool);
	// printf(" >>>>>>>>>>>>>>> PASSOU AQUI!!!\n");
	// call_cuda_check_memcd((int*)gpuMempool, accountsSize/sizeof(int));
	// TODO: does not work: need to set the bitmap in order to copy
	// HeTM_mempool_cpy_to_gpu(NULL); // copies the populated cache to GPU

  TM_INIT(parsedData.nb_threads);
  // ###########################################################################
  // ### Start iterations ######################################################
  // ###########################################################################
  for(j = 0; j < parsedData.iter; j++) { // Loop for testing purposes
    //Clear flags
    HeTM_set_is_stop(0);
    global_fix = 0;

    // ##############################################
    // ### create threads
    // ##############################################
		printf(" >>> Creating %d threads\n", parsedData.nb_threads);
    for (i = 0; i < parsedData.nb_threads; i++) {
      /* SET CPU AFFINITY */
      /* INIT DATA STRUCTURE */
      // remove last iter status
      parsedData.reads = parsedData.writes = parsedData.updates = 0;
      parsedData.nb_aborts = 0;
      parsedData.nb_aborts_2 = 0;
      memcpy(&data[i], &parsedData, sizeof(thread_data_t));
      data[i].id      = i;
      data[i].seed    = input_seed * (i ^ 12345); // rand();
      data[i].cd      = cuda_st;
    }
    // ### end create threads

    /* Start threads */
    HeTM_after_gpu_finish(afterGPU);
    HeTM_before_cpu_start(beforeCPU);
    HeTM_after_cpu_finish(afterCPU);
    HeTM_start(test, test_cuda, data);
		HeTM_after_batch(after_batch);
		HeTM_before_batch(before_batch);

		HeTM_choose_policy(choose_policy);

    printf("STARTING...(Run %d)\n", j);

    TIMER_READ(parsedData.start);

    if (parsedData.duration > 0) {
      nanosleep(&parsedData.timeout, NULL);
    } else {
      sigemptyset(&block_set);
      sigsuspend(&block_set);
    }
    HeTM_set_is_stop(1);

    TIMER_READ(parsedData.end);
    printf("STOPPING...\n");

    /* Wait for thread completion */
    HeTM_join_CPU_threads();

    // reset accounts
    // memset(bank->accounts, 0, bank->size * sizeof(account_t));

    TIMER_READ(parsedData.last);
    bank_between_iter(&parsedData, j);
    bank_printStats(&parsedData);
  }

	// for (int i = 0; i < memcd->nbSets; ++i) {
	// 	for (int j = 0; j < memcd->nbWays; ++j) {
	// 		printf("%i ", memcd->key[i*memcd->nbSets + j]);
	// 	}
	// 	printf("\n");
	// }

	// call_cuda_check_memcd((int*)gpuMempool, accountsSize/sizeof(int));

  /* Cleanup STM */
  TM_EXIT();
  /*Cleanup GPU*/
  jobWithCuda_exit(cuda_st);
  free(cuda_st);
  // ### End iterations ########################################################

  bank_statsFile(&parsedData);

  /* Delete bank and accounts */
  HeTM_mempool_destroy();
  free(memcd);

  free(threads);
  free(data);

  return EXIT_SUCCESS;
}
