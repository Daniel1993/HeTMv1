#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define __USE_GNU

#include <random>
#include <thread>

#include "bank.hpp"
#include "hetm-cmp-kernels.cuh"
#include "bank_aux.h"
#include "CheckAllFlags.h"
#include "zipf_dist.hpp"

#include "memcd.h"
#include "memman.hpp"

using namespace memman;

/* ################################################################### *
* GLOBALS
* ################################################################### */

// global
thread_data_t parsedData;

// static std::random_device randDev{};
static std::mt19937 generator;
// static zipf_distribution<int, double> *zipf_dist = NULL;

static int probStealBatch = 0;
static int isCPUBatchSteal = 0;
static int isGPUBatchSteal = 0;

static unsigned memcached_global_clock = 0;

size_t size_of_CPU_input_buffer;
size_t size_of_GPU_input_buffer;
size_t maxGPUoutputBufferSize;
size_t currMaxCPUoutputBufferSize;
size_t accountsSize; // TODO: these are probably local variables
size_t sizePool;
void *gpuMempool;

int nbOfGPUSetKernels = 0; // called extern in memcdKernel.cu
int *GPUoutputBuffer[HETM_NB_DEVICES]; // called extern in input_buffer.c
int *CPUoutputBuffer;
int *GPUInputBuffer[HETM_NB_DEVICES];
int *CPUInputBuffer;

memman::MemObjOnDev GPU_input_buffer_good;
memman::MemObjOnDev GPU_input_buffer_bad;
memman::MemObjOnDev GPU_input_buffer;
memman::MemObjOnDev GPU_output_buffer;
extern memman::MemObjOnDev memcd_global_ts;

#ifndef REQUEST_LATENCY
#define REQUEST_LATENCY 10.0
#endif /* REQUEST_LATENCY */

#ifndef REQUEST_GRANULARITY
#define REQUEST_GRANULARITY 1000000
#endif /* REQUEST_LATENCY */

#ifndef REQUEST_GPU
#define REQUEST_GPU 0.5 /* TXs that go into GPU */
#endif /* REQUEST_LATENCY */

#ifndef REQUEST_CPU
#define REQUEST_CPU 0.4 /* TXs that go into CPU */
#endif /* REQUEST_LATENCY */

// shared is the remainder of 1-(REQUEST_GPU+REQUEST_CPU)

// -----------------------------------------------------------------------------

static int NB_GPU_TXS; // must be loaded at run time

enum NAME_QUEUE {
	CPU_QUEUE = 0,
	GPU_QUEUE = 1,
	SHARED_QUEUE = 2
};

static volatile long *startInputPtr, *endInputPtr;

static unsigned long long input_seed = 0x3F12514A3F12514A;

// CPU output --> must fill the buffer until the log ends (just realloc on full)


// TODO: break the dataset
// static cudaStream_t *streams;
// -----------------------------------------------------------------------------

// TODO: input <isSET, key, val>
// 	int setKernelPerc = parsedData.set_percent;

// TODO: memory access

static int fill_GPU_input_buffers()
{
#if BANK_PART == 1 /* shared queue is %3 == 2 */
	GPUbufferReadFromFile_NO_CONFLS();
#elif BANK_PART == 2 /* shared queue is the other device */
	GPUbufferReadFromFile_CONFLS();
#elif BANK_PART == 3 /* unif rand */
	GPUbufferReadFromFile_UNIF_RAND();
#elif BANK_PART == 4
	GPUbuffer_NO_CONFLS();
#elif BANK_PART == 5
	GPUbuffer_UNIF_2();
#elif BANK_PART == 6
	GPUbuffer_ZIPF_2();
#endif
	return 0;
}

static int fill_CPU_input_buffers()
{
#if BANK_PART == 1 /* shared queue is %3 == 1 */
	CPUbufferReadFromFile_NO_CONFLS();
#elif BANK_PART == 2 /* shared queue is the other device */
	CPUbufferReadFromFile_CONFLS();
#elif BANK_PART == 3 /* unif rand */
	CPUbufferReadFromFile_UNIF_RAND();
#elif BANK_PART == 4
	CPUbuffer_NO_CONFLS();
#elif BANK_PART == 5
	CPUbuffer_UNIF_2();
#elif BANK_PART == 6
	CPUbuffer_ZIPF_2();
#endif
	return 0;
}

static void wait_ms(float msTime)
{
	struct timespec duration;
	float secs = msTime / 1000.0f;
	float nanos = (secs - floor(secs)) * 1e9;
	duration.tv_sec = (long)secs;
  duration.tv_nsec = (long)nanos;
	nanosleep(&duration, NULL);
}

static void produce_input()
{
	long txsForGPU = (long)((float)REQUEST_GRANULARITY * (float)REQUEST_GPU);
	long txsForCPU = (long)((float)REQUEST_GRANULARITY * (float)REQUEST_CPU);
	const float sharedAmount = (float)REQUEST_CPU + (float)REQUEST_GPU;
	long txsForSHARED = (long)((float)REQUEST_GRANULARITY * (1.0f - sharedAmount));

	endInputPtr[CPU_QUEUE] += txsForCPU;
	endInputPtr[GPU_QUEUE] += txsForGPU;
	endInputPtr[SHARED_QUEUE] += txsForSHARED;
	// for (int i = 0; i < parsedData.nb_threads; ++i) {
	// 	printf("[%2i] start=%9li end=%9li\n", i, startInputPtr[i], endInputPtr[i]);
	// }
	__sync_synchronize(); // memory fence
	wait_ms(REQUEST_LATENCY);
	if (!HeTM_is_stop()) {
		produce_input();
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
	int sizeCache = memcd->nbSets*memcd->nbWays;

	// 1) hash key
	// modHash = (key>>4) % memcd->nbSets;
	modHash = key % memcd->nbSets;
	// TODO: this would be nice, but we lose control of where the key goes to
	// memset(key, 0, size_of_hash);
	// memcpy(key, input_key, sizeof(int));
	// MurmurHash3_x64_128(key, size_of_hash, 0, hash);
	// modHash = hash[0];
	// modHash += hash[1];

// #if BANK_PART == 1 /* use MOD 3 */
// 	setIdx = (modHash / 3 + (modHash % 3) * (memcd->nbSets / 3)) % memcd->nbSets;
// #else /* use MOD 2 */
// 	setIdx = (modHash / 2 + (modHash % 2) * (memcd->nbSets / 2)) % memcd->nbSets;
// #endif
	setIdx = modHash;

	// 2) setIdx <- hash % nbSets
	setIdx = setIdx * memcd->nbWays;
	int mod_set = setIdx;

	for (int i = 0; i < memcd->nbWays; ++i) {
		size_t newIdx = setIdx + i;
		__builtin_prefetch(&memcd->state[newIdx], 0, 1);
		__builtin_prefetch(&memcd->key[newIdx], 0, 1);
		// __builtin_prefetch(&memcd->extraKey[newIdx], 0, 1);
		// __builtin_prefetch(&memcd->extraKey[newIdx+sizeCache], 0, 1);
		// __builtin_prefetch(&memcd->extraKey[newIdx+2*sizeCache], 0, 1);
		__builtin_prefetch(&memcd->ts_CPU[newIdx], 1, 1);
		__builtin_prefetch(&memcd->val[newIdx], 0, 1);
		// __builtin_prefetch(&memcd->extraVal[newIdx], 0, 1);
		// __builtin_prefetch(&memcd->extraVal[newIdx+sizeCache], 0, 1);
		// __builtin_prefetch(&memcd->extraVal[newIdx+2*sizeCache], 0, 1);
		// __builtin_prefetch(&memcd->extraVal[newIdx+3*sizeCache], 0, 1);
		// __builtin_prefetch(&memcd->extraVal[newIdx+4*sizeCache], 0, 1);
		// __builtin_prefetch(&memcd->extraVal[newIdx+5*sizeCache], 0, 1);
		// __builtin_prefetch(&memcd->extraVal[newIdx+6*sizeCache], 0, 1);
	}

  /* Allow overdrafts */
  TM_START(z, RW);

	// 3) find in set the key, if not found write not found in the output
	for (int i = 0; i < memcd->nbWays; ++i)
	{
		size_t newIdx = setIdx + i;
		int readState = TM_LOAD(&memcd->state[newIdx]);
		int readKey   = TM_LOAD(&memcd->key[newIdx]);
		int readKey1  = TM_LOAD(&memcd->extraKey[newIdx]);
		int readKey2  = TM_LOAD(&memcd->extraKey[newIdx+sizeCache]);
		int readKey3  = TM_LOAD(&memcd->extraKey[newIdx+2*sizeCache]);
		if ((readState & MEMCD_VALID) && readKey == key && readKey1 == key
				&& readKey2 == key && readKey3 == key) {
			// found it!
			int readVal = TM_LOAD(&memcd->val[newIdx]);
			int readVal1 = TM_LOAD(&memcd->extraVal[newIdx]);
			int readVal2 = TM_LOAD(&memcd->extraVal[newIdx+sizeCache]);
			int readVal3 = TM_LOAD(&memcd->extraVal[newIdx+2*sizeCache]);
			int readVal4 = TM_LOAD(&memcd->extraVal[newIdx+3*sizeCache]);
			int readVal5 = TM_LOAD(&memcd->extraVal[newIdx+4*sizeCache]);
			int readVal6 = TM_LOAD(&memcd->extraVal[newIdx+5*sizeCache]);
			int readVal7 = TM_LOAD(&memcd->extraVal[newIdx+6*sizeCache]);
			int* volatile ptr_ts = &memcd->ts_CPU[newIdx];
			int ts = TM_LOAD(ptr_ts);
			TM_STORE(ptr_ts, ts+1);
			// *ptr_ts = input_clock; // Done non-transactionally
			response.isFound = 1;
			response.value   = readVal|readVal1|readVal2|readVal3|readVal4|readVal5|readVal6|readVal7;
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
	int sizeCache = memcd->nbSets*memcd->nbWays;

	// 1) hash key
	// modHash = (key>>4) % memcd->nbSets;
	modHash = key % memcd->nbSets;
	// TODO: this would be nice, but we lose control of where the key goes to
	// memset(key, 0, size_of_hash);
	// memcpy(key, input_key, sizeof(int));
	// MurmurHash3_x64_128(key, size_of_hash, 0, hash);
	// modHash = hash[0];
	// modHash += hash[1];

// #if BANK_PART == 1 /* use MOD 3 */
// 	setIdx = modHash / 3 + (modHash % 3) * (memcd->nbSets / 3);
// #else /* use MOD 2 */
// 	setIdx = modHash / 2 + (modHash % 2) * (memcd->nbSets / 2);
// #endif
	setIdx = modHash;

	// 2) setIdx <- hash % nbSets
	int mod_set = setIdx;
	setIdx = setIdx * memcd->nbWays;

	__builtin_prefetch(&memcd->setUsage[mod_set], 1, 0);
	for (int i = 0; i < memcd->nbWays; ++i) {
		size_t newIdx = setIdx + i;
		__builtin_prefetch(&memcd->state[newIdx], 1, 1);
		__builtin_prefetch(&memcd->key[newIdx], 1, 1);
		// __builtin_prefetch(&memcd->extraKey[newIdx], 1, 1);
		// __builtin_prefetch(&memcd->extraKey[newIdx+sizeCache], 1, 1);
		// __builtin_prefetch(&memcd->extraKey[newIdx+2*sizeCache], 1, 1);
		__builtin_prefetch(&memcd->ts_CPU[newIdx], 1, 1);
		__builtin_prefetch(&memcd->ts_GPU[newIdx], 1, 1);
		__builtin_prefetch(&memcd->val[newIdx], 1, 1);
		// __builtin_prefetch(&memcd->extraVal[newIdx], 1, 1);
		// __builtin_prefetch(&memcd->extraVal[newIdx+sizeCache], 1, 1);
		// __builtin_prefetch(&memcd->extraVal[newIdx+2*sizeCache], 1, 1);
		// __builtin_prefetch(&memcd->extraVal[newIdx+3*sizeCache], 1, 1);
		// __builtin_prefetch(&memcd->extraVal[newIdx+4*sizeCache], 1, 1);
		// __builtin_prefetch(&memcd->extraVal[newIdx+5*sizeCache], 1, 1);
		// __builtin_prefetch(&memcd->extraVal[newIdx+6*sizeCache], 1, 1);
	}

	int usageValue = memcd->setUsage[mod_set];

	// before starting the transaction take a clock value
	unsigned memcd_clock_val = __sync_fetch_and_add(&memcached_global_clock, 1);

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
		int readKey1 = TM_LOAD(&memcd->extraKey[newIdx]);
		int readKey2 = TM_LOAD(&memcd->extraKey[newIdx+sizeCache]);
		int readKey3 = TM_LOAD(&memcd->extraKey[newIdx+2*sizeCache]);
		if (readKey == key && readKey1 == key && readKey2 == key && readKey3 == key
				&& ((readState & MEMCD_VALID) != 0)) {
			// found the key in the cache --> just use this spot
			isInCache = 1;
			idxFound = i;
			break;
		}
		unsigned readTS_CPU = TM_LOAD(&memcd->ts_CPU[newIdx]);
		unsigned readTS_GPU = memcd->ts_GPU[newIdx]; // hack here
		unsigned readTS = std::max(readTS_CPU, readTS_GPU);
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
	int* volatile ptr_ts = &memcd->ts_CPU[newIdx]; // TODO: optimizer screws the ptrs
	int* volatile ptr_val = &memcd->val[newIdx];
	int* volatile ptr_val1 = &memcd->extraVal[newIdx];
	int* volatile ptr_val2 = &memcd->extraVal[newIdx+sizeCache];
	int* volatile ptr_val3 = &memcd->extraVal[newIdx+2*sizeCache];
	int* volatile ptr_val4 = &memcd->extraVal[newIdx+3*sizeCache];
	int* volatile ptr_val5 = &memcd->extraVal[newIdx+4*sizeCache];
	int* volatile ptr_val6 = &memcd->extraVal[newIdx+5*sizeCache];
	int* volatile ptr_val7 = &memcd->extraVal[newIdx+6*sizeCache];

	int* volatile ptr_setUsage = &memcd->setUsage[mod_set];
	// if (usageValue != input_clock) {
		TM_STORE(ptr_setUsage, input_clock);
	// }

	// Done non-transactionally after commit
	// TM_STORE(ptr_ts, input_clock); // *ptr_ts = input_clock;
	*ptr_ts = input_clock;

	TM_STORE(ptr_val, val); // *ptr_val = val;
	TM_STORE(ptr_val1, val);
	TM_STORE(ptr_val2, val);
	TM_STORE(ptr_val3, val);
	TM_STORE(ptr_val4, val);
	TM_STORE(ptr_val5, val);
	TM_STORE(ptr_val6, val);
	TM_STORE(ptr_val7, val);

	if (!isInCache) {
		volatile int newState = MEMCD_VALID|MEMCD_WRITTEN;
		int* volatile ptr_key = &memcd->key[newIdx];
		int* volatile ptr_key1 = &memcd->extraKey[newIdx];
		int* volatile ptr_key2 = &memcd->extraKey[newIdx+sizeCache];
		int* volatile ptr_key3 = &memcd->extraKey[newIdx+2*sizeCache];
		int* volatile ptr_state = &memcd->state[newIdx];

		TM_STORE(ptr_key, key); // *ptr_key = key;
		TM_STORE(ptr_key1, key);
		TM_STORE(ptr_key2, key);
		TM_STORE(ptr_key3, key);

		TM_STORE(ptr_state, newState); // *ptr_state = newState;
	}
	TM_COMMIT;

	// printf("wrote %4i %p (ts %p state %p)\n", modHash, &memcd->setUsage[modHash], &memcd->ts[0], &memcd->state[0]);
}

void cpu_SET_kernel_NOTX(memcd_t *memcd, int *input_key, int *input_value, unsigned input_clock)
{
	size_t modHash, setIdx;
	memcd_get_output_t response;
	int key = *input_key;
	int val = *input_value;
	int extraKey[3] = {key, key, key};
	int sizeCache = memcd->nbSets*memcd->nbWays;

	// 1) hash key
	// modHash = (key>>4) % memcd->nbSets;
	modHash = key % memcd->nbSets;

// #if BANK_PART == 1 /* use MOD 3 */
// 	setIdx = modHash / 3 + (modHash % 3) * (memcd->nbSets / 3);
// #else /* use MOD 2 */
// 	setIdx = modHash / 2 + (modHash % 2) * (memcd->nbSets / 2);
// #endif
	setIdx = modHash;

	// 2) setIdx <- hash % nbSets
	int mod_set = setIdx;
	setIdx = setIdx * memcd->nbWays;

	// 3) find in set the key, if not found write not found in the output
	int idxFound = -1;
	int idxEvict = -1;
	int isInCache = 0;
	int cacheHasSpace = 0;
	unsigned TS   = (unsigned)-1; // largest TS
	for (int i = 0; i < memcd->nbWays; ++i)
	{
		size_t newIdx = setIdx + i;
		int readState = memcd->state[newIdx];
		if (((readState & MEMCD_VALID) == 0) && idxFound == -1) {
			// found empty spot
			cacheHasSpace = 1;
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
		unsigned readTS = memcd->ts_CPU[newIdx];
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
	memcd->ts_CPU[newIdx] = input_clock;
	memcd->val[newIdx] = val;
	memcd->extraVal[newIdx] = val;
	memcd->extraVal[newIdx+sizeCache] = val;
	memcd->extraVal[newIdx+2*sizeCache] = val;
	memcd->extraVal[newIdx+3*sizeCache] = val;
	memcd->extraVal[newIdx+4*sizeCache] = val;
	memcd->extraVal[newIdx+5*sizeCache] = val;
	memcd->extraVal[newIdx+7*sizeCache] = val;
	// if (!cacheHasSpace && !isInCache) {
	// 	printf("evicted %i -> %i\n", memcd->key[newIdx], key);
	// }
	if (!isInCache) {
		int newState = MEMCD_VALID|MEMCD_WRITTEN;
		memcd->key[newIdx] = key;
		memcd->extraKey[newIdx] = key;
		memcd->extraKey[newIdx+sizeCache] = key;
		memcd->extraKey[newIdx+2*sizeCache] = key;
		memcd->val[newIdx] = key;
		memcd->extraVal[newIdx] = key;
		memcd->extraVal[newIdx+sizeCache] = key;
		memcd->extraVal[newIdx+2*sizeCache] = key;
		memcd->extraVal[newIdx+3*sizeCache] = key;
		memcd->extraVal[newIdx+4*sizeCache] = key;
		memcd->extraVal[newIdx+5*sizeCache] = key;
		memcd->extraVal[newIdx+6*sizeCache] = key;
		memcd->extraVal[newIdx+7*sizeCache] = key;
		memcd->state[newIdx] = newState;
	}
}

/* ################################################################### *
* TRANSACTION THREADS
* ################################################################### */

static const int CPU_K_TXS = 20;
thread_local static int myK_TXs = 0;
thread_local static int buffers_start = 0; // choose whether is CPU or SHARED

static void test(int id, void *data)
{
  thread_data_t *d = &((thread_data_t *)data)[id];
  cuda_t       *cd = d->cd;
  memcd_t   *memcd = d->memcd;
	// TODO: this goes into the input
	float setKernelPerc = parsedData.set_percent * 1000;
	static thread_local volatile unsigned seed = 0x213FAB + id;

	int good_buffers_start = 0;
	int bad_buffers_start = size_of_CPU_input_buffer/sizeof(int);

  int input_key;
  int input_val;

  int nbSets = memcd->nbSets;
  int nbWays = memcd->nbWays;
  unsigned rnd;
	int rndOpt = RAND_R_FNC(d->seed);
	static thread_local int curr_tx = 0;

	if (myK_TXs == 0 && !isCPUBatchSteal/*&& startInputPtr[CPU_QUEUE] + CPU_K_TXS <= endInputPtr[CPU_QUEUE]*/) {
		// fetch transactions from the CPU_QUEUE
		volatile long oldStartInputPtr;
		volatile long newStartInputPtr;
		do {
			oldStartInputPtr = startInputPtr[CPU_QUEUE];
			newStartInputPtr = oldStartInputPtr + CPU_K_TXS;
		} while (!__sync_bool_compare_and_swap(&startInputPtr[CPU_QUEUE], oldStartInputPtr, newStartInputPtr));
		myK_TXs = CPU_K_TXS;
		buffers_start = good_buffers_start; // CPU input buffer
	}

	if (myK_TXs == 0 && isCPUBatchSteal/*&& startInputPtr[SHARED_QUEUE] + CPU_K_TXS <= endInputPtr[SHARED_QUEUE]*/) {
		// could not fetch from the CPU, let us see the SHARED_QUEUE
		// NOTE: case BANK_PART == 2 the shared queue is actually the GPU
		volatile long oldStartInputPtr;
		volatile long newStartInputPtr;
		do {
			oldStartInputPtr = startInputPtr[SHARED_QUEUE];
			newStartInputPtr = oldStartInputPtr + CPU_K_TXS;
		} while (!__sync_bool_compare_and_swap(&startInputPtr[SHARED_QUEUE], oldStartInputPtr, newStartInputPtr));
		myK_TXs = CPU_K_TXS;
		buffers_start = bad_buffers_start; // SHARED input buffer
	}

	if (myK_TXs == 0) {
		// failed to get new TXs
		// need to discount this transaction
		HeTM_thread_data[0]->curNbTxs--;
		if (HeTM_get_GPU_status(0) == HETM_BATCH_DONE) {
			HeTM_thread_data[0]->curNbTxsNonBlocking--;
		}
		return; // wait for more input
	}

	// Ok, we have TXs to do!
	myK_TXs--;

	// int buffers_start = isInterBatch ? bad_buffers_start : good_buffers_start;
	input_key = CPUInputBuffer[buffers_start+id*NB_CPU_TXS_PER_THREAD + curr_tx];
	input_val = input_key;

	int isSet = rndOpt % 100000 < setKernelPerc;

#ifdef CPU_STEAL_ONLY_GETS
	isSet = isCPUBatchSteal ? 0 : isSet;
#endif /* CPU_STEAL_ONLY_GETS */

	/* 100% * 1000*/
	if (isSet) {
		// Set kernel
		cpu_SET_kernel(d->memcd, &input_key, &input_val, *d->memcd->globalTs);
	} else {
		// Get kernel
		memcd_get_output_t res; // TODO: write it in the output buffer
		res = cpu_GET_kernel(d->memcd, &input_key, *d->memcd->globalTs);
	}
	curr_tx += 1;
	curr_tx = curr_tx % NB_CPU_TXS_PER_THREAD;

	volatile int spin = 0;
	int randomBackoff = RAND_R_FNC(seed) % parsedData.CPU_backoff;
	while (spin++ < randomBackoff);

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

static int nbGPUStealBatches = 0;
static int nbCPUStealBatches = 0;
static int nbBatches = 0;

static void before_batch(int id, void *data)
{
	// NOTE: now batch is selected based on the SHARED_QUEUE
	// thread_local static unsigned long seed = 0x0012112A3112514A;
	// isInterBatch = RAND_R_FNC(seed) % 100 < parsedData.shared_percent;
	// __sync_synchronize();
	//
	// if (isInterBatch) {
	// 	memman_select("GPU_input_buffer_bad");
	// } else {
	// 	memman_select("GPU_input_buffer_good");
	// }
	// memman_cpy_to_gpu(NULL, NULL);

	thread_local static unsigned long seed = 0x4FA3112E14A;
	isCPUBatchSteal = (RAND_R_FNC(seed) % 1000) < (parsedData.CPU_steal_prob * 1000);
	isGPUBatchSteal = (RAND_R_FNC(seed) % 1000) < (parsedData.GPU_steal_prob * 1000);

	if (nbBatches % 3 == 0) {
		// check the ratio
		float ratioCPU = ((float)nbCPUStealBatches / (float)nbBatches);
		float ratioGPU = ((float)nbGPUStealBatches / (float)nbBatches);
		if (ratioCPU < parsedData.CPU_steal_prob && !isCPUBatchSteal) {
			if (parsedData.CPU_steal_prob > 0) isCPUBatchSteal = 1;
		} else if (ratioCPU > parsedData.CPU_steal_prob && isCPUBatchSteal) {
			if (parsedData.CPU_steal_prob < 1.0) isCPUBatchSteal = 0;
		} else if (ratioGPU < parsedData.GPU_steal_prob && !isGPUBatchSteal) {
			if (parsedData.GPU_steal_prob > 0) isGPUBatchSteal = 1;
		} else if (ratioGPU > parsedData.GPU_steal_prob && isGPUBatchSteal) {
			if (parsedData.GPU_steal_prob > 0) isGPUBatchSteal = 0;
		}
	}

	nbBatches++;
	if (isGPUBatchSteal) nbGPUStealBatches++;
	if (isCPUBatchSteal) nbCPUStealBatches++;
}

static void after_batch(int id, void *data)
{
	for (int j = 0; j < HETM_NB_DEVICES; ++j)
	{
		Config::GetInstance()->SelDev(j);
		GPU_output_buffer.GetMemObj(j)->CpyDtH();
	}
}

static void before_kernel(int id, void *data) { }

static void after_kernel(int id, void *data) { }

static int wasWaitingTXs = 0;

static void choose_policy(int, void*) {
  // -----------------------------------------------------------------------
  // int idGPUThread = HeTM_shared_data.nbCPUThreads;
  // long TXsOnCPU = 0;
	// long TXsOnGPU = 0;
  // for (int i = 0; i < HeTM_shared_data.nbCPUThreads; ++i) {
  //   TXsOnCPU += HeTM_shared_data.threadsInfo[i].curNbTxs;
	// }
	//
	// if (!wasWaitingTXs) {
	// 	TXsOnGPU = parsedData.GPUthreadNum*parsedData.GPUblockNum*parsedData.trans;
	// }

  // TODO: this picks the one with higher number of TXs
	// TODO: GPU only gets the stats in the end --> need to remove the dropped TXs
	// HeTM_stats_data.nbTxsGPU += TXsOnGPU;
  // if (HeTM_gshared_data.policy == HETM_GPU_INV) {
	// 	if (HeTM_is_interconflict()) {
	// 		HeTM_stats_data.nbDroppedTxsGPU += TXsOnGPU;
	// 	} else {
	// 		HeTM_stats_data.nbCommittedTxsGPU += TXsOnGPU;
	// 	}
	// } else if (HeTM_gshared_data.policy == HETM_CPU_INV) {
	// 	HeTM_stats_data.nbCommittedTxsGPU += TXsOnGPU;
	// }

	// can only choose the policy for the next round
  // if (TXsOnCPU > TXsOnGPU) {
  //   HeTM_gshared_data.policy = HETM_GPU_INV;
  // } else {
  //   HeTM_gshared_data.policy = HETM_CPU_INV;
  // }
	// wasWaitingTXs = 0;
	// __sync_synchronize();
  // -----------------------------------------------------------------------
}

static void test_cuda(int id, void *data)
{
  thread_data_t    *d = &((thread_data_t *)data)[id];
  cuda_t          *cd = d->cd;
  account_t *base_ptr = d->memcd->key;
	int  notEnoughInput = 0;
	int          gotTXs = 0;
	int           devId = Config::GetInstance()->SelDev();

	static int counter = 0;

	*(d->memcd->globalTs) += 1;

	// -------------------
	if (!isGPUBatchSteal/*startInputPtr[GPU_QUEUE] + NB_GPU_TXS <= endInputPtr[GPU_QUEUE]*/) {
		// fetch transactions from the CPU_QUEUE
		volatile long oldStartInputPtr;
		volatile long newStartInputPtr;
		do {
			oldStartInputPtr = startInputPtr[GPU_QUEUE];
			newStartInputPtr = oldStartInputPtr + NB_GPU_TXS;
		} while (!__sync_bool_compare_and_swap(&startInputPtr[GPU_QUEUE], oldStartInputPtr, newStartInputPtr));
		gotTXs = 1;
		counter = (counter + 1) % NB_OF_GPU_BUFFERS;

		int *cpuInput = (int*)GPU_input_buffer_good.GetMemObj()->host;
		// TODO: multiGPU
		int *gpuInput = (int*)GPU_input_buffer_good.GetMemObj(0)->dev;
		cpuInput += counter * maxGPUoutputBufferSize;

		CUDA_CPY_TO_DEV_ASYNC(gpuInput, cpuInput, maxGPUoutputBufferSize * sizeof(int), PR_getCurrentStream());
		// memman_cpy_to_gpu(NULL, NULL, *hetm_batchCount);
	}

	if (isGPUBatchSteal /*&& startInputPtr[SHARED_QUEUE] + NB_GPU_TXS <= endInputPtr[SHARED_QUEUE]*/) {
		// could not fetch from the CPU, let us see the SHARED_QUEUE
		// NOTE: case BANK_PART == 2 the shared queue is actually the GPU
		volatile long oldStartInputPtr = startInputPtr[SHARED_QUEUE];
		volatile long newStartInputPtr = oldStartInputPtr + NB_GPU_TXS;
		do {
			oldStartInputPtr = startInputPtr[SHARED_QUEUE];
			newStartInputPtr = oldStartInputPtr + NB_GPU_TXS;
		} while (!__sync_bool_compare_and_swap(&startInputPtr[SHARED_QUEUE], oldStartInputPtr, newStartInputPtr));
		gotTXs = 1;
		counter = (counter + 1) % NB_OF_GPU_BUFFERS;

		int *cpuInput = (int*)GPU_input_buffer_bad.GetMemObj(devId)->host;
		int *gpuInput = (int*)GPU_input_buffer_bad.GetMemObj(devId)->dev;
		cpuInput += counter * maxGPUoutputBufferSize;

		// TODO: after a bad batch all abort (only happens on the GPU steal)
		CUDA_CPY_TO_DEV_ASYNC(gpuInput, cpuInput, maxGPUoutputBufferSize * sizeof(int), PR_getCurrentStream());
		// memman_cpy_to_gpu(NULL, NULL, *hetm_batchCount);
	}

	// if (!gotTXs) {
	// 	// failed to get new TXs
	// 	// need to discount this transaction
	// 	wasWaitingTXs = 1;
	// }
	// -------------------

	__sync_synchronize();

	if (wasWaitingTXs) { // Not used anymore
		// need to wait for more input

		// TODO: this HETM_GPU_IDLE is not working !!!

		// if the GPU is idle for too long, a large chunk of data may need to
		// be sync'ed, actually VERS could handle this issue by sending its
		// logs asynchronously
		// HeTM_set_GPU_status(HETM_GPU_IDLE);

		do {
			COMPILER_FENCE(); // reads HeTM_is_stop() (needed for optimization flags)
		} while ((startInputPtr[GPU_QUEUE] + NB_GPU_TXS > endInputPtr[GPU_QUEUE]
			&& startInputPtr[SHARED_QUEUE] + NB_GPU_TXS > endInputPtr[SHARED_QUEUE])
			&& !HeTM_is_stop()); // wait

		// HeTM_set_GPU_status(HETM_BATCH_RUN);
		// __sync_synchronize();
		jobWithCuda_runEmptyKernel(d, cd, base_ptr, *(d->memcd->globalTs));
	}
	else
	{
		memcd_global_ts.GetMemObj(devId)->CpyHtD(PR_getCurrentStream());
		jobWithCuda_runMemcd(d, cd, base_ptr, *(d->memcd->globalTs));
	}
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
	const int nbGPUs = HETM_NB_DEVICES;

  memset(&parsedData, 0, sizeof(thread_data_t));

  PRINT_FLAGS();

  // ##########################################
  // ### Input management
  bank_parseArgs(argc, argv, &parsedData);

	// ---------------------------------------------------------------------------
	maxGPUoutputBufferSize = parsedData.GPUthreadNum*parsedData.GPUblockNum*sizeof(int);
	currMaxCPUoutputBufferSize = maxGPUoutputBufferSize; // realloc on full

	size_of_GPU_input_buffer = maxGPUoutputBufferSize; // each transaction has 1 input
	// TODO: this parsedData.nb_threads is what comes of the -n input
	size_of_CPU_input_buffer = parsedData.nb_threads*sizeof(int) * (nbGPUs+1) * NB_CPU_TXS_PER_THREAD;

	for (int j = 0; j < nbGPUs; ++j)
	{
		MemObjBuilder b;
		MemObj *m;
		Config::GetInstance()->SelDev(j);
		GPU_output_buffer.AddMemObj(m = new MemObj(b
			.SetSize(maxGPUoutputBufferSize)
			->SetOptions(MEMMAN_NONE)
			->AllocDevPtr()
			->AllocHostPtr(),
			j
		));
		GPUoutputBuffer[j] = (int*)m->dev;
		// printf("<<<<<<<<<<<< dev %i GPUoutputBuffer = %p\n", j, GPUoutputBuffer[j]);
	}

	for (int j = 0; j < nbGPUs; ++j)
	{
		MemObjBuilder b_input_buffer;
		MemObjBuilder b_input_buffer_good;
		MemObjBuilder b_input_buffer_bad;
		MemObj *m_input_buffer;
		MemObj *m_input_buffer_good;
		MemObj *m_input_buffer_bad;

		Config::GetInstance()->SelDev(j);

		GPU_input_buffer.AddMemObj(m_input_buffer = new MemObj(b_input_buffer
			.SetOptions(MEMMAN_NONE)
			->SetSize(size_of_GPU_input_buffer)
			->AllocDevPtr(),
			j
		));
		GPUInputBuffer[j] = (int*)m_input_buffer->dev;

		GPU_input_buffer_good.AddMemObj(m_input_buffer_good = new MemObj(b_input_buffer_good
			.SetOptions(MEMMAN_NONE)
			->SetSize(size_of_GPU_input_buffer*NB_OF_GPU_BUFFERS)
			->SetDevPtr(m_input_buffer->dev)
			->AllocHostPtr(),
			j
		));
		GPU_input_buffer_bad.AddMemObj(m_input_buffer_bad = new MemObj(b_input_buffer_bad
			.SetOptions(MEMMAN_NONE)
			->SetSize(size_of_GPU_input_buffer*NB_OF_GPU_BUFFERS)
			->SetDevPtr(m_input_buffer->dev)
			->AllocHostPtr(),
			j
		));
	}

	// CPU Buffers
	malloc_or_die(CPUInputBuffer, size_of_CPU_input_buffer * 2); // good and bad
	malloc_or_die(CPUoutputBuffer, currMaxCPUoutputBufferSize); // kinda big

	fill_GPU_input_buffers();
	fill_CPU_input_buffers();
	// ---------------------------------------------------------------------------

	size_t nbSets = parsedData.nb_accounts;
	size_t nbWays = parsedData.num_ways;
	accountsSize = nbSets*nbWays*sizeof(account_t);
	// last one is to check if the set was changed or not
	sizePool = accountsSize * 5 + nbSets*sizeof(account_t);

	// Setting the key size to be 16
	sizePool += accountsSize * 3; // already have 4B missing 3*4B

	// Setting the value size to be 32
	sizePool += accountsSize * 7; // already have 4B missing 7*4B

  HeTM_init((HeTM_init_s){
// #if CPU_INV == 1
    // .policy       = HETM_CPU_INV,
// #else /* GPU_INV */
    .policy       = HETM_GPU_INV,
// #endif /**/
    .nbCPUThreads = parsedData.nb_threads,
    .nbGPUBlocks  = parsedData.GPUblockNum,
    .nbGPUThreads = parsedData.GPUthreadNum,
		.timeBudget   = parsedData.timeBudget,
#if HETM_CPU_EN == 0
    .isCPUEnabled = 0,
    .isGPUEnabled = 1,
#elif HETM_GPU_EN == 0
    .isCPUEnabled = 1,
    .isGPUEnabled = 0,
#else /* both on */
    .isCPUEnabled = 1,
    .isGPUEnabled = 1,
#endif
		.mempool_size = (long)sizePool,
		.mempool_opts = MEMMAN_NONE
  });

	MemObjBuilder b_memcd_global_ts;
	MemObjBuilder b_CPU_memcd_global_ts;
	MemObj *m_memcd_global_ts;
	MemObj *m_CPU_memcd_global_ts = new MemObj(b_CPU_memcd_global_ts
				.SetOptions(MEMMAN_NONE)
				->SetSize(sizeof(unsigned))
				->AllocHostPtr(), 0);
	memcd_global_ts.AddMemObj(m_memcd_global_ts = new MemObj(b_memcd_global_ts
			.SetOptions(MEMMAN_NONE)
			->SetSize(sizeof(unsigned))
			->AllocDevPtr()
			->SetHostPtr(m_CPU_memcd_global_ts->host),
			/* for CPU only to work */0
		));
	for (int j = 1; j < nbGPUs; ++j)
	{
		memcd_global_ts.AddMemObj(m_memcd_global_ts = new MemObj(b_memcd_global_ts
				.SetOptions(MEMMAN_NONE)
				->SetSize(sizeof(unsigned))
				->AllocDevPtr()
				->SetHostPtr(m_CPU_memcd_global_ts->host),
				j
			));
	}

	malloc_or_die(memcd, nbGPUs);
	for (int j = 0; j < nbGPUs; ++j)
	{
		memcd[j].nbSets = nbSets;
		memcd[j].nbWays = nbWays;
		memcd[j].globalTs = (unsigned*)memcd_global_ts.GetMemObj(j)->host; // TODO: multiGPU
	}

  // TODO:
  parsedData.nb_threadsCPU = HeTM_gshared_data.nbCPUThreads;
  parsedData.nb_threads    = HeTM_gshared_data.nbThreads;

	// input manager will handle these
	malloc_or_die(startInputPtr, 3);
	malloc_or_die(endInputPtr, 3);

	// each device starts with some input
	startInputPtr[CPU_QUEUE]    = 0;
	startInputPtr[GPU_QUEUE]    = 0;
	startInputPtr[SHARED_QUEUE] = 0;
	endInputPtr[CPU_QUEUE]      = 0; // NB_CPU_TXS_PER_THREAD / 2;
	endInputPtr[GPU_QUEUE]      = 0; // NB_GPU_TXS;
	endInputPtr[SHARED_QUEUE]   = 0; // NB_GPU_TXS;

  bank_check_params(&parsedData);

  malloc_or_die(data, parsedData.nb_threads + 1);
  memset(data, 0, (parsedData.nb_threads + 1)*sizeof(thread_data_t)); // safer
  parsedData.dthreads     = data;
  // ##########################################

  jobWithCuda_exit(NULL); // Reset Cuda Device

  malloc_or_die(threads, parsedData.nb_threads);
  
	DEBUG_PRINT("Initializing GPU.\n");

	// mallocs 4 arrays (accountsSize * NUMBER_WAYS * 4)
	for (int j = 0; j < nbGPUs; ++j)
	{
		HeTM_alloc(j, (void**)&memcd[j].key, &gpuMempool, sizePool); // <K,V,TS,STATE>
		memcd[j].extraKey = memcd[j].key + (memcd[j].nbSets*memcd[j].nbWays);
		memcd[j].val      = memcd[j].extraKey + 3*(memcd[j].nbSets*memcd[j].nbWays);
		memcd[j].extraVal = memcd[j].val + (memcd[j].nbSets*memcd[j].nbWays);
		memcd[j].ts_CPU   = memcd[j].extraVal + 7*(memcd[j].nbSets*memcd[j].nbWays);
		memcd[j].ts_GPU   = memcd[j].ts_CPU + (memcd[j].nbSets*memcd[j].nbWays);
		memcd[j].state    = memcd[j].ts_GPU + (memcd[j].nbSets*memcd[j].nbWays);
		memcd[j].setUsage = memcd[j].state + (memcd[j].nbSets*memcd[j].nbWays);
		memset(memcd[j].key, 0, sizePool);
	}
	cuda_t *cuda_st;
	cuda_st = jobWithCuda_init(memcd[j].key, parsedData.nb_threadsCPU,
		sizePool, parsedData.trans, 0, parsedData.GPUthreadNum, parsedData.GPUblockNum,
		parsedData.hprob, parsedData.hmult);
	for (int j = 0; j < nbGPUs; ++j)
	{
		jobWithCuda_initMemcd(cuda_st+j, parsedData.num_ways, parsedData.nb_accounts,
			parsedData.set_percent, parsedData.shared_percent);
		cuda_st->memcd_array_size = accountsSize;
		cuda_st->memcd_nbSets = memcd[j].nbSets;
		cuda_st->memcd_nbWays = memcd[j].nbWays;
  	parsedData.cd = cuda_st;
		//DEBUG_PRINT("Base: %lu %lu \n", bank->accounts, &bank->accounts);
		if (cuda_st == NULL) {
			printf("CUDA init failed.\n");
			exit(-1);
		}
	}
  parsedData.memcd = memcd;

  /* Init STM */
  printf("Initializing STM\n");

	/* POPULATE the cache */
	int *gpu_buffer_cpu_ptr = (int*)GPU_input_buffer_good.GetMemObj()->host;
	for (int i = 0; i < size_of_GPU_input_buffer/sizeof(int); ++i) {
		cpu_SET_kernel_NOTX(memcd, &gpu_buffer_cpu_ptr[i], &gpu_buffer_cpu_ptr[i], 0);
	}
	gpu_buffer_cpu_ptr = (int*)GPU_input_buffer_bad.GetMemObj()->host;
	for (int i = 0; i < size_of_GPU_input_buffer/sizeof(int); ++i) {
		cpu_SET_kernel_NOTX(memcd, &gpu_buffer_cpu_ptr[i], &gpu_buffer_cpu_ptr[i], 0);
	}

	// for (int i = 0; i < 32*4; ++i) {
	// 	printf("%i : KEY=%i STATE=%i\n", i, memcd->key[i], memcd->state[i]);
	// }

	for (int j = 0; j < nbGPUs; ++j)
	{
		CUDA_CPY_TO_DEV(gpuMempool, memcd[j].key, sizePool);
	}
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
		HeTM_after_kernel(after_kernel);
		HeTM_before_kernel(before_kernel);

		HeTM_choose_policy(choose_policy);

    printf("STARTING...(Run %d)\n", j);

		// std::thread inputThread(produce_input); // NO LONGER USED

    TIMER_READ(parsedData.start);

    if (parsedData.duration > 0) {
      nanosleep(&parsedData.timeout, NULL);
    } else {
      sigemptyset(&block_set);
      sigsuspend(&block_set);
    }
		HeTM_set_is_stop(1);
		__sync_synchronize();

    TIMER_READ(parsedData.end);
    printf("STOPPING...\n");

    /* Wait for thread completion */
    HeTM_join_CPU_threads();

		// inputThread.join();

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
  free(memcd);

  free(threads);
  free(data);

	printf("CPU_start=%9li CPU_end=%9li\n", startInputPtr[CPU_QUEUE], endInputPtr[CPU_QUEUE]);
	printf("GPU_start=%9li GPU_end=%9li\n", startInputPtr[GPU_QUEUE], endInputPtr[GPU_QUEUE]);
	printf("SHARED_start=%9li SHARED_end=%9li\n", startInputPtr[SHARED_QUEUE], endInputPtr[SHARED_QUEUE]);
	printf("nbOfGPUSetKernels=%i\n", nbOfGPUSetKernels);

	printf("nbGPUStealBatches=%9i nbCPUStealBatches=%9i\n", nbGPUStealBatches, nbCPUStealBatches);
	printf("timeDtD = %f\n", HeTM_stats_data.timeDtD);
	HeTM_destroy();

  return EXIT_SUCCESS;
}
