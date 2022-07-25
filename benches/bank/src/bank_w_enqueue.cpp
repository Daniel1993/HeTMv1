#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define __USE_GNU

#include "bank.hpp"
#include <cmath>
#include "hetm-cmp-kernels.cuh"
#include "setupKernels.cuh"
#include "bank_aux.h"
#include "CheckAllFlags.h"
#include "input_handler.h"
#include "rdtsc.h"

#include "tbb/concurrent_queue.h"

typedef struct enqueue_req_ {
	int isReadIntensive;
	int isReadTX;
	volatile unsigned accounts_vec[128];
	int isInterConflict;
	int nbAccesses;
	int nbAccounts;
} enqueue_req_t;

// TODO: test with multiple disjoint queues
tbb::concurrent_queue<enqueue_req_t*> sharedConcQueue;

/* ################################################################### *
* GLOBALS
* ################################################################### */

// global

// TODO: memory pool for requests
static int flagEnqueuerToExit = 0;
static enqueue_req_t *requests; // memory pool for the requests
static const int MAX_REQUESTS = 0xFFFF;
static int request_index = 0;

thread_data_t parsedData;
int isInterBatch = 0;

// -----------------------------------------------------------------------------
const size_t NB_OF_BUFFERS = 512; // 2 good + 2 bad

const static int NB_CPU_TXS_PER_THREAD = 16384;

int *GPUoutputBuffer, *CPUoutputBuffer;
int *GPUInputBuffer, *CPUInputBuffer;

static unsigned long long input_seed = 0x3F12514A3F12514A;
int bank_cpu_sample_data = 0;

// CPU output --> must fill the buffer until the log ends (just realloc on full)
static size_t currMaxCPUoutputBufferSize, currCPUoutputBufferPtr = 0;
static size_t maxGPUoutputBufferSize;
static size_t size_of_GPU_input_buffer, size_of_CPU_input_buffer;
static int lockOutputBuffer = 0;

static const int PAGE_SIZE = 4096; // TODO: this is also defined in hetm proj (CACHE_GRANULE_SIZE)

FILE *GPU_input_file = NULL;
FILE *CPU_input_file = NULL;
// -----------------------------------------------------------------------------

#define INTEREST_RATE 0.5

#define COMPUTE_TRANSFER(val) \
	val // TODO: do math that does not kill the final result

/**
 * --- WORKLOADS ---
 * BANK_PART == 1 --> split the dataset with GPU_PART/CPU_PART
 * BANK_PART == 2 --> Uniform at random inter-leaved
 * BANK_PART == 3 --> Zipf interleaved
 * BANK_PART == 4 --> Zipf separate (else)
 */

static int fill_GPU_input_buffers()
{
	int buffer_last = size_of_GPU_input_buffer/sizeof(int) * NB_OF_BUFFERS;

	memman_select("GPU_input_buffer_good");
	int *cpu_ptr = (int*)memman_get_cpu(NULL);

	int nbPages = parsedData.nb_accounts / PAGE_SIZE;
	// if (parsedData.nb_accounts % PAGE_SIZE) nbPages++;

#if BANK_PART == 1 /* Uniform at random: split */
	unsigned rnd = RAND_R_FNC(input_seed);
	for (int i = 0; i < buffer_last; ++i) {
		cpu_ptr[i] = GPU_ACCESS(rnd, parsedData.nb_accounts-20);
		rnd = RAND_R_FNC(input_seed);
	}

	memman_select("GPU_input_buffer_bad");
	cpu_ptr = (int*)memman_get_cpu(NULL);

	rnd = RAND_R_FNC(input_seed);
	for (int i = 0; i < buffer_last; ++i) {
		if (i % 512 == 0) {
			cpu_ptr[i] = 0; // deterministic intersection
			continue;
		}
		cpu_ptr[i] = INTERSECT_ACCESS_GPU(rnd, parsedData.nb_accounts-20);
		rnd = RAND_R_FNC(input_seed);
	}
#elif BANK_PART == 2 /* Uniform at random: interleaved CPU accesses incremental pages */
	unsigned rnd = RAND_R_FNC(input_seed);
	for (int i = 0; i < buffer_last; ++i) {
		unsigned j = rnd % parsedData.nb_accounts;
		cpu_ptr[i] = j;
		cpu_ptr[i] &= (unsigned)-2; // get even accounts
		rnd = RAND_R_FNC(input_seed);
	}

	memman_select("GPU_input_buffer_bad");
	cpu_ptr = (int*)memman_get_cpu(NULL);

	cpu_ptr[0] = 0; // deterministic intersection
	for (int i = 1; i < buffer_last; ++i) {
		unsigned rnd = RAND_R_FNC(input_seed);
		unsigned j = rnd % parsedData.nb_accounts;
		cpu_ptr[i] = j;
		cpu_ptr[i] &= (unsigned)-2; // get even accounts
	}
#elif BANK_PART == 3 /* Zipf: from file */

	GPU_input_file = fopen(parsedData.GPUInputFile, "r");
	printf("Opening %s\n", parsedData.GPUInputFile);

	unsigned rnd; // RAND_R_FNC(input_seed);
	for (int i = 0; i < buffer_last; ++i) {
		// int access = GPU_ACCESS(rnd, (parsedData.nb_accounts-BANK_NB_TRANSFERS-1));
		if (!fscanf(GPU_input_file, "%i\n", &rnd)) {
			printf("Error reading from file\n");
		}
		cpu_ptr[i] = rnd % parsedData.nb_accounts;
		cpu_ptr[i] &= (unsigned)-2;
	}

	memman_select("GPU_input_buffer_bad");
	cpu_ptr = (int*)memman_get_cpu(NULL);

	// cpu_ptr[0] = 0; // deterministic abort
	for (int i = 0; i < buffer_last; ++i) {
		if (!fscanf(GPU_input_file, "%i\n", &rnd)) {
			printf("Error reading from file\n");
		}
		cpu_ptr[i] = rnd % parsedData.nb_accounts;
	}

#elif BANK_PART == 4 /* Zipf */

	unsigned maxGen = parsedData.nb_accounts;
	zipf_setup(maxGen, 0.8);
	unsigned rnd;

	for (int i = 0; i < buffer_last; ++i) {
		rnd = zipf_gen();
		cpu_ptr[i] = rnd;
	}

	memman_select("GPU_input_buffer_bad");
	cpu_ptr = (int*)memman_get_cpu(NULL);

	for (int i = 0; i < buffer_last; ++i) {
		rnd = zipf_gen();
		cpu_ptr[i] = rnd;
	}

#elif BANK_PART == 5 || BANK_PART == 9 || BANK_PART == 10
	// TODO: I should use more input buffers: some caching effects may show up
	unsigned rnd = RAND_R_FNC(input_seed);
	for (int i = 0; i < buffer_last; ++i) {
		unsigned cnfl_rnd = RAND_R_FNC(input_seed);
		unsigned pos = RAND_R_FNC(input_seed);
		if (cnfl_rnd % 100 >= BANK_INTRA_CONFL*100) {
			// no conflict
			cpu_ptr[i] = GPU_ACCESS(pos, parsedData.nb_accounts-20);
		} else {
			cpu_ptr[i] = GPU_ACCESS(pos % 128, parsedData.nb_accounts-20);
		}
	}

	memman_select("GPU_input_buffer_bad");
	cpu_ptr = (int*)memman_get_cpu(NULL);

	rnd = RAND_R_FNC(input_seed);
	for (int i = 0; i < buffer_last; ++i) {
		if (i % 512 == 0) {
			cpu_ptr[i] = 0; // deterministic intersection
			continue;
		}
		unsigned cnfl_rnd = RAND_R_FNC(input_seed);
		unsigned pos = RAND_R_FNC(input_seed);
		if (cnfl_rnd % 100 >= BANK_INTRA_CONFL*100) {
			// no conflict
			cpu_ptr[i] = GPU_ACCESS(pos, parsedData.nb_accounts-20);
		} else {
			cpu_ptr[i] = GPU_ACCESS(pos % 128, parsedData.nb_accounts-20);
		}
	}
#elif BANK_PART == 6 /* BANK_PART == 6 contiguous */
	unsigned rnd = 0;
	for (int i = 0; i < buffer_last; ++i) {
		cpu_ptr[i] = GPU_ACCESS(rnd, parsedData.nb_accounts-20);
		rnd += (parsedData.read_intensive_size + 1) % 8192;
	}

	memman_select("GPU_input_buffer_bad");
	cpu_ptr = (int*)memman_get_cpu(NULL);

	rnd = 0;
	for (int i = 0; i < buffer_last; ++i) {
		if (i % 128 == 0) {
			cpu_ptr[i] = 0; // deterministic intersection
			continue;
		}
		cpu_ptr[i] = INTERSECT_ACCESS_GPU(rnd, parsedData.nb_accounts-20);
		rnd += (parsedData.read_intensive_size + 1) % 8192;
	}
#elif BANK_PART == 7 || BANK_PART == 8 /* not used on CPU, GPU blocks */
	unsigned rnd = RAND_R_FNC(input_seed);
	for (int i = 0; i < buffer_last; ++i) {
		cpu_ptr[i] = GPU_ACCESS(rnd, parsedData.nb_accounts-20);
		rnd = RAND_R_FNC(input_seed);
	}

	memman_select("GPU_input_buffer_bad");
	cpu_ptr = (int*)memman_get_cpu(NULL);

	rnd = RAND_R_FNC(input_seed);
	for (int i = 0; i < buffer_last; ++i) {
		if (i % 128 == 0) {
			cpu_ptr[i] = 0; // deterministic intersection
			continue;
		}
		cpu_ptr[i] = INTERSECT_ACCESS_GPU(rnd, parsedData.nb_accounts-20);
		rnd = RAND_R_FNC(input_seed);
	}
#endif /* BANK_PART == 1 */
	return 0;
}

static int fill_CPU_input_buffers()
{
	int good_buffers_last = size_of_CPU_input_buffer/sizeof(int) * NB_OF_BUFFERS;
	int bad_buffers_last = 2*size_of_CPU_input_buffer/sizeof(int) * NB_OF_BUFFERS;

	// TODO: make the CPU access the end --> target_page * PAGE_SIZE - access

	int nbPages = parsedData.nb_accounts / PAGE_SIZE;
	// if (parsedData.nb_accounts % PAGE_SIZE) nbPages++;
	// TODO: if the last page is incomplete it is never accessed

#if BANK_PART == 1
	unsigned rnd = RAND_R_FNC(input_seed);
	for (int i = 0; i < good_buffers_last; ++i) {
		CPUInputBuffer[i] = CPU_ACCESS(rnd, parsedData.nb_accounts-20);
		rnd = RAND_R_FNC(input_seed);
	}

	rnd = RAND_R_FNC(input_seed);
	for (int i = good_buffers_last; i < bad_buffers_last; ++i) {
		if (i % 128 == 0) {
			CPUInputBuffer[i] = 0; // deterministic intersection
			continue;
		}
		CPUInputBuffer[i] = INTERSECT_ACCESS_CPU(rnd, parsedData.nb_accounts-20);
		rnd = RAND_R_FNC(input_seed);
	}
#elif BANK_PART == 2 /* CPU write once per page */
	for (int i = 0; i < good_buffers_last; ++i) {
		unsigned rnd = RAND_R_FNC(input_seed);
		unsigned j = rnd % (parsedData.nb_accounts - 40);
		CPUInputBuffer[i] = j;
		CPUInputBuffer[i] |= 1; // get odd accounts
	}

	CPUInputBuffer[good_buffers_last] = 0;
	for (int i = good_buffers_last + 1; i < bad_buffers_last; ++i) {
		unsigned rnd = RAND_R_FNC(input_seed);
		unsigned j = rnd % (parsedData.nb_accounts - 20);
		CPUInputBuffer[i] = j;
	}
#elif BANK_PART == 3

	CPU_input_file = fopen(parsedData.CPUInputFile, "r");
	printf("Opening %s\n", parsedData.CPUInputFile);

// #if BANK_PART == 3
	for (int i = 0; i < good_buffers_last; ++i) {
		// unsigned rnd = (*zipf_dist)(generator); // RAND_R_FNC(input_seed);
		unsigned rnd;
		if (fscanf(CPU_input_file, "%i\n", &rnd) == EOF) {
			printf("Error reading from file\n");
		}
		CPUInputBuffer[i] = rnd % (parsedData.nb_accounts - 20);
		CPUInputBuffer[i] |= 1;
	}
	// CPUInputBuffer[good_buffers_last] = 0; // deterministic abort
	for (int i = good_buffers_last; i < bad_buffers_last; ++i) {
		// unsigned rnd = (*zipf_dist)(generator); // RAND_R_FNC(input_seed);
		unsigned rnd; // RAND_R_FNC(input_seed);
		if (fscanf(CPU_input_file, "%i\n", &rnd) == EOF) {
			printf("Error reading from file\n");
		}
		CPUInputBuffer[i] = rnd % (parsedData.nb_accounts - 20);
		// CPUInputBuffer[i] |= 1;
	}
#elif BANK_PART == 4

	unsigned maxGen = parsedData.nb_accounts;
	zipf_setup(maxGen, 0.8);
	unsigned rnd;

	for (int i = 0; i < good_buffers_last; ++i) {
		rnd = zipf_gen();
		CPUInputBuffer[i] = parsedData.nb_accounts - rnd;
	}

	memman_select("GPU_input_buffer_bad");
	cpu_ptr = (int*)memman_get_cpu(NULL);

	for (int i = good_buffers_last + 1; i < bad_buffers_last; ++i) {
		rnd = zipf_gen();
		CPUInputBuffer[i] = parsedData.nb_accounts - rnd;
	}

#elif BANK_PART == 5 || BANK_PART == 9 || BANK_PART == 10
	unsigned reset_rnd = RAND_R_FNC(input_seed);
	unsigned rnd = reset_rnd;
	for (int i = 0; i < good_buffers_last; ++i) {
		unsigned cnfl_rnd = RAND_R_FNC(input_seed);
		unsigned pos = RAND_R_FNC(input_seed);
		if (cnfl_rnd % 100 >= BANK_INTRA_CONFL*100) {
			// no conflict
			CPUInputBuffer[i] = CPU_ACCESS(pos, parsedData.nb_accounts-20);
		} else {
			CPUInputBuffer[i] = CPU_ACCESS(pos % 128, parsedData.nb_accounts-20);
		}
	}

	reset_rnd = RAND_R_FNC(input_seed);
	rnd = reset_rnd;
	for (int i = good_buffers_last; i < bad_buffers_last; ++i) {
		if (i % 128 == 0) {
			CPUInputBuffer[i] = 0; // deterministic intersection
			continue;
		}
		unsigned cnfl_rnd = RAND_R_FNC(input_seed);
		unsigned pos = RAND_R_FNC(input_seed);
		if (cnfl_rnd % 100 >= BANK_INTRA_CONFL*100) {
			// no conflict
			CPUInputBuffer[i] = CPU_ACCESS(pos, parsedData.nb_accounts-20);
		} else {
			CPUInputBuffer[i] = CPU_ACCESS(pos % 128, parsedData.nb_accounts-20);
		}
	}
#elif BANK_PART == 6 /* BANK_PART == 6 contiguous */
	unsigned rnd = 0;
	for (int i = 0; i < good_buffers_last; ++i) {
		CPUInputBuffer[i] = CPU_ACCESS(rnd, parsedData.nb_accounts-20);
		// printf("[%i]: %i (rnd=%u, size=%u)\n", i, CPUInputBuffer[i], rnd, parsedData.nb_accounts-20);
		rnd = (rnd + parsedData.read_intensive_size + 5) % 8192;
	}

	rnd = 0;
	for (int i = good_buffers_last; i < bad_buffers_last; ++i) {
		if (i % 64 == 0) {
			CPUInputBuffer[i] = 0; // deterministic intersection
			continue;
		}
		CPUInputBuffer[i] = INTERSECT_ACCESS_CPU(rnd, parsedData.nb_accounts-20);
		rnd = (rnd + parsedData.read_intensive_size + 5) % 8192;
	}
#elif BANK_PART == 7 || BANK_PART == 8 /* random on GPU contiguous on CPU */
	unsigned rnd = 0;
	for (int i = 0; i < good_buffers_last; ++i) {
		CPUInputBuffer[i] = CPU_ACCESS(rnd, parsedData.nb_accounts-20);
		rnd += parsedData.read_intensive_size + 16;
	}

	rnd = 0;
	for (int i = good_buffers_last; i < bad_buffers_last; ++i) {
		if (i % 128 == 0) {
			CPUInputBuffer[i] = 0; // deterministic intersection
			continue;
		}
		CPUInputBuffer[i] = INTERSECT_ACCESS_CPU(i, parsedData.nb_accounts-20);
		rnd += parsedData.read_intensive_size + 16;
	}
#endif /* BANK_PART */
	return 0;
}

static inline int total(bank_t *bank, int transactional)
{
  long i;
  long total;

  if (!transactional) {
    total = 0;
    for (i = 0; i < bank->size; i++) {
      total += bank->accounts[i];
    }
  } else {
    TM_START(1, RO);
    total = 0;
    for (i = 0; i < bank->size; i++) {
      total += TM_LOAD(&bank->accounts[i]);
    }
    TM_COMMIT;
  }

  return total;
}

static int transfer(account_t *accounts, volatile unsigned *positions, int count, int amount)
{
  volatile int i;
	uintptr_t load1, load2;
  int z = 0;
  int n;
	void *src;
	void *dst;
	void *pos[count];
	void *pos_write;

#if BANK_PART == 7
	pos_write = &accounts[HeTM_thread_data[0]->id * 16];
	asm volatile("" ::: "memory");
#elif BANK_PART == 8
	static thread_local unsigned local_seed = (0x1234 << 10) + HeTM_thread_data[0]->id;
	unsigned rnd = RAND_R_FNC(local_seed);
	unsigned partionSize = parsedData.nb_accounts / parsedData.nb_threadsCPU;
	unsigned rnd_pos = (rnd % partionSize) * HeTM_thread_data[0]->id;
	rnd_pos = rnd_pos / parsedData.access_controller;
	pos_write = &accounts[rnd_pos];

#elif BANK_PART == 6
	void *pos_write2;
	for (n = 0; n < count; n += 2) {
		pos[n] = &accounts[positions[n]];
		pos[n+1] = &accounts[positions[n+1]];
		__builtin_prefetch(pos[n], 0, 0);
		__builtin_prefetch(pos[n+1], 0, 0);
		// printf("prefetch %i and %i\n", positions[n], positions[n+1]);
	}

	int halfAccounts = parsedData.nb_accounts / 2;
	int writeAccountIdx = (positions[0] - halfAccounts) / parsedData.access_controller + halfAccounts;
	int writeAccountIdx2 = (positions[1] - halfAccounts) / parsedData.access_controller + halfAccounts;
	pos_write = &accounts[writeAccountIdx];
	pos_write2 = &accounts[writeAccountIdx2];
#else
	for (n = 0; n < count; n += 2) {
		pos[n] = &accounts[positions[n]];
		pos[n+1] = &accounts[positions[n+1]];
		__builtin_prefetch(pos[n], 0, 0);
		__builtin_prefetch(pos[n+1], 0, 0);
		// printf("prefetch %i and %i\n", positions[n], positions[n+1]);
	}
	// int writeAccountIdx = src;
	int halfAccounts = parsedData.nb_accounts / 2;
	int writeAccountIdx = (positions[0] - halfAccounts) / parsedData.access_controller + halfAccounts;
	pos_write = &accounts[writeAccountIdx];
#endif /* BANK_PART == 7 */

  /* Allow overdrafts */
  TM_START(z, RW);
#if BANK_PART == 7 || BANK_PART == 8
	TM_STORE((int*)pos_write, 1234);
	// TM_STORE((int*)pos_write + 1, 1234);
	// TM_STORE((int*)pos_write + 2, 1234);
	// TM_STORE((int*)pos_write + 3, 1234);
	// TM_STORE((int*)pos_write + 4, 1234);
	// TM_STORE((int*)pos_write + 5, 1234);
	// TM_STORE((int*)pos_write + 6, 1234);
	// TM_STORE((int*)pos_write + 7, 1234);
	// TM_STORE((int*)pos_write + 8, 1234);
	// TM_STORE((int*)pos_write + 9, 1234);
	// TM_STORE((int*)pos_write + 10, 1234);
	// TM_STORE((int*)pos_write + 11, 1234);
	// TM_STORE((int*)pos_write + 12, 1234);
	// TM_STORE((int*)pos_write + 13, 1234);
	// TM_STORE((int*)pos_write + 14, 1234);
	// TM_STORE((int*)pos_write + 15, 1234);
#else
  for (n = 0; n < count; n += 2) {
    src = pos[n];
    dst = pos[n+1];

		// Problem: TinySTM works with the granularity of 8B, PR-STM works with 4B
    load1 = TM_LOAD(src);
    load1 -= COMPUTE_TRANSFER(amount);

    load2 = TM_LOAD(dst);
    load2 += COMPUTE_TRANSFER(amount);
  }

	// TODO: store must be controlled with parsedData.access_controller
	// -----------------

	TM_STORE(pos_write, load1); // TODO: now is 2 reads 1 write
#if BANK_PART == 6
	TM_STORE(pos_write2, load2);
#endif

#endif /* BANK_PART != 7 */
  TM_COMMIT;

  // TODO: remove this
//   volatile int j = 0;
// loop:
//   j++;
//   if (j < 100) goto loop;
  return amount;
}

static int transferReadOnly(account_t *accounts, volatile unsigned *positions, int count, int amount)
{
  volatile int i;
	uintptr_t load1, load2;
  int z = 0;
  int n, src, dst;

	for (n = 0; n < count; ++n) {
		__builtin_prefetch(&accounts[positions[n]], 0, 0);
	}

  /* Allow overdrafts */
  TM_START(z, RW);

  for (n = 0; n < count; n += 2) {
    src = positions[n];
    dst = positions[n+1];

    load1 = TM_LOAD(&accounts[src]);
    load1 -= COMPUTE_TRANSFER(amount);

    load2 = TM_LOAD(&accounts[dst]);
    load2 += COMPUTE_TRANSFER(amount);

		// TM_STORE(&accounts[src], load1);
    // TM_STORE(&accounts[dst], load2);
  }
  TM_COMMIT;
  return load1 + load2;
}

static int readIntensive(account_t *accounts, volatile unsigned *positions, int count, int amount)
{
  int i;
  int z;
  int n;
	int loads[count];
	float res = 0;
	int resI = 0;

	__builtin_prefetch(&accounts[positions[0]], 1, 2);
	for (n = 1; n < count; ++n) {
		__builtin_prefetch(&accounts[positions[n]], 0, 0);
	}

  /* Allow overdrafts */
	z = 0;
  TM_START(z, RW);

  for (n = 0; n < count; ++n) {
    loads[n] = TM_LOAD(&accounts[positions[n]]);
		res += loads[n] * INTEREST_RATE;
  }
	res += amount;
	resI = (int)res;

	TM_STORE(&accounts[positions[0]], resI);
	accounts[positions[0]] = resI;

  TM_COMMIT;

// #if BANK_PART == 3
//   // TODO: BANK_PART 3 benifits VERS somehow
  // volatile int j = 0;
	// loop:
	//   j++;
	//   if (j < 150) goto loop;
// #endif /* BANK_PART == 3 */
  return amount;
}

static int readOnly(account_t *accounts, volatile unsigned *positions, int count, int amount)
{
  int i;
  int z;
  int n;
	int loads[count];
	float res = 0;
	int resI = 0;

	for (n = 0; n < count; ++n) {
		__builtin_prefetch(&accounts[positions[n]], 0, 0);
	}

	z = 0;
  TM_START(z, RW);

  for (n = 0; n < count; ++n) {
    loads[n] = TM_LOAD(&accounts[positions[n]]);
		res += loads[n] * INTEREST_RATE;
		// res = __builtin_cos(res);
  }
	res += amount;
	resI = (int)res;

	// TM_STORE(&accounts[positions[0]], resI);
	// accounts[positions[0]] = resI;

  TM_COMMIT;

  return amount;
}

static void reset(bank_t *bank)
{
  long i;

  TM_START(2, RW);
  for (i = 0; i < bank->size; i++) {
    TM_STORE(&bank->accounts[i], 0);
  }
  TM_COMMIT;
}

/* ################################################################### *
* TRANSACTION THREADS
* ################################################################### */

static void* test_enqueuer(void *data)
{
	volatile int id;
	enqueue_req_t  *req;
	thread_data_t    *d = (thread_data_t *)data;
	cuda_t          *cd = d->cd;
	account_t *accounts = d->bank->accounts;
	int nb_accounts = d->bank->size;
	int rndOpt = RAND_R_FNC(d->seed);
	int rndOpt2 = RAND_R_FNC(d->seed);
	int curr_tx = 0;
	volatile int resReadOnly = 0;
	volatile unsigned seed = 0x213FAB;

	while (!flagEnqueuerToExit) { // TODO: EXIT condition

		// req = (enqueue_req_t*)malloc(sizeof(enqueue_req_t));
		req = &requests[request_index];
		request_index = (request_index + 1) % MAX_REQUESTS;

		int good_buffers_start = 0;
		int bad_buffers_start = size_of_CPU_input_buffer/sizeof(int) * NB_OF_BUFFERS;
		int buffers_start = isInterBatch ? bad_buffers_start : good_buffers_start;

		long index = buffers_start+id*NB_CPU_TXS_PER_THREAD + curr_tx;
		if (size_of_CPU_input_buffer * 2 * NB_OF_BUFFERS < index) {
			curr_tx = 0;
			index = buffers_start+id*NB_CPU_TXS_PER_THREAD + curr_tx;
		}
		
		req->accounts_vec[0] = CPUInputBuffer[index];

		// if (accounts_vec[0] == 0) {
		// 	printf("conflict access stm_baseMemPool=%p accounts=%p\n", stm_baseMemPool, accounts);
		// }

	#if BANK_PART == 3
		static thread_local unsigned offset = 4096 * (id*1000);
		offset += (4096 % nb_accounts) & ~0xFFF;
		req->accounts_vec[0] = accounts_vec[0] % 4096;
		req->accounts_vec[0] += offset;
		req->accounts_vec[0] %= (nb_accounts - 40);
		req->accounts_vec[0] |= 1;
	#endif /* BANK_PART == 3 */

		if (rndOpt % 100 < d->nb_read_intensive) {
			/* ** READ_INTENSIVE BRANCH (-R >0) ** */
			req->isReadIntensive = 1;
			// BANK_PREPARE_READ_INTENSIVE(d->id, d->seed, rnd, d->hmult, d->nb_threads, accounts_vec, nb_accounts);
			// TODO: change the buffers
			for (int i = 1; i < d->read_intensive_size; ++i) {
				req->accounts_vec[i] = req->accounts_vec[i-1]+1;
			}

			if (rndOpt2 % 100000000 < (d->prec_write_txs * 1000000)) {
				// 1% readIntensive
				req->isReadTX = 1;
				req->nbAccesses = d->read_intensive_size;
			} else {
				req->isReadTX = 0;
				req->nbAccesses = d->read_intensive_size;
			}
		} else {
			/* ** (current in use with -R 0) ** */
			req->isReadIntensive = 0;
			// BANK_PREPARE_TRANSFER(d->id, d->seed, rnd, d->hmult, d->nb_threads, accounts_vec, nb_accounts);
			for (int i = 1; i < d->read_intensive_size; ++i) {
				req->accounts_vec[i] = req->accounts_vec[i-1]+1;
			}
			// for (int i = 1; i < d->trfs*2; ++i)
			// 	accounts_vec[i] = accounts_vec[i-1] + parsedData.access_offset;
	#if BANK_PART == 5
			// first transactions are write
			// printf("NB_CPU_TXS_PER_THREAD - curr_tx = %i   (float)d->prec_write_txs*0.01f*(float)NB_CPU_TXS_PER_THREAD = %f\n",
			// 	NB_CPU_TXS_PER_THREAD - curr_tx, (float)d->prec_write_txs*0.01f*(float)NB_CPU_TXS_PER_THREAD);

			if ((NB_CPU_TXS_PER_THREAD - curr_tx) <= (float)d->prec_write_txs*0.01f*(float)NB_CPU_TXS_PER_THREAD) {
				req->isReadTX = 0;
				req->nbAccesses = d->read_intensive_size;
			} else {
				req->isReadTX = 1;
				req->nbAccesses = d->read_intensive_size;
			}

	#elif BANK_PART == 9 || BANK_PART == 10
			if (rndOpt2 % 100000000 < (d->prec_write_txs * 1000000)) {
				req->isReadTX = 0;
				req->nbAccesses = d->read_intensive_size;
				req->nbAccounts = nb_accounts;
				req->isInterConflict = isInterBatch;
			} else {
				req->isReadTX = 1;
				req->nbAccesses = d->read_intensive_size;
				req->nbAccounts = nb_accounts;
				req->isInterConflict = isInterBatch;
				// resReadOnly += transferReadOnly(d->bank->accounts, accounts_vec, d->trfs, 1);
			}
	#else /* BANK_PART == 5 */
			if (rndOpt2 % 100000000 < (d->prec_write_txs * 1000000)) {
				req->isReadTX = 0;
				req->nbAccesses = d->read_intensive_size;
			} else {
				req->isReadTX = 1;
				req->nbAccesses = d->read_intensive_size;
			}
	#endif /* BANK_PART == 5 */
		}
		sharedConcQueue.push(req);
		curr_tx += 1;
		if (d->nb_threadsCPU == 0) id = id;
		else id = (id + 1) % d->nb_threadsCPU;
	}
	return NULL;
}

static void test(int id, void *data)
{
	enqueue_req_t  *req;
  thread_data_t    *d = &((thread_data_t *)data)[id];
  cuda_t          *cd = d->cd;
  account_t *accounts = d->bank->accounts;
  volatile unsigned accounts_vec[d->trfs*20];
  int nb_accounts = d->bank->size;
	int rndOpt = RAND_R_FNC(d->seed);
	int rndOpt2 = RAND_R_FNC(d->seed);
	static thread_local int curr_tx = 0;
	static thread_local volatile int resReadOnly = 0;
	static thread_local volatile unsigned seed = 0x213FAB + id;

	int good_buffers_start = 0;
	int bad_buffers_start = size_of_CPU_input_buffer/sizeof(int) * NB_OF_BUFFERS;
	int buffers_start = isInterBatch ? bad_buffers_start : good_buffers_start;

	// for (int i = 0; i < (d->trfs * 20); ++i) {
	// 	int index = buffers_start+id*NB_CPU_TXS_PER_THREAD + curr_tx;
	// #if BANK_PART == 6
	// 	accounts_vec[i] = CPU_ACCESS(id+i+curr_tx, parsedData.nb_accounts-20); //CPUInputBuffer[index];
	// #else
	// 	accounts_vec[i] = CPUInputBuffer[index];
	// #endif
	// 	curr_tx = (curr_tx + 1) % NB_CPU_TXS_PER_THREAD;
	// }

	while (!sharedConcQueue.try_pop(req)) /* wait */; // TODO: the other thread spams mallocs....

	if (req->isReadIntensive) {
		/* ** READ_INTENSIVE BRANCH (-R >0) ** */
		if (req->isReadTX) {
			// 1% readIntensive
			readIntensive(d->bank->accounts, req->accounts_vec, req->nbAccesses, 1);
		} else {
			resReadOnly += readOnly(d->bank->accounts, req->accounts_vec, req->nbAccesses, 1);
		}
	} else {
		/* ** current in use with -R 0 ** */
#if BANK_PART == 5
		if (!req->isReadTX) {
			transfer(d->bank->accounts, req->accounts_vec, req->nbAccesses, 1);
		} else {
			resReadOnly += readOnly(d->bank->accounts, req->accounts_vec, req->nbAccesses, 1);
		}

#elif BANK_PART == 9 || BANK_PART == 10
		if (!req->isReadTX) {
			transfer2(d->bank->accounts, req->accounts_vec, req->isInterConflict, d->read_intensive_size, id, req->nbAccounts);
		} else {
			// resReadOnly += transferReadOnly(d->bank->accounts, accounts_vec, d->trfs, 1);
			resReadOnly += readOnly2(d->bank->accounts, req->accounts_vec, req->isInterConflict, req->nbAccesses, id, req->nbAccounts);
		}
#else /* BANK_PART == 5 */
		if (!req->isReadTX) {
			transfer(d->bank->accounts, req->accounts_vec, req->nbAccesses, 1);
		} else {
			resReadOnly += readOnly(d->bank->accounts, req->accounts_vec, req->nbAccesses, 1);
		}
#endif /* BANK_PART == 5 */
	}
	curr_tx += 1;
	curr_tx = curr_tx % NB_CPU_TXS_PER_THREAD;

	// asm volatile ("" ::: "memory");

	volatile uint64_t tsc = rdtsc();
	while ((rdtsc() - tsc) < parsedData.CPU_backoff);

	// asm volatile ("" ::: "memory");

	// free(req); // not needed because it is taken from the pool

  d->nb_transfer++;
  d->global_commits++;
}

static void beforeCPU(int id, void *data)
{
  thread_data_t *d = &((thread_data_t *)data)[id];
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

thread_local static unsigned long seed = 0x0012112A3112514A;

static void before_kernel(int id, void *data)
{
	if (isInterBatch) {
		memman_select("GPU_input_buffer_bad");
	} else {
		memman_select("GPU_input_buffer_good");
	}
	int bufferId = RAND_R_FNC(seed) % NB_OF_BUFFERS; // updates the seed
	int *cpuInput = (int*)memman_get_cpu(NULL) + size_of_GPU_input_buffer/sizeof(int) * bufferId;
	void *gpuInput = memman_get_gpu(NULL);
	CUDA_CPY_TO_DEV_ASYNC(gpuInput, cpuInput, size_of_GPU_input_buffer, PR_getCurrentStream()); // inputSteam
}

static void after_kernel(int, void*) { /* empty: should it copy some stuff */ }

static int nbBatches = 0;
static int nbConflBatches = 0;

static void before_batch(int id, void *data)
{
	RAND_R_FNC(seed); // updates the seed
	isInterBatch = IS_INTERSECT_HIT( seed );

	if (nbBatches % 3 == 0) {
		// check the ratio
		float ratio = ((float)nbConflBatches / (float)nbBatches);
		if (ratio < P_INTERSECT && !isInterBatch) {
			if (P_INTERSECT > 0) isInterBatch = 1;
		} else if (ratio > P_INTERSECT && isInterBatch) {
			if (P_INTERSECT < 1.0) isInterBatch = 0;
		}
	}

	nbBatches++;
	if (isInterBatch) nbConflBatches++;

	__sync_synchronize();
}

static void after_batch(int id, void *data)
{
	for (int j = 0; j < HETM_NB_DEVICES; ++j) {
		memman_select_device(j);
		memman_select("GPU_output_buffer");
		memman_cpy_to_cpu(NULL, NULL, *hetm_batchCount[j]);
	}
	// TODO: conflict mechanism
}

static void choose_policy(int, void*) { /* TODO: drop policy */ }

static void test_cuda(int id, void *data)
{
  thread_data_t    *d = &((thread_data_t *)data)[id];
  cuda_t          *cd = d->cd;
  account_t *accounts = d->bank->accounts;
  size_t bank_size = d->bank->size;

#if BANK_PART == 7 || BANK_PART == 8
	// makes the batch artificially longer
	struct timespec timeout = {
		.tv_sec = parsedData.GPU_batch_duration / 1000,
		.tv_nsec = (parsedData.GPU_batch_duration * 1000000) % 1000000000 // in millis
	};
	nanosleep(&timeout, NULL);
	// NOTE: Wait time must come before (kernel sets some flags for the waiting)
#endif /* BANK_PART == 7 */

  jobWithCuda_run(cd, accounts);

	int idGPUThread = HeTM_shared_data.nbCPUThreads;
	long TXsOnGPU = parsedData.GPUthreadNum*parsedData.GPUblockNum*parsedData.trans;
}

static void afterGPU(int id, void *data)
{
  thread_data_t *d = &((thread_data_t *)data)[id];

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
//   printf("nb_transfer=%li\n", d->nb_transfer);
//   printf("nb_batches=%li\n", d->nb_aborts_locked_read);
//   printf(" <<<<<<<< PR-STM aborts=%12li\n", HeTM_stats_data.nbAbortsGPU);
//   printf(" <<<<<<< PR-STM commits=%12li\n", HeTM_stats_data.nbCommittedTxsGPU);

  // leave this one
  printf("CUDA thread terminated after %li(%li successful) run(s). \nTotal cuda execution time: %f ms.\n",
    HeTM_stats_data.nbBatches, HeTM_stats_data.nbBatchesSuccess, HeTM_stats_data.timeGPU);
	HeTM_flush_barrier();
}

/* ################################################################### *
*
* MAIN
*
* ################################################################### */

int main(int argc, char **argv)
{
	bank_t *bank;
	// barrier_t cuda_barrier;
	int i, j, ret = -1;
	thread_data_t *data;
	pthread_t *threads;
	barrier_t barrier;
	sigset_t block_set;
	size_t mempoolSize;

	memset(&parsedData, 0, sizeof(thread_data_t));

	PRINT_FLAGS();

	// ##########################################
	// ### Input management
	bank_parseArgs(argc, argv, &parsedData);

	// ---------------------------------------------------------------------------
	// --- prototype (meaning shitty code)
	maxGPUoutputBufferSize = parsedData.GPUthreadNum*parsedData.GPUblockNum*sizeof(int);
	currMaxCPUoutputBufferSize = maxGPUoutputBufferSize; // realloc on full

	memman_set_nb_gpus(HETM_NB_DEVICES); // define the number of GPUs for memman

	// malloc_or_die(GPUoutputBuffer, maxGPUoutputBufferSize);
	memman_alloc_dual("GPU_output_buffer", maxGPUoutputBufferSize, 0);
	GPUoutputBuffer = (int*)memman_get_gpu(NULL);

	size_of_GPU_input_buffer = maxGPUoutputBufferSize; // each transaction has 1 input

	// TODO: this parsedData.nb_threads is what comes of the -n input
	size_of_CPU_input_buffer = parsedData.nb_threads*sizeof(int) * NB_CPU_TXS_PER_THREAD;

	// malloc_or_die(GPUInputBuffer, size_of_GPU_input_buffer*NB_OF_BUFFERS);
	memman_alloc_gpu("GPU_input_buffer", size_of_GPU_input_buffer, NULL, 0);
	GPUInputBuffer = (int*)memman_get_gpu(NULL);

	memman_alloc_cpu("GPU_input_buffer_good", size_of_GPU_input_buffer*NB_OF_BUFFERS, GPUInputBuffer, 0);
	memman_alloc_cpu("GPU_input_buffer_bad", size_of_GPU_input_buffer*NB_OF_BUFFERS, GPUInputBuffer, 0);

	// CPU Buffers
	malloc_or_die(CPUInputBuffer, size_of_CPU_input_buffer * 2 * NB_OF_BUFFERS); // good and bad
	malloc_or_die(CPUoutputBuffer, currMaxCPUoutputBufferSize); // kinda big

	fill_GPU_input_buffers();
	fill_CPU_input_buffers();
	// ---------------------------------------------------------------------------

	HeTM_init((HeTM_init_s){
// #if CPU_INV == 1
//     .policy       = HETM_CPU_INV,
// #else /* GPU_INV */
		.policy       = HETM_GPU_INV,
// #endif /**/
		.nbCPUThreads = parsedData.nb_threads,
		.nbGPUBlocks  = parsedData.GPUblockNum,
		.nbGPUThreads = parsedData.GPUthreadNum,
		.timeBudget   = parsedData.timeBudget,
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
	mempoolSize = parsedData.nb_accounts*sizeof(account_t);
	HeTM_mempool_init(mempoolSize, 0);

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
	malloc_or_die(bank, 1);
	HeTM_alloc((void**)&bank->accounts, (void**)&bank->devAccounts, mempoolSize);
	bank->size = parsedData.nb_accounts;
		memset(bank->accounts, 0, parsedData.nb_accounts * sizeof(account_t));
	printf("Total before   : %d (array ptr=%p size=%zu)\n", total(bank, 0),
			bank->accounts, parsedData.nb_accounts * sizeof(account_t));
	parsedData.bank = bank;

	DEBUG_PRINT("Initializing GPU.\n");

	cuda_t *cuda_st;
	cuda_st = jobWithCuda_init(bank->accounts, parsedData.nb_threadsCPU,
		bank->size, parsedData.trans, 0, parsedData.GPUthreadNum, parsedData.GPUblockNum,
		parsedData.hprob, parsedData.hmult);

	parsedData.cd = cuda_st;
	//DEBUG_PRINT("Base: %lu %lu \n", bank->accounts, &bank->accounts);
	if (cuda_st == NULL) {
		printf("CUDA init failed.\n");
		exit(-1);
	}

	/* Init STM */
	printf("Initializing STM\n");

	TM_INIT(parsedData.nb_threads);
	// ###########################################################################
	// ### Start iterations ######################################################
	// ###########################################################################
	
	pthread_t enqueuer;
	// TODO: memory pool
	requests = (enqueue_req_t *)malloc(sizeof(enqueue_req_t)*MAX_REQUESTS);
	if (pthread_create(&enqueuer, NULL, test_enqueuer, &parsedData)) {
		perror("launch enqueuer thread");
	}

	for (j = 0; j < parsedData.iter; j++) { // Loop for testing purposes
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
			data[i].seed    = input_seed * (i ^ 12345);// rand();
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

		// for (int i = 0; i < bank->size; ++i) {
		// 	printf("[%i]=%i ", i, bank->accounts[i]);
		// }
		// printf("\n");

	/* Cleanup STM */
	TM_EXIT();
	/*Cleanup GPU*/
		HeTM_flush_barrier();
	jobWithCuda_exit(cuda_st);
	free(cuda_st);
	// ### End iterations ########################################################

	bank_statsFile(&parsedData);

	/* Delete bank and accounts */
	HeTM_mempool_destroy();
	free(bank);

	free(threads);
	free(data);

	__atomic_store_n(&flagEnqueuerToExit, 1, __ATOMIC_RELEASE);
	pthread_join(enqueuer, NULL);
	free(requests);

  return EXIT_SUCCESS;
}
