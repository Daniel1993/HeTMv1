#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define __USE_GNU

#include <random>

#include "bank.hpp"
#include "hetm-cmp-kernels.cuh"
#include "bank_aux.h"
#include "CheckAllFlags.h"

/* ################################################################### *
* GLOBALS
* ################################################################### */

thread_local static std::random_device randDev{};
thread_local static std::mt19937 generator{randDev()};
thread_local static std::geometric_distribution<> *distGen;

// global
thread_data_t parsedData;
int isInterBatch = 0;

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

static const int PAGE_SIZE = 4096; // TODO: this is also defined in hetm proj (CACHE_GRANULE_SIZE)

// -----------------------------------------------------------------------------

#define INTEREST_RATE 0.5

#define COMPUTE_TRANSFER(val) \
	val // TODO: do math that does not kill the final result

static int fill_GPU_input_buffers()
{
	int buffer_last = size_of_GPU_input_buffer/sizeof(int);

	memman_select("GPU_input_buffer_good");
	int *cpu_ptr = (int*)memman_get_cpu(NULL);

	int nbPages = parsedData.nb_accounts / PAGE_SIZE;
	// if (parsedData.nb_accounts % PAGE_SIZE) nbPages++;

#if BANK_PART == 1
	unsigned rnd = RAND_R_FNC(input_seed);
	for (int i = 0; i < buffer_last; ++i) {
		unsigned j = rnd % parsedData.nb_accounts;
		int target_page = (j / parsedData.GPUthreadNum) % nbPages;
		int access = (j % PAGE_SIZE) + target_page * PAGE_SIZE;
		cpu_ptr[i] = access;
		cpu_ptr[i] &= (unsigned)-2; // get even accounts
		rnd += parsedData.access_offset * 2; // TODO: 2 is sizeof TX
	}

	memman_select("GPU_input_buffer_bad");
	cpu_ptr = (int*)memman_get_cpu(NULL);

	cpu_ptr[0] = 0; // deterministic intersection
	rnd = RAND_R_FNC(input_seed);
	for (int i = 1; i < buffer_last; ++i) {
		unsigned j = rnd % parsedData.nb_accounts;
		int target_page = (j / parsedData.GPUthreadNum) % nbPages;
		int access = (j % PAGE_SIZE) + target_page * PAGE_SIZE;
		cpu_ptr[i] = access;
		cpu_ptr[i] &= (unsigned)-2; // get even accounts
		rnd += parsedData.access_offset * 2; // TODO: 2 is sizeof TX
	}
#else /* Uniform at random */
	for (int i = 0; i < buffer_last; ++i) {
		unsigned rnd = RAND_R_FNC(input_seed);
		unsigned j = rnd % parsedData.nb_accounts;
		int target_page = (j / parsedData.GPUthreadNum) % nbPages;
		int access = (j % PAGE_SIZE) + target_page * PAGE_SIZE;
		cpu_ptr[i] = access;
		cpu_ptr[i] &= (unsigned)-2; // get even accounts
	}

	memman_select("GPU_input_buffer_bad");
	cpu_ptr = (int*)memman_get_cpu(NULL);

	cpu_ptr[0] = 0; // deterministic intersection
	for (int i = 1; i < buffer_last; ++i) {
		unsigned rnd = RAND_R_FNC(input_seed);
		unsigned j = rnd % parsedData.nb_accounts;
		int target_page = (j / parsedData.GPUthreadNum) % nbPages;
		int access = (j % PAGE_SIZE) + target_page * PAGE_SIZE;
		cpu_ptr[i] = access;
		cpu_ptr[i] &= (unsigned)-2; // get even accounts
	}
#endif /* BANK_PART == 1 */
}

static int fill_CPU_input_buffers()
{
	int good_buffers_last = size_of_CPU_input_buffer/sizeof(int);
	int bad_buffers_last = 2*size_of_CPU_input_buffer/sizeof(int);

	// TODO: make the CPU access the end --> target_page * PAGE_SIZE - access

	int nbPages = parsedData.nb_accounts / PAGE_SIZE;
	// if (parsedData.nb_accounts % PAGE_SIZE) nbPages++;
	// TODO: if the last page is incomplete it is never accessed

#if BANK_PART == 1

	unsigned rnd = RAND_R_FNC(input_seed);
	for (int i = 0; i < good_buffers_last; ++i) {
		unsigned j = rnd % parsedData.nb_accounts;
		int target_page = (j / parsedData.nb_threadsCPU) % nbPages;
		int access = (j % PAGE_SIZE) + target_page * PAGE_SIZE;
		CPUInputBuffer[i] = access;
		CPUInputBuffer[i] |= 1; // get odd accounts
		rnd += parsedData.access_offset * 2; // TODO: 2 is sizeof TX
	}
	// for (int i = 0; i < parsedData.nb_threadsCPU; ++i) {
	// 	CPUInputBuffer[good_buffers_last + i*NB_CPU_TXS_PER_THREAD] = i; // deterministic intersection
	// }
	rnd = RAND_R_FNC(input_seed);
	for (int i = good_buffers_last; i < bad_buffers_last; ++i) {
		unsigned j = rnd % parsedData.nb_accounts;
		int target_page = (j / parsedData.nb_threadsCPU) % nbPages;
		int access = (j % PAGE_SIZE) + target_page * PAGE_SIZE;
		CPUInputBuffer[i] = access;
		CPUInputBuffer[i] |= 1; // get odd accounts
		rnd += parsedData.access_offset * 2; // TODO: 2 is sizeof TX
	}
#else /* BANK_PART != 3 */
	for (int i = 0; i < good_buffers_last; ++i) {
		unsigned rnd = RAND_R_FNC(input_seed);
		unsigned j = rnd % parsedData.nb_accounts;
		int target_page = (j / parsedData.nb_threadsCPU) % nbPages;
		int access = (j % PAGE_SIZE) + target_page * PAGE_SIZE;
		CPUInputBuffer[i] = access;
		CPUInputBuffer[i] |= 1; // get odd accounts
	}
	// for (int i = 0; i < parsedData.nb_threadsCPU; ++i) {
	// 	CPUInputBuffer[good_buffers_last + i*NB_CPU_TXS_PER_THREAD] = i; // deterministic intersection
	// }
	for (int i = good_buffers_last; i < bad_buffers_last; ++i) {
		unsigned rnd = RAND_R_FNC(input_seed);
		unsigned j = rnd % parsedData.nb_accounts;
		int target_page = (j / parsedData.nb_threadsCPU) % nbPages;
		int access = (j % PAGE_SIZE) + target_page * PAGE_SIZE;
		CPUInputBuffer[i] = access;
		CPUInputBuffer[i] |= 1; // get odd accounts
	}
#endif /* BANK_PART == 3 */
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

static inline int transfer(account_t *accounts, volatile int *positions, int count, int amount)
{
  int i;
	uintptr_t load1, load2;
  int z = 0;
  int n, src, dst;

  for (n = 0; n < count; n += 2) {
    src = positions[n];
    dst = positions[n+1];
		// printf ("%i -- %i\n", positions[n], positions[n+1]);
    if (src < 0 || dst < 0) {
      return 0;
    }
    i += accounts[src];
    i += accounts[dst];
  }

  /* Allow overdrafts */
  TM_START(z, RW);

  for (n = 0; n < count; n += 2) {
    src = positions[n];
    dst = positions[n+1];

		// Problem: TinySTM works with the granularity of 8B, PR-STM works with 4B
    load1 = TM_LOAD(&accounts[src]);
    load1 -= COMPUTE_TRANSFER(amount);

    load2 = TM_LOAD(&accounts[dst]);
    load2 += COMPUTE_TRANSFER(amount);

		TM_STORE(&accounts[src], load1);
    TM_STORE(&accounts[dst], load2);
  }

  TM_COMMIT;

  // TODO: remove this
//   volatile int j = 0;
// loop:
//   j++;
//   if (j < 100) goto loop;

  return amount;
}

static inline int transferReadOnly(account_t *accounts, volatile int *positions, int count, int amount)
{
  int i;
	uintptr_t load1, load2;
  int z = 0;
  int n, src, dst;

  for (n = 0; n < count; n += 2) {
    src = positions[n];
    dst = positions[n+1];
		// printf ("%i -- %i\n", positions[n], positions[n+1]);
    if (src < 0 || dst < 0) {
      return 0;
    }
    i += accounts[src];
    i += accounts[dst];
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

static inline int readIntensive(account_t *accounts, volatile int *positions, int count, int amount)
{
  int i;
  int z;
  int n;
	int loads[count];
	float res = 0;
	int resI = 0;

  // for (n = 0; n < count; ++n) {
  //   z = positions[n];
  //   i += accounts[z]; // prefetch
  // }

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
	// accounts[positions[0]] = resI;

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

static inline int readOnly(account_t *accounts, volatile int *positions, int count, int amount)
{
  int i;
  int z;
  int n;
	int loads[count];
	float res = 0;
	int resI = 0;

	z = 0;
  TM_START(z, RW);

  for (n = 0; n < count; ++n) {
    loads[n] = TM_LOAD(&accounts[positions[n]]);
		res += loads[n] * INTEREST_RATE;
  }
	res += amount;
	resI = (int)res;

	// TM_STORE(&accounts[positions[0]], resI);
	// accounts[positions[0]] = resI;

  TM_COMMIT;

  return amount;
}

static inline void reset(bank_t *bank)
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

static void test(int id, void *data)
{
  thread_data_t    *d = &((thread_data_t *)data)[id];
  cuda_t          *cd = d->cd;
  account_t *accounts = d->bank->accounts;
  volatile int accounts_vec[d->trfs*20];
  int nb_accounts = d->bank->size;
  unsigned rnd;
	int rndOpt = RAND_R_FNC(d->seed);
	int rndOpt2 = RAND_R_FNC(d->seed);
	static thread_local int curr_tx = 0;
	static thread_local volatile int resReadOnly = 0;

#if BANK_PART == 2
  rnd = (unsigned)d->hprob;
#else
  rnd = (*distGen)(generator);
#endif

	int good_buffers_start = 0;
	int bad_buffers_start = size_of_CPU_input_buffer/sizeof(int);
	int buffers_start = isInterBatch ? bad_buffers_start : good_buffers_start;
	int index = buffers_start+id*NB_CPU_TXS_PER_THREAD + curr_tx;
	accounts_vec[0] = CPUInputBuffer[index];
	if (rndOpt % 100 < d->nb_read_intensive) {
		// BANK_PREPARE_READ_INTENSIVE(d->id, d->seed, rnd, d->hmult, d->nb_threads, accounts_vec, nb_accounts);
		// TODO: change the buffers
		for (int i = 1; i < d->read_intensive_size; ++i)
			accounts_vec[i] = accounts_vec[i-1] + 2; // only odd accounts

		if (rndOpt2 % 100 < d->prec_write_txs) {
			// 1% readIntensive
			readIntensive(d->bank->accounts, accounts_vec, d->read_intensive_size, 1);
		} else {
			resReadOnly += readOnly(d->bank->accounts, accounts_vec, d->read_intensive_size, 1);
		}
	} else {
		// BANK_PREPARE_TRANSFER(d->id, d->seed, rnd, d->hmult, d->nb_threads, accounts_vec, nb_accounts);
		for (int i = 1; i < d->trfs*2; ++i)
			accounts_vec[i] = accounts_vec[i-1] + parsedData.access_offset;
		if (rndOpt2 % 100 < d->prec_write_txs) {
			transfer(d->bank->accounts, accounts_vec, d->trfs, 1);
		} else {
			transferReadOnly(d->bank->accounts, accounts_vec, d->trfs, 1);
		}
	}
	curr_tx += 1;
	curr_tx = curr_tx % NB_CPU_TXS_PER_THREAD;

  d->nb_transfer++;
  d->global_commits++;
}

static void beforeCPU(int id, void *data)
{
  thread_data_t *d = &((thread_data_t *)data)[id];
  distGen = new std::geometric_distribution<>(parsedData.stddev);
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
	// TODO: moved to GPU callback
}

static void after_batch(int id, void *data)
{
	memman_select("GPU_output_buffer");
	memman_cpy_to_cpu(NULL, NULL, *hetm_batchCount);
	// TODO: conflict mechanism
}

static void choose_policy(int, void*) {
  // -----------------------------------------------------------------------
  int idGPUThread = HeTM_shared_data.nbCPUThreads;
  long TXsOnCPU = 0,
		TXsOnGPU = parsedData.GPUthreadNum*parsedData.GPUblockNum*parsedData.trans;
  for (int i = 0; i < HeTM_shared_data.nbCPUThreads; ++i) {
    TXsOnCPU += HeTM_shared_data.threadsInfo[i].curNbTxs;
  }

  // TODO: this picks the one with higher number of TXs
	// TODO: GPU only gets the stats in the end --> need to remove the dropped TXs
	HeTM_stats_data.nbTxsGPU += TXsOnGPU;
  if (TXsOnCPU > TXsOnGPU) {
    HeTM_gshared_data.policy = HETM_GPU_INV;
		if (HeTM_is_interconflict()) {
			HeTM_stats_data.nbDroppedTxsGPU += TXsOnGPU;
		} else {
			HeTM_stats_data.nbCommittedTxsGPU += TXsOnGPU;
		}
  } else {
    HeTM_gshared_data.policy = HETM_CPU_INV;
		HeTM_stats_data.nbCommittedTxsGPU += TXsOnGPU;
  }
  // -----------------------------------------------------------------------
}

static void test_cuda(int id, void *data)
{
  thread_data_t    *d = &((thread_data_t *)data)[id];
  cuda_t          *cd = d->cd;
  account_t *accounts = d->bank->accounts;

	thread_local static unsigned long seed = 0x0012112A3112514A;
	isInterBatch = IS_INTERSECT_HIT( RAND_R_FNC(seed) );
	__sync_synchronize();

	if (isInterBatch) {
		memman_select("GPU_input_buffer_bad");
	} else {
		memman_select("GPU_input_buffer_good");
	}
	memman_cpy_to_gpu(HeTM_memStream2, NULL, *hetm_batchCount);

  jobWithCuda_run(cd, accounts);
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
  // printf("nb_transfer=%li\n", d->nb_transfer);
  // printf("nb_batches=%li\n", d->nb_aborts_locked_read);
  // printf(" <<<<<<<< PR-STM aborts=%12li\n", HeTM_stats_data.nbAbortsGPU);
  // printf(" <<<<<<< PR-STM commits=%12li\n", HeTM_stats_data.nbCommittedTxsGPU);

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
  bank_t *bank;
  // barrier_t cuda_barrier;
  int i, j, ret = -1;
  thread_data_t *data;
  pthread_t *threads;
  barrier_t barrier;
  sigset_t block_set;
  size_t accountsSize;

  memset(&parsedData, 0, sizeof(thread_data_t));

  PRINT_FLAGS();

  // ##########################################
  // ### Input management
  bank_parseArgs(argc, argv, &parsedData);

	// ---------------------------------------------------------------------------
	// --- prototype (meaning shitty code)
	maxGPUoutputBufferSize = parsedData.GPUthreadNum*parsedData.GPUblockNum*sizeof(int);
	currMaxCPUoutputBufferSize = maxGPUoutputBufferSize; // realloc on full

	// malloc_or_die(GPUoutputBuffer, maxGPUoutputBufferSize);
	memman_alloc_dual("GPU_output_buffer", maxGPUoutputBufferSize, 0);
	GPUoutputBuffer = (int*)memman_get_gpu(NULL);

	size_of_GPU_input_buffer = maxGPUoutputBufferSize*parsedData.trans;

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
  accountsSize = parsedData.nb_accounts*sizeof(account_t);
  HeTM_mempool_init(accountsSize);

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
  HeTM_alloc((void**)&bank->accounts, (void**)&bank->devAccounts, accountsSize);
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
      data[i].seed    = rand();
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

	// for (int i = 0; i < bank->size; ++i) {
	// 	printf("[%i]=%i ", i, bank->accounts[i]);
	// }
	// printf("\n");

  /* Cleanup STM */
  TM_EXIT();
  /*Cleanup GPU*/
  jobWithCuda_exit(cuda_st);
  free(cuda_st);
  // ### End iterations ########################################################

  bank_statsFile(&parsedData);

  /* Delete bank and accounts */
  HeTM_mempool_destroy();
  free(bank);

  free(threads);
  free(data);

  return EXIT_SUCCESS;
}
