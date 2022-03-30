#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define __USE_GNU

#include <cmath>

#include "bank.hpp"
#include "hetm-cmp-kernels.cuh"
#include "setupKernels.cuh"
#include "bank_aux.h"
#include "CheckAllFlags.h"
#include "input_handler.h"
#include "rdtsc.h"
#include "all_bank_parts.cuh"

using namespace memman;

static void fill_input_buffers()
{
#if BANK_PART == 1 /* Uniform at random: split */
	bank_part1_init();
#elif BANK_PART == 2 /* Uniform at random: interleaved CPU accesses incremental pages */
	bank_part2_init();
#elif BANK_PART == 3 /* Zipf: from file */
	bank_part3_init();
#elif BANK_PART == 4 /* Zipf */
	bank_part4_init();
#elif BANK_PART == 5
	bank_part5_init();
#elif BANK_PART == 6 /* contiguous */
	bank_part6_init();
#elif BANK_PART == 7 /* not used on CPU, GPU blocks */
	bank_part7_init();
#elif BANK_PART == 8 // same as 7
	bank_part8_init();
#elif BANK_PART == 9 // same as 5 
	bank_part9_init();
#elif BANK_PART == 10 // same as 5
	bank_part10_init();
#else
#error "ERROR: Bank part not defined"
#endif
}

/* ################################################################### *
* TRANSACTION THREADS
* ################################################################### */

static void test(int id, void *data)
{
  	thread_data_t *d = &((thread_data_t *)data)[id];
	int curr_tx = 0;
#if BANK_PART == 1 /* Uniform at random: split */
	bank_part1_cpu_run(id, d);
#elif BANK_PART == 2 /* Uniform at random: interleaved CPU accesses incremental pages */
	bank_part2_cpu_run(id, d);
#elif BANK_PART == 3 /* Zipf: from file */
	bank_part3_cpu_run(id, d);
#elif BANK_PART == 4 /* Zipf */
	bank_part4_cpu_run(id, d);
#elif BANK_PART == 5
	bank_part5_cpu_run(id, d);
#elif BANK_PART == 6 /* contiguous */
	bank_part6_cpu_run(id, d);
#elif BANK_PART == 7 /* not used on CPU, GPU blocks */
	bank_part7_cpu_run(id, d);
#elif BANK_PART == 8 // same as 7
	bank_part8_cpu_run(id, d);
#elif BANK_PART == 9 // same as 5 
	bank_part9_cpu_run(id, d);
#elif BANK_PART == 10 // same as 5
	bank_part10_cpu_run(id, d);
#else
#error "ERROR: Bank part not defined"
#endif
	curr_tx += 1;
	curr_tx = curr_tx % NB_CPU_TXS_PER_THREAD;

	// asm volatile ("" ::: "memory");

	volatile uint64_t tsc = rdtsc();
	while ((rdtsc() - tsc) < (uint64_t)parsedData.CPU_backoff);

	// asm volatile ("" ::: "memory");

	d->nb_transfer++;
	d->global_commits++;
}

static void beforeCPU(int id, void *data)
{
  // thread_data_t *d = &((thread_data_t *)data)[id];
}

static void afterCPU(int id, void *data)
{
  // thread_data_t *d = &((thread_data_t *)data)[id];
  BANK_GET_STATS((&((thread_data_t *)data)[id]));
}

/* ################################################################### *
* CUDA THREAD
* ################################################################### */

// TODO: add a beforeBatch and afterBatch callbacks

thread_local static unsigned long seed = 0x0012112A3112514A;

static void before_kernel(int id, void *data)
{
	int *cpuInput;
	int *cpuPtr;
	int bufferId;
	
	for (int j = 0; j < HETM_NB_DEVICES; ++j)
	{
		MemObj *m;
		Config::GetInstance()->SelDev(j);
		if (isInterBatch)
			m = GPU_input_buffer_bad.GetMemObj(j);
		else
			m = GPU_input_buffer_good.GetMemObj(j);
		cpuInput = (int*)m->host;
		bufferId = RAND_R_FNC(seed) % NB_OF_BUFFERS; // updates the seed
		cpuPtr = cpuInput + size_of_GPU_input_buffer/sizeof(int) * bufferId;

		void *gpuInput = GPU_input_buffer.GetMemObj(j)->dev;
		CUDA_CPY_TO_DEV_ASYNC(gpuInput, cpuPtr, size_of_GPU_input_buffer, PR_getCurrentStream()); // inputSteam
	}
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
	for (int j = 0; j < HETM_NB_DEVICES; ++j)
	{
		Config::GetInstance()->SelDev(j);
		GPU_output_buffer.GetMemObj(j)->CpyDtH();
	}
	// TODO: conflict mechanism
}

static void choose_policy(int, void*) {
  // -----------------------------------------------------------------------
  // int idGPUThread = HeTM_shared_data.nbCPUThreads;
  // long TXsOnCPU = 0;
	// long TXsOnGPU = parsedData.GPUthreadNum*parsedData.GPUblockNum*parsedData.trans;
  // for (int i = 0; i < HeTM_shared_data.nbCPUThreads; ++i) {
  //   TXsOnCPU += HeTM_shared_data.threadsInfo[i].curNbTxs;
  // }

  // TODO: this picks the one with higher number of TXs
	// TODO: GPU only gets the stats in the end --> need to remove the dropped TXs
// #if BANK_PART != 7 && BANK_PART != 8
// 	HeTM_stats_data.nbTxsGPU += TXsOnGPU;
//
// 	if (HeTM_gshared_data.policy == HETM_GPU_INV) {
// 		if (HeTM_is_interconflict()) {
// 			HeTM_stats_data.nbDroppedTxsGPU += TXsOnGPU;
// 		} else {
// 			HeTM_stats_data.nbCommittedTxsGPU += TXsOnGPU;
// 		}
// 	} else if (HeTM_gshared_data.policy == HETM_CPU_INV) {
// 		HeTM_stats_data.nbCommittedTxsGPU += TXsOnGPU;
// 	}
// #endif /* BANK_PART != 7 */

	// can only choose the policy for the next round
  // if (TXsOnCPU > TXsOnGPU) {
  //   HeTM_gshared_data.policy = HETM_GPU_INV;
  // } else {
  //   HeTM_gshared_data.policy = HETM_CPU_INV;
  // }
  // -----------------------------------------------------------------------
}

static void test_cuda(int id, void *data)
{
  thread_data_t    *d = &((thread_data_t *)data)[id];
  cuda_t          *cd = d->cd;
  account_t *accounts = d->bank->accounts;
//   int devId = memman_get_curr_dev();
  // size_t bank_size = d->bank->size;

// #if HETM_CPU_EN == 0
// 	// with GPU only, CPU samples some data between batches
// 	for (int i = 0; i < bank_size; ++i) {
// 		bank_cpu_sample_data += accounts[i];
// 	}
// #endif /* HETM_CPU_EN == 0 */

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

	// int idGPUThread = HeTM_shared_data.nbCPUThreads;
	// long TXsOnGPU = parsedData.GPUthreadNum*parsedData.GPUblockNum*parsedData.trans;

// #if BANK_PART != 7 && BANK_PART != 8
// 	HeTM_stats_data.nbTxsGPU += TXsOnGPU;
//
// 	if (HeTM_gshared_data.policy == HETM_GPU_INV) {
// 		if (HeTM_is_interconflict()) {
// 			HeTM_stats_data.nbDroppedTxsGPU += TXsOnGPU;
// 		} else {
// 			HeTM_stats_data.nbCommittedTxsGPU += TXsOnGPU;
// 		}
// 	} else if (HeTM_gshared_data.policy == HETM_CPU_INV) {
// 		HeTM_stats_data.nbCommittedTxsGPU += TXsOnGPU;
// 	}
// #endif /* BANK_PART != 7 */
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
//   printf("nb_transfer=%li\n", d->nb_transfer);
//   printf("nb_batches=%li\n", d->nb_aborts_locked_read);
//   printf(" <<<<<<<< PR-STM aborts=%12li\n", HeTM_stats_data.nbAbortsGPU);
//   printf(" <<<<<<< PR-STM commits=%12li\n", HeTM_stats_data.nbCommittedTxsGPU);

  // leave this one
  printf("CUDA thread terminated after %li(%li successful) run(s). \nTotal cuda execution time: %f ms.\n",
    HeTM_stats_data.nbBatches, HeTM_stats_data.nbBatchesSuccess, HeTM_stats_data.timeGPU);
	for (int j = 0; j < HETM_NB_DEVICES; ++j) {
		HeTM_flush_barrier(j);
	}
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
	int i, j;
	thread_data_t *data;
	pthread_t *threads;
	sigset_t block_set;
	size_t mempoolSize;
	const int nbGPUs = HETM_NB_DEVICES;

	memset(&parsedData, 0, sizeof(thread_data_t));

	PRINT_FLAGS();

	Config::GetInstance(nbGPUs, CHUNK_GRAN*sizeof(PR_GRANULE_T)); // define the number of GPUs for memman
	
	// ##########################################
	// ### Input management
	bank_parseArgs(argc, argv, &parsedData);

	// ---------------------------------------------------------------------------
	// --- prototype (meaning shitty code)
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
			->SetSize(size_of_GPU_input_buffer*NB_OF_BUFFERS)
			->SetDevPtr(m_input_buffer->dev)
			->AllocHostPtr(),
			j
		));
		GPU_input_buffer_bad.AddMemObj(m_input_buffer_bad = new MemObj(b_input_buffer_bad
			.SetOptions(MEMMAN_NONE)
			->SetSize(size_of_GPU_input_buffer*NB_OF_BUFFERS)
			->SetDevPtr(m_input_buffer->dev)
			->AllocHostPtr(),
			j
		));
	}

	// CPU Buffers
	malloc_or_die(CPUInputBuffer, size_of_CPU_input_buffer * 2 * NB_OF_BUFFERS); // good and bad
	malloc_or_die(CPUoutputBuffer, currMaxCPUoutputBufferSize); // kinda big

	fill_input_buffers();
	// fill_GPU_input_buffers();
	// fill_CPU_input_buffers();
	// ---------------------------------------------------------------------------

  	// #define EXPLICIT_LOG_BLOCK (parsedData.trans * BANK_NB_TRANSFERS)
  	HeTM_set_explicit_log_block_size(parsedData.trans * BANK_NB_TRANSFERS); // TODO:

	// TODO: needs to pass here the number of GPUs (or get from memman)
  	HeTM_init((HeTM_init_s) {
// #if CPU_INV == 1
//     .policy       = HETM_CPU_INV,
// #else /* GPU_INV */
    	.policy       = HETM_GPU_INV, // TODO: not used anymore
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
			.mempool_size = (long)(mempoolSize = parsedData.nb_accounts*sizeof(account_t)),
			.mempool_opts = MEMMAN_NONE
		});

	// TODO:
	parsedData.nb_threadsCPU = HeTM_gshared_data.nbCPUThreads;
	parsedData.nb_threads    = HeTM_gshared_data.nbThreads;
	bank_check_params(&parsedData);

	malloc_or_die(data, parsedData.nb_threads + 1);
	memset(data, 0, (parsedData.nb_threads + 1)*sizeof(thread_data_t)); // safer
	parsedData.dthreads     = data;
	// ##########################################

	jobWithCuda_exit(NULL); // Reset Cuda Device

	malloc_or_die(threads, parsedData.nb_threads);
	malloc_or_die(bank, nbGPUs);
	for (int j = 0; j < nbGPUs; ++j)
	{
		HeTM_alloc(j, (void**)&bank[j].accounts, (void**)&bank[j].devAccounts, mempoolSize);
		bank[j].size = parsedData.nb_accounts;
		memset(bank[j].accounts, 0, mempoolSize);
	}
	printf("Total before   : %d (array ptr=%p size=%zu)\n", total(&(bank[0]), 0),
			bank[0].accounts, mempoolSize);
	parsedData.bank = bank;

	DEBUG_PRINT("Initializing GPU.\n");

	cuda_t *cuda_st;
	cuda_st = jobWithCuda_init(bank[0].accounts, parsedData.nb_threadsCPU,
		bank[0].size, parsedData.trans, 0, parsedData.GPUthreadNum, parsedData.GPUblockNum,
		parsedData.hprob, parsedData.hmult);

	parsedData.cd = cuda_st;
	//DEBUG_PRINT("Base: %lu %lu \n", bank->accounts, &bank->accounts);
	if (cuda_st == NULL)
	{
		printf("CUDA init failed.\n");
		exit(-1);
	}

	/* Init STM */
	printf("Initializing STM\n");

	TM_INIT(parsedData.nb_threads);
	// ###########################################################################
	// ### Start iterations ######################################################
	// ###########################################################################
	for (j = 0; j < parsedData.iter; j++)
	{ // Loop for testing purposes
		//Clear flags
		for (int d = 0; d < nbGPUs; d++)
		{
			HeTM_set_is_stop(d, 0);
		}
		global_fix = 0;

		// ##############################################
		// ### create threads
		// ##############################################
		printf(" >>> Creating %d threads\n", parsedData.nb_threads);
		for (i = 0; i < parsedData.nb_threads; i++)
		{
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

		/* Start threads, set callbacks */
		HeTM_after_gpu_finish(afterGPU);
		HeTM_before_cpu_start(beforeCPU);
		HeTM_after_cpu_finish(afterCPU);
		HeTM_after_batch(after_batch);
		HeTM_before_batch(before_batch);
		HeTM_after_kernel(after_kernel);
		HeTM_before_kernel(before_kernel);
		HeTM_choose_policy(choose_policy);

		HeTM_start(test, test_cuda, data);
		printf("STARTING...(Run %d)\n", j);

		TIMER_READ(parsedData.start);

		if (parsedData.duration > 0)
		{
			nanosleep(&parsedData.timeout, NULL);
		}
		else
		{
			sigemptyset(&block_set);
			sigsuspend(&block_set);
		}

		for (int d = 0; d < nbGPUs; d++)
		{
			HeTM_set_is_stop(d, 1);
			HeTM_async_set_is_stop(d, 1);
		}

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
	HeTM_mempool_destroy(0);
	free(bank);

	free(threads);
	free(data);

	return EXIT_SUCCESS;
}
