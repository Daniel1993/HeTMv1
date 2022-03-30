#define _GNU_SOURCE
#define __USE_GNU

#include "bank.hpp"
#include "bank_aux.h"
#include "CheckAllFlags.h"

/* ################################################################### *
* GLOBALS
* ################################################################### */

// TODO: Memcached stores strings, in the format <key:int, path:string>
//  --> this one only stores ints <key:int, value:int>

// global
thread_data_t parsedData;

static volatile int global_ts = 1;

packet_t search_entry(memcd_t *b, unsigned int *positions)
{
  int z = 0;
  int n, base, goal;
  int val, vers;
  int pos;
  packet_t response;

  /* Allow overdrafts */
  TM_START(z, RW);

  base = positions[0] * b->ways;
  goal = (positions[0] & 0xffff) % b->ways;

  if (goal > 0) {
    for (n = 0; n < goal; n++) {
      val  = TM_LOAD(&b->accounts[base+n]);
      vers = TM_LOAD(&b->version[base+n]);

      response.key  = val;
      response.vers = vers;
    }
    n -= 1;
    vers++;
    TM_STORE(&b->version[base+n], vers);
  }
  TM_COMMIT; // TODO: this must be at the same level as TM_START

  if (goal > 0) {
    pos = (base + n) >> BM_HASH;
    b->version_bm[pos] = 1;
  }

  return response;
}

packet_t new_entry(memcd_t *b, unsigned int *positions)
{
  int z = 0;
  int n, base, goal;
  int val, vers, min_pos, min;
  packet_t response;

  /* Allow overdrafts */
  TM_START(z, RW);
  min = -1;
  min_pos = 0;

  base = positions[0] * b->ways; // position in the hash map
  goal = positions[0] & 0xffff;

  // printf("\nbase=%5i goal=%5i b->ways=%5i\n", base, goal, b->ways);
  for (n = 0; n < b->ways; n++) {

	  val = TM_LOAD(&b->accounts[base+n]);
	  vers = TM_LOAD(&b->version[base+n]);

    // printf("   n=%5i  val=%5i    vers=%5i\n", n, val, vers);

	  // Check if it is free or if it is the same value
	  if (val == goal || vers == 0) {
  		min_pos = n;
  		break;
	  } else {
		  // Record timestamp value if is the oldest
		  if ( (min == -1) || (vers < min) ) {
  			min = vers;
  			min_pos = n;
		  }
	  }
  }

  vers++; // increase the TS

  TM_STORE(&b->accounts[base + min_pos], goal);
  TM_STORE(&b->version[base + min_pos], vers);
  // printf("b->accounts[%i]=%i b->version[%i]=%i min_pos=%i global_ts=%i\n",
  //   base + min_pos, b->accounts[base + min_pos], base + min_pos,
  //   b->version[base + min_pos], min_pos, global_ts);

  response.key = goal;
  response.vers = base + min_pos;

  TM_COMMIT;

  return response;
}

/* ################################################################### *
* TRANSACTION THREADS
* ################################################################### */

static __thread int queue_current = 0;
static __thread unsigned int *queue_pointer = NULL;

static void test(int id, void *data)
{
  int nb;
  int accounts_vec[MAX_ITER];
  thread_data_t *d = &((thread_data_t *)data)[id];
  int nb_accounts = d->cd->queue_size;
  size_t partition = d->cd->q->cpu_queue.size/HeTM_shared_data.nbCPUThreads;
  int queue_pos = HeTM_thread_data[0]->id * partition;

  // INIT stream
	int n = 0, count = 0;

  // TODO: when everyone is done
  // --> go for the shared queue
  //     ---> only the GPU is going for the shared one!

	/* Fetch values from the queue */
	if (queue_current >= partition || queue_current == 0) {
		//queue emptied, fetch new values
    // TODO: readQueue is broken!
		// queue_pointer = readQueue(&d->seed, d->cd->q, queue_pos, 0);
		queue_pointer = &(d->cd->q->cpu_queue.values[queue_pos]);
		queue_current = 0;
	}
	accounts_vec[0] = queue_pointer[queue_current];
	queue_current++;

	nb = (int)(RAND_R_FNC(d->seed) % 100);
	if (nb >= d->set_percent) {
		/* Read (GET) */
		search_entry(d->memcd, accounts_vec); // TODO: do something with the response
		d->nb_read_all++;
	} else {
		/* Write (SET) */
		new_entry(d->memcd, accounts_vec); // TODO: add entry
		d->nb_write_all++;
	}
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

static void test_cuda(int id, void *data)
{
  thread_data_t    *d = &((thread_data_t *)data)[id];
  cuda_t          *cd = d->cd;
  account_t *accounts = d->memcd->accounts;

  cd->clock++;

  jobWithCuda_runMemcd(d, cd, accounts, global_ts);

  global_ts++; // TODO
}

static void afterGPU(int id, void *data)
{
  thread_data_t *d = &((thread_data_t *)data)[id];
  d->nb_transfer               = HeTM_stats_data.nbCommittedTxsGPU;
  d->nb_transfer_gpu_only      = HeTM_stats_data.nbTxsGPU;
  d->nb_aborts                 = HeTM_stats_data.nbAbortsGPU;
  d->nb_aborts_1               = HeTM_stats_data.timeGPU; /* duration; */ // TODO
  d->nb_aborts_2               = 0; /* duration_cmp; */ // TODO
  d->nb_aborts_locked_read     = HeTM_stats_data.nbBatches; // TODO: these names are really bad
  d->nb_aborts_locked_write    = HeTM_stats_data.nbBatchesSuccess; /*counter_sync;*/ // TODO: successes?
  d->nb_aborts_validate_read   = 0; /* trf_2cpu; */ // TODO
  d->nb_aborts_validate_write  = 0; /* trf_2gpu; */ // TODO
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

  // TODO:
  parsedData.nb_threadsCPU = HeTM_shared_data.nbCPUThreads;
  parsedData.nb_threads    = HeTM_shared_data.nbThreads;

  malloc_or_die(data, parsedData.nb_threads + 1);
  memset(data, 0, (parsedData.nb_threads + 1)*sizeof(thread_data_t)); // safer
  parsedData.dthreads     = data;
  // ##########################################

  jobWithCuda_exit(NULL); // Reset Cuda Device

  bank_check_params(&parsedData);

  malloc_or_die(threads, parsedData.nb_threads);

  malloc_or_die(memcd, 1);
  long entries = parsedData.nb_accounts*parsedData.num_ways;
  long totalMemory = entries*(sizeof(account_t) + sizeof(int));
  char *memPoolPtr; // TODO: move this to the HeTM library

  HeTM_mempool_init(totalMemory);
  HeTM_alloc((void**)&memPoolPtr, NULL, totalMemory);

  memcd->accounts = (account_t*)memPoolPtr;
  memcd->version = (int*)(memPoolPtr + entries*sizeof(account_t));

  memcd->version_bm_size = 1 + (entries >> BM_HASH);
  malloc_or_die(memcd->version_bm, memcd->version_bm_size);
  printf("memcd->version_bm_size: %i\n", memcd->version_bm_size);

  memcd->size = parsedData.nb_accounts;
  memcd->ways = parsedData.num_ways;
  memset(memcd->accounts, 0, entries*sizeof(account_t));
  memset(memcd->version_bm, 0, memcd->version_bm_size*sizeof(int));
  // printf("Total before   : %d\n", total(memcd, 0)); // TODO
  parsedData.memcd = memcd;

  DEBUG_PRINT("Initializing GPU.\n");

  cuda_t *cuda_st;
  cuda_st = jobWithCuda_init(memcd->accounts, parsedData.nb_threadsCPU,
    memcd->size, parsedData.trans, 0, parsedData.GPUthreadNum, parsedData.GPUblockNum,
    parsedData.hprob, parsedData.hmult);

  cuda_st->threadNum = parsedData.GPUthreadNum;
  cuda_st->blockNum  = parsedData.GPUblockNum;

  jobWithCuda_initMemcd(cuda_st, parsedData.num_ways,
    parsedData.set_percent, parsedData.shared_percent);
  memcd->ways = parsedData.num_ways;
  cuda_st->version = memcd->version; // TODO: move this!!!

  parsedData.cd = cuda_st;
  //DEBUG_PRINT("Base: %lu %lu \n", memcd->accounts, &memcd->accounts);
  if (cuda_st == NULL) {
    printf("CUDA init failed.\n");
    exit(-1);
  }

  /* Init STM */
  printf("Initializing STM\n");

  TM_INIT(parsedData.nb_threads);
  // ############################################################################
  // ### Start iterations #######################################################
  // ############################################################################
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
    memset(memcd->accounts, 0, memcd->size * sizeof(account_t));

    TIMER_READ(parsedData.last);
    bank_between_iter(&parsedData, j);
    bank_printStats(&parsedData);
  }

  /* Cleanup STM */
  TM_EXIT();
  /*Cleanup GPU*/
  jobWithCuda_exit(cuda_st);
  free(cuda_st);
  // ### End iterations #########################################################

  bank_statsFile(&parsedData);

  /* Delete memcd and accounts */
  HeTM_mempool_destroy();
  free(memcd);

  free(threads);
  free(data);

  return EXIT_SUCCESS;
}
