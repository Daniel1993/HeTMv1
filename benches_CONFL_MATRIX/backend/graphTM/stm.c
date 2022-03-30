#define _GNU_SOURCE
#include "stm.h"

#include <pthread.h>
#include <stdlib.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// #include "../../bench/lib/thread.h"

// TODO: double check if you want reads or writes in the rows
#define MAT_POS(writeTX, readTX) g_conflMatrix[readTX*g_conflMatrixSize + writeTX]
#define NB_ITERATION_BEFORE_PRINT 1000

// TODO: check thread.c barrier --> static int handleMatrixLine(int isRestart);
static void writeBack();

static int g_isExit = 0;
unsigned long long g_nbEnterThreads;
unsigned long long g_nbExitThreads;
static THREAD_BARRIER_T *g_barrier;
// static ticket_barrier_t g_barrier;
static pthread_mutex_t g_lock;
static unsigned char *g_conflMatrix;
static unsigned int g_conflMatrixSize;
static unsigned long long g_iteCount;
static int g_firstGraph;

// statistics
unsigned long long g_stm_StatNbWrites;
unsigned long long g_stm_StatNbReads;
unsigned long long g_stm_StatNbCommittedWrites;
unsigned long long g_stm_StatNbCommittedReads;
unsigned long long g_stm_StatNbCommits;
unsigned long long g_stm_StatNbAborts;

static graphTM g_state;
static graphTM_thr *g_thr_state;
// static __thread int g_tid;

// statistics
static __thread int g_thr_stats_written;
static __thread unsigned long long g_thr_StatNbWritesInTX;
static __thread unsigned long long g_thr_StatNbReadsInTX;
static __thread unsigned long long g_thr_StatNbWrites;
static __thread unsigned long long g_thr_StatNbReads;
static __thread unsigned long long g_thr_StatNbCommittedWrites;
static __thread unsigned long long g_thr_StatNbCommittedReads;
static __thread unsigned long long g_thr_StatNbCommits;
static __thread unsigned long long g_thr_StatNbAborts;

static graphTM_rwset_entry *allocRWSetEntries(graphTM_rwset *rwset)
{
  if (!rwset->d) {
    rwset->d = (graphTM_rwset_entry*)malloc(sizeof(graphTM_rwset_entry)*GRAPH_TM_INIT_SIZE);
    rwset->size = 0;
    rwset->allocSize = GRAPH_TM_INIT_SIZE;
  } else { // needs more space
    rwset->allocSize <<= 1;
    rwset->d = (graphTM_rwset_entry*)realloc(rwset->d, sizeof(graphTM_rwset_entry)*rwset->allocSize);
  }
  return rwset->d;
}

static graphTM_rwset_entry *findInRWSet(graphTM_rwset *rwset, void *addr)
{
  graphTM_rwset_entry *ptr = &(rwset->d[0]);
  graphTM_rwset_entry *end = &(rwset->d[rwset->size]);
  while (ptr != end) {
    if (ptr->addr == addr) {
      return ptr;
    }
    ptr++;
  }
  return NULL;
}

static graphTM_rwset_entry *insertInRWSet(graphTM_rwset *rwset, void *addr, graphTM_typesSize value)
{
  if (rwset->size >= rwset->allocSize) allocRWSetEntries(rwset);
  graphTM_rwset_entry *end = &(rwset->d[rwset->size]);
  rwset->size++;
  end->addr = addr;
  end->v = value;
  return end;
}

void stm_startup()
{
  g_nbEnterThreads = 0;
  g_nbExitThreads = 0;
  g_iteCount = 0;
  pthread_mutex_init(&g_lock, NULL);
  /* empty. See stm_new_threads(...) */
}

void stm_new_threads(int numThread)
{
  g_barrier = THREAD_BARRIER_ALLOC(numThread);
  THREAD_BARRIER_INIT(g_barrier, numThread);
  // ticket_barrier_init(&g_barrier, numThread);

  g_thr_state = (graphTM_thr*)malloc(sizeof(graphTM_thr)*numThread);
  g_state.nbThreads = numThread;
  g_conflMatrixSize = numThread;
  g_conflMatrix = malloc(numThread*numThread*sizeof(unsigned char));
  g_isExit = 0;

  // read set
  g_state.rs_entries = (graphTM_rwset*)malloc(sizeof(graphTM_rwset)*numThread);
  // write set
  g_state.ws_entries = (graphTM_rwset*)malloc(sizeof(graphTM_rwset)*numThread);

  for (int i = 0; i < numThread; ++i) {
    g_state.rs_entries[i].d = NULL;
    g_state.ws_entries[i].d = NULL;
    allocRWSetEntries(&(g_state.rs_entries[i]));
    allocRWSetEntries(&(g_state.ws_entries[i]));
  }
}

void stm_shutdown()
{
  pthread_mutex_destroy(&g_lock);
  ticket_barrier_destroy(&g_barrier);
  // THREAD_BARRIER_FREE(g_barrier);

  g_stm_StatNbReads = 0;
  g_stm_StatNbCommittedWrites = 0;
  g_stm_StatNbCommittedReads = 0;
  g_stm_StatNbCommits = 0;
  g_stm_StatNbAborts = 0;

  for (int i = 0; i < g_state.nbThreads; ++i) {
    free(g_state.rs_entries[i].d);
    free(g_state.ws_entries[i].d);
  }
  free(g_state.rs_entries);
  free(g_state.ws_entries);
  free(g_conflMatrix);
}

STM_THREAD_T *stm_new_thread()
{
  __sync_fetch_and_add(&g_nbEnterThreads, 1);
  g_thr_stats_written = 0;
  int tid = thread_get_task_id();
  
  g_thr_state[tid].tid = tid;
  g_thr_state[tid].rs_entries = &(g_state.rs_entries[tid]);
  g_thr_state[tid].ws_entries = &(g_state.ws_entries[tid]);
  // same as get
  return stm_get_thread();
}

STM_THREAD_T *stm_get_thread()
{
  int tid = thread_get_task_id();
  return &g_thr_state[tid];
}

void stm_set_self(STM_THREAD_T *thr)
{
  /* does nothing */
}

void stm_init_thread(STM_THREAD_T *thr, int tid)
{
  // replaces the tid --> does not work tid is per task
}

void stm_free_thread(STM_THREAD_T *thr)
{
  // NOTE: must be called before exiting the thread
  // checks if other threads are running and possibly doing transactions
  if (!g_thr_stats_written) {
    g_thr_stats_written = 1;
    __sync_add_and_fetch(&g_stm_StatNbWrites, g_thr_StatNbWrites);
    __sync_add_and_fetch(&g_stm_StatNbReads, g_thr_StatNbReads);
    __sync_add_and_fetch(&g_stm_StatNbCommittedWrites, g_thr_StatNbCommittedWrites);
    __sync_add_and_fetch(&g_stm_StatNbCommittedReads, g_thr_StatNbCommittedReads);
    __sync_add_and_fetch(&g_stm_StatNbCommits, g_thr_StatNbCommits);
    __sync_add_and_fetch(&g_stm_StatNbAborts, g_thr_StatNbAborts);
    __sync_fetch_and_add(&g_nbExitThreads, 1);
  }
  do {
    // printf("THR%i waits exit (wsSize = %i)\n", g_thr_state.tid, g_thr_state.ws_entries->size);
    // ticket_barrier_cross(&g_barrier);
    handleMatrixLine(1);
    thread_yield();
  } while (__atomic_load_n(&g_nbExitThreads, __ATOMIC_ACQUIRE) != __atomic_load_n(&g_nbEnterThreads, __ATOMIC_ACQUIRE));
  __atomic_store_n(&g_isExit, 1, __ATOMIC_RELEASE); // also unblocks anyone in the barrier
}

void stm_begin_wr()
{
  // printf("THR%i stm_begin_wr\n", g_thr_state.tid);
  int tid = thread_get_task_id();
  g_thr_state[tid].rs_entries->size = 0; 
  g_thr_state[tid].ws_entries->size = 0; 
  g_thr_state[tid].startedAsRO = 0; 
  g_thr_state[tid].isRO = 0; 
  g_thr_StatNbWritesInTX = 0;
  g_thr_StatNbReadsInTX = 0;
  // pthread_mutex_lock(&g_lock);
  // printf("[%i] lock\n", g_tid);
}

void stm_begin_rd()
{
  int tid = thread_get_task_id();
  g_thr_state[tid].rs_entries->size = 0; 
  g_thr_state[tid].ws_entries->size = 0; 
  g_thr_state[tid].startedAsRO = 0; 
  g_thr_state[tid].isRO = 1; 
  g_thr_StatNbWritesInTX = 0;
  g_thr_StatNbReadsInTX = 0;
  // pthread_mutex_lock(&g_lock);
  // printf("[%i] lock\n", g_tid);
}

int stm_end()
{
  int abort = 0;
  int ret = 1; // success
  int tid = thread_get_task_id();
  // printf("THR%i commits\n", g_thr_state.tid);

  THREAD_BARRIER(g_barrier, thread_get_task_id());
  // writeBack(); // all transactions commit --> single global lock
  // pthread_mutex_unlock(&g_lock);
  // printf("[%i] unlock\n", g_tid);
  // ticket_barrier_cross(&g_barrier);
  abort = handleMatrixLine(0);

  if (!abort) {
    // TODO: I think we do not need the lock after knowing the conflicts
    pthread_mutex_lock(&g_lock);
  printf("[%i] writeBack()\n", g_tid);
    writeBack();
    pthread_mutex_unlock(&g_lock);
    g_thr_StatNbCommittedWrites += g_thr_StatNbWritesInTX;
    g_thr_StatNbCommittedReads += g_thr_StatNbReadsInTX;
    g_thr_StatNbCommits++;
    // printf("    THR%i commit: SUCCESS\n", g_thr_state.tid);
  } else {
    g_thr_StatNbAborts++;
    ret = 0; // failure
    // printf("    THR%i commit: FAILURE\n", g_thr_state.tid);
  }
  
  // aborted TXs still have to wait before accessing the shared state
  g_thr_state[tid].rs_entries->size = 0;
  g_thr_state[tid].ws_entries->size = 0;
  // ticket_barrier_cross(&g_barrier);
  THREAD_BARRIER(g_barrier, thread_get_task_id());
  return ret;
}

void stm_restart()
{
  int tid = thread_get_task_id();
  // printf("THR%i stm_restart\n", g_thr_state.tid);
  // needs to handle commit logic, if not restarting transaction may enter a live-lock
  g_thr_state[tid].rs_entries->size = 0;
  g_thr_state[tid].ws_entries->size = 0;
  g_thr_StatNbAborts++;
  // ticket_barrier_cross(&g_barrier);
  THREAD_BARRIER(g_barrier, thread_get_task_id());
  handleMatrixLine(1);
}

void *stm_malloc(long unsigned size)
{
  int tid = thread_get_task_id();
  void *res = malloc((size_t)size); // need to mark as written
  g_thr_StatNbWritesInTX++;
  g_thr_StatNbWrites++;
  insertInRWSet(g_thr_state[tid].ws_entries, res, (graphTM_typesSize)NULL)->code = STM_PTR_CODE;
  return res;
}

void stm_free(void *addr)
{
  int tid = thread_get_task_id();
  // cannot free in transaction --> write back may catch a free region
  // there are some writes that may be avoided ...
  insertInRWSet(g_thr_state[tid].ws_entries, addr, (graphTM_typesSize)addr)->code = STM_FRE_CODE;
}

#define STM_READ_AUX(addr, _code) ({ \
  int tid = thread_get_task_id(); \
  graphTM_rwset_entry *res = findInRWSet(g_thr_state[tid].ws_entries, addr); \
  graphTM_typesSize retVal = ((res == NULL) ? (graphTM_typesSize)*(addr) : res->v); \
  if (res == NULL) { \
    if (findInRWSet(g_thr_state[tid].rs_entries, addr) == NULL) { \
      insertInRWSet(g_thr_state[tid].rs_entries, addr, (graphTM_typesSize)retVal)->code = _code; \
    } \
  } \
  retVal; \
})

STM_INT stm_read(STM_INT *addr)
{
  g_thr_StatNbReadsInTX++;
  g_thr_StatNbReads++;
  graphTM_typesSize ret = STM_READ_AUX(addr, STM_INT_CODE);
  return ret.valInt;
}

STM_FLT stm_read_F(STM_FLT *addr)
{
  g_thr_StatNbReadsInTX++;
  g_thr_StatNbReads++;
  graphTM_typesSize ret = STM_READ_AUX(addr, STM_FLT_CODE);
  return ret.valFloat;
}

STM_PTR stm_read_P(STM_PTR *addr)
{
  g_thr_StatNbReadsInTX++;
  g_thr_StatNbReads++;
  graphTM_typesSize ret = STM_READ_AUX(addr, STM_PTR_CODE);
  return ret.valPtr;
}

// no blind writes
#define STM_WRITE_AUX(addr, val, _code) ({ \
  int tid = thread_get_task_id(); \
  graphTM_rwset_entry *res = findInRWSet(g_thr_state[tid].ws_entries, addr); \
  if (res == NULL) { \
    STM_READ_AUX(addr, _code); /* also (tries to) insert in read set */ \
    insertInRWSet(g_thr_state[tid].ws_entries, addr, (graphTM_typesSize)val)->code = _code; \
  } else { \
    res->v = (graphTM_typesSize)val; \
  } \
  val; \
})

STM_INT stm_write(STM_INT *addr, STM_INT val)
{
  g_thr_StatNbWritesInTX++;
  g_thr_StatNbWrites++;
  return STM_WRITE_AUX(addr, val, STM_INT_CODE);
}

STM_FLT stm_write_F(STM_FLT *addr, STM_FLT val)
{
  g_thr_StatNbWritesInTX++;
  g_thr_StatNbWrites++;
  return STM_WRITE_AUX(addr, val, STM_FLT_CODE);
}

STM_PTR stm_write_P(STM_PTR *addr, STM_PTR val)
{
  g_thr_StatNbWritesInTX++;
  g_thr_StatNbWrites++;
  // printf("  write ptr addr=%p val=%p\n", addr, val);
  return STM_WRITE_AUX(addr, val, STM_PTR_CODE);
}

// returns if it is abort
int handleMatrixLine(int isRestart)
{
  int tid = thread_get_task_id();
  int abort = isRestart;
  unsigned nbThreads = g_nbEnterThreads - g_nbExitThreads;
  nbThreads = nbThreads > 0 ? nbThreads : 1;
  unsigned ite = g_iteCount % nbThreads;
  FILE *fp;
  char matFilename[1024];
  int countEdges = 0;
  // memset(&(MAT_POS(g_tid, 0)), 0, g_conflMatrixSize*sizeof(unsigned char)); // each threads erases its own line
  if (!isRestart) {
    /* construct conflic matrix in parallel */
    // if some some thread with smaller tid conflicts with me --> abort
    // TODO: some threads can be saved

    graphTM_rwset_entry *rPtr = &(g_thr_state[tid].rs_entries->d[0]);
    graphTM_rwset_entry *rEnd = &(g_thr_state[tid].rs_entries->d[g_thr_state[tid].rs_entries->size]);
    while (rPtr != rEnd) {
      for (int t = 0; t < g_state.nbThreads; ++t) {
        MAT_POS(g_tid, t) = 0;
        if (t == g_tid) continue; // conflict already detected
        // if (!abort) printf("   THR%i check against THR%i\n", g_tid, t);
        graphTM_rwset_entry *wPtr = &(g_state.ws_entries[t].d[0]);
        graphTM_rwset_entry *wEnd = &(g_state.ws_entries[t].d[g_state.ws_entries[t].size]);
        while (wPtr != wEnd) {
          if (rPtr->addr == wPtr->addr) {
            MAT_POS(g_tid, t) = 1;
            if (((t + ite) % nbThreads) < ((g_tid + ite) % nbThreads)) {
              // if (!abort) printf("   THR%i aborts after check against THR%i writeset\n", g_tid, t);
              abort = 1; // higher prio thread wrote where I read/wrote
              // printf("ABORT detected (%i --> %i)!\n", t, g_tid);
            } 
            break;
          }
          wPtr++;
        }
      }
      rPtr++;
    }
  }
  if (g_nbEnterThreads > 1 && g_tid > 0) {
    // ticket_barrier_cross(&g_barrier);
    THREAD_BARRIER(g_barrier, thread_get_task_id());
  }
  if (g_tid == 0) { // TODO: this may slow down the others
    countEdges = 0;
    sprintf(matFilename, GRAPH_TM_FILENAME, g_state.nbThreads, (int)(externalBarTotal/g_state.nbThreads), (int)++g_iteCount);
    for (int r = 0; r < g_conflMatrixSize; ++r) {
      for (int w = 0; w < g_conflMatrixSize; ++w) {
        if (r != w && MAT_POS(r, w)) {
          countEdges++;
        }
      }
    }
    if (g_firstGraph == 0 || countEdges > g_conflMatrixSize ||
        (g_iteCount % NB_ITERATION_BEFORE_PRINT) == (NB_ITERATION_BEFORE_PRINT-1)) {
      // printf("g_conflMatrixSize = %i countEdges = %i file = %s\n", g_conflMatrixSize, countEdges, matFilename);
      if (countEdges > 0) {
        fp = fopen(matFilename, "w");
        if (fp) {
          g_firstGraph++;
          fprintf(fp, "%i %i\n", g_conflMatrixSize, countEdges);
          // TODO: this is not efficient --> maybe atomic inc?
          for (int r = 0; r < g_conflMatrixSize; ++r) {
            for (int w = 0; w < g_conflMatrixSize; ++w) {
              if (r != w && MAT_POS(r, w)) {
                fprintf(fp, "%i %i\n", r, w);
              }
            }
          }
          fclose(fp);
        }
      }
    }
    // ticket_barrier_cross(&g_barrier);
    THREAD_BARRIER(g_barrier, thread_get_task_id());
  }
  return abort;
}

static void writeBack()
{
  int tid = thread_get_task_id();
  graphTM_rwset_entry *wPtr = &(g_thr_state[tid].ws_entries->d[0]);
  graphTM_rwset_entry *wEnd = &(g_thr_state[tid].ws_entries->d[g_thr_state[tid].ws_entries->size]);
  while (wPtr != wEnd) {
    switch (wPtr->code)
    {
    case STM_FRE_CODE:
      printf("    write-back ptr in addr=%p free'd\n", wPtr->v.valPtr);
      free(wPtr->v.valPtr);
      break;
    case STM_PTR_CODE:
      printf("    write-back ptr in addr=%p with val=%p\n", wPtr->addr, wPtr->v.valPtr);
      *((STM_PTR*)wPtr->addr) = wPtr->v.valPtr;
      break;
    case STM_INT_CODE:
      printf("    write-back int in addr=%p with val=%li\n", wPtr->addr, wPtr->v.valInt);
      *((STM_INT*)wPtr->addr) = wPtr->v.valInt;
      break;
    case STM_FLT_CODE:
      printf("    write-back float in addr=%p with val=%f\n", wPtr->addr, wPtr->v.valFloat);
      *((STM_FLT*)wPtr->addr) = wPtr->v.valFloat;
      break;
    case STM_NON_CODE:
    default:
      *((graphTM_typesSize*)wPtr->addr) = (graphTM_typesSize)wPtr->v;
      break;
    }
    wPtr++;
  }
}
