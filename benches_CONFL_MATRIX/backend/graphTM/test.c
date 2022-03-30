#include "stm.h"

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#define NB_THREADS    16
#define NB_INCS       10

extern unsigned long long g_stm_StatNbWrites;
extern unsigned long long g_stm_StatNbReads;
extern unsigned long long g_stm_StatNbCommittedWrites;
extern unsigned long long g_stm_StatNbCommittedReads;
extern unsigned long long g_stm_StatNbCommits;
extern unsigned long long g_stm_StatNbAborts;

static int counter = 0;
static unsigned long long array[NB_THREADS];

static void* test1(void *arg)
{
  STM_THREAD_T *STM_SELF = STM_NEW_THREAD();
  STM_INIT_THREAD(STM_SELF, (uintptr_t)arg);
  
  for (int i = 0; i < NB_INCS; ++i) { 
    STM_BEGIN_WR();
    STM_WRITE(counter, STM_READ(counter) + 1);
    STM_END();
  }

  STM_FREE_THREAD(STM_SELF);
  return NULL;
}

static void* test2(void *arg)
{
  STM_THREAD_T *STM_SELF = STM_NEW_THREAD();
  STM_INIT_THREAD(STM_SELF, (uintptr_t)arg);
  
  for (int i = 0; i < NB_INCS; ++i) { 
    int count = 0;
    STM_BEGIN_WR();
    count += STM_READ(array[((uintptr_t)arg+NB_THREADS)%NB_THREADS]);
    STM_WRITE(array[(uintptr_t)arg], count + 1);
    STM_END();
  }

  STM_FREE_THREAD(STM_SELF);
  return NULL;
}

static void printStats()
{
  extern unsigned long long g_stm_StatNbReads;
  extern unsigned long long g_stm_StatNbCommittedWrites;
  extern unsigned long long g_stm_StatNbCommittedReads;
  extern unsigned long long g_stm_StatNbCommits;
  extern unsigned long long g_stm_StatNbAborts;

  printf("------------------ Statistics:\n");
  printf("           g_stm_StatNbWrites: %9llu\n", g_stm_StatNbWrites);
  printf("            g_stm_StatNbReads: %9llu\n", g_stm_StatNbReads);
  printf("  g_stm_StatNbCommittedWrites: %9llu\n", g_stm_StatNbCommittedWrites);
  printf("   g_stm_StatNbCommittedReads: %9llu\n", g_stm_StatNbCommittedReads);
  printf("          g_stm_StatNbCommits: %9llu\n", g_stm_StatNbCommits);
  printf("           g_stm_StatNbAborts: %9llu\n", g_stm_StatNbAborts);
}

int main()
{
  pthread_t threads[NB_THREADS];

  STM_STARTUP();
  STM_NEW_THREADS(NB_THREADS);

  // launch NB_THREADS threads each inc the counter NB_INCS times
  for (uintptr_t i = 1; i < NB_THREADS; ++i) {
    pthread_create(&(threads[i]), NULL, test1, (void*)i);
  }
  test1(0);
  for (uintptr_t i = 1; i < NB_THREADS; ++i) {
    pthread_join(threads[i], NULL);
  }

  printf("counter=%i (expected %i)\n", counter, NB_THREADS*NB_INCS);
  printStats();
  STM_SHUTDOWN();


  STM_STARTUP();
  STM_NEW_THREADS(NB_THREADS);

  // launch NB_THREADS threads each inc the counter NB_INCS times
  for (uintptr_t i = 1; i < NB_THREADS; ++i) {
    pthread_create(&(threads[i]), NULL, test2, (void*)i);
  }
  test2(0);
  for (uintptr_t i = 1; i < NB_THREADS; ++i) {
    pthread_join(threads[i], NULL);
  }

  printf("array: ");
  for (int i = 0; i < NB_THREADS; ++i) {
    printf("[%i]=%lli ", i, array[i]);
  }
  printf("\n");
  printStats();
  STM_SHUTDOWN();

  return 0;
}
