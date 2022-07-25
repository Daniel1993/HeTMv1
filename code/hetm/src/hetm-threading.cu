#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

#include "hetm-log.h"

#include "pr-stm-wrapper.cuh" // TODO: check if linkage breaks
#include "knlman.hpp"
#include "hetm.cuh"
#include "hetm-cmp-kernels.cuh"
#include "hetm-producer-consumer.h"

#ifndef SYS_CPU_NB_CORES
#define SYS_CPU_NB_CORES 56
#endif

int HeTM_run_sync = 0;

// -------------------- // TODO: organize them
// Functions
static int pinThread(int tid);
static void* threadWait(void* argPtr);
static void* offloadThread(void*);
// static void* offloadThread_GPU(void *args);
static void emptyRequest(void*);
// --------------------

// --------------------
// Variables
HeTM_thread_s HeTM_thread_data_all[HETM_NB_DEVICES][HETM_MAX_CPU_THREADS];
__thread HeTM_thread_s *HeTM_thread_data[HETM_NB_DEVICES];
static barrier_t wait_callback;
static int isCreated = 0;
static HeTM_map_tid_to_core threadMapFunction = pinThread;
// --------------------

int HeTM_set_thread_mapping_fn(HeTM_map_tid_to_core fn)
{
  threadMapFunction = fn;
  return 0;
}

int HeTM_start(HeTM_callback CPUclbk, HeTM_callback GPUclbk, void *args)
{
  HeTM_set_is_stop(0);
  // Inits threading sync barrier
  if (!HeTM_gshared_data.isCPUEnabled) {
    barrier_init(wait_callback, 2); // GPU on
  } else if (!HeTM_gshared_data.isGPUEnabled) {
    barrier_init(wait_callback, HeTM_gshared_data.nbCPUThreads+1); // CPU on
  } else {
    barrier_init(wait_callback, HeTM_gshared_data.nbCPUThreads+2); // both on
  }

  int i;
  for (int j = 0; j < HETM_NB_DEVICES; j++) {
    for (i = 0; i < HeTM_gshared_data.nbThreads; i++) {
      if(i == HeTM_gshared_data.nbCPUThreads) {
        // last thread is the GPU
        HeTM_shared_data[j].threadsInfo[i].callback = GPUclbk;
      } else {
        HeTM_shared_data[j].threadsInfo[i].callback = CPUclbk;
      }
      HeTM_shared_data[j].threadsInfo[i].id = i;
      HeTM_shared_data[j].threadsInfo[i].devId = j;
      HeTM_shared_data[j].threadsInfo[i].args = args;
      __sync_synchronize(); // makes sure threads see the new callback
    }
  }
  for (i = 0; i < HeTM_gshared_data.nbThreads; i++) {
    if (!isCreated) {
      thread_create_or_die(&HeTM_shared_data[0].threadsInfo[i].thread,
        NULL, threadWait, &HeTM_shared_data[0].threadsInfo[i]);
    }
  }

  if (!isCreated) {
    thread_create_or_die(&HeTM_shared_data[0].asyncThread,
      NULL, offloadThread, NULL);
  }

  HETM_DEB_THREADING("Signal threads to start");
  barrier_cross(wait_callback); // all start sync
  isCreated = 1;
  return 0;
}

int HeTM_join_CPU_threads()
{
  HeTM_set_is_stop(1);
  HeTM_async_set_is_stop(1);
  // WARNING: HeTM_set_is_stop(1) must be called before this point
  for (int j = 0; j < HETM_NB_DEVICES; ++j) HeTM_flush_barrier(j);
  PR_SET_IS_DONE(1); // unblocks who is wanting for PR-STM
  barrier_cross(wait_callback);

  int i;
  // int tmp = HeTM_async_is_stop(0);
  for (i = 0; i < HeTM_gshared_data.nbThreads; i++) {
    HETM_DEB_THREADING("Joining with thread %i ...", i);
    for (int j = 0; j < HETM_NB_DEVICES; ++j) HeTM_flush_barrier(j);
    thread_join_or_die(HeTM_shared_data[0].threadsInfo[i].thread, NULL);
  }
  barrier_destroy(wait_callback);
  HETM_DEB_THREADING("Joining with offload thread ...");
  RUN_ASYNC(emptyRequest, NULL);
  for (int j = 0; j < HETM_NB_DEVICES; ++j) HeTM_flush_barrier(j);
  thread_join_or_die(HeTM_shared_data[0].asyncThread, NULL);
  // HeTM_async_set_is_stop(tmp);
  isCreated = 0;
  return 0;
}

static int pinThread(int tid)
{
  int coreId = tid;

  // TODO: this is hardware dependent
  // intel14
  if (tid >= 14 && tid < 28)
    coreId += 14;
  if (tid >= 28 && tid < 42)
    coreId -= 14;

  coreId = tid % SYS_CPU_NB_CORES;
  return coreId;
}

static void emptyRequest(void*) { }

// Producer-Consumer thread: consumes requests from the other ones
static void* offloadThread(void*)
{
  HeTM_async_req_s *req;

  cpu_set_t my_set;
  CPU_ZERO(&my_set);
  int core = threadMapFunction(55); // last thread
  CPU_SET(core, &my_set);
  sched_setaffinity(0, sizeof(cpu_set_t), &my_set);

  // TODO: only waiting GPU 0 to stop
  while (!HeTM_async_is_stop() || !hetm_pc_is_empty(HeTM_offload_pc)) {
    hetm_pc_consume(HeTM_offload_pc, (void**)&req);
    req->fn(req->args);
    HeTM_free_async_request(req);
  }
  HETM_DEB_THREADING("Offload thread exit");
  return NULL;
}

// // Producer-Consumer thread: consumes requests from the other ones
// static void* offloadThread_GPU(void *args)
// {
//   HeTM_async_req_s *req;
//   offload_gpu_s *a = (offload_gpu_s*)args;

//   cpu_set_t my_set;
//   CPU_ZERO(&my_set);
//   int core = threadMapFunction(54 - a->GPUid); // last thread
//   CPU_SET(core, &my_set);
//   sched_setaffinity(0, sizeof(cpu_set_t), &my_set);

//   // TODO: only waiting GPU 0 to stop
//   while (1) {
//     while (!__atomic_load_n(a->hasWork, __ATOMIC_ACQUIRE)) // wait
//     {
//       if (HeTM_async_is_stop(0)) break;
//     }
//     if (HeTM_async_is_stop(0)) goto exit;
//     (*(a->fn))(*(a->args));
//     __atomic_store_n(a->hasWork, 0, __ATOMIC_RELEASE);
//   }
// exit:
//   __atomic_store_n(a->hasWork, 0, __ATOMIC_ACQUIRE);
//   HETM_DEB_THREADING("Offload thread exit");
//   free(args);
//   return NULL;
// }

static void* threadWait(void *argPtr)
{
  HeTM_thread_s *args = (HeTM_thread_s*)argPtr;
  int threadId = args->id;
  int idGPU = HeTM_gshared_data.nbCPUThreads; // GPU is last
  HeTM_callback callback = NULL;
  int core = -1;
  cpu_set_t my_set;

  CPU_ZERO(&my_set);
  for (int j = 0; j < HETM_NB_DEVICES; ++j) {
    // malloc_or_die(HeTM_thread_data[j], 1);
    // memcpy(HeTM_thread_data[j], args, sizeof(HeTM_thread_s));
    HeTM_thread_data[j] = &(HeTM_shared_data[j].threadsInfo[threadId]);
    HeTM_thread_data[j]->devId = j;
  }
  // TODO: each thread runs on one core only (more complex schedules not allowed)
  if (threadId != idGPU) { // only pin STM threads
    core = threadMapFunction(threadId);
    CPU_SET(core, &my_set);
    sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
  }

  if (threadId == idGPU) {
    core = threadMapFunction(54); // last thread
    CPU_SET(core, &my_set);
    sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
  }

  HETM_DEB_THREADING("Thread %i started on core %i", threadId, core);
  HETM_DEB_THREADING("Thread %i wait start barrier", threadId);
  barrier_cross(wait_callback); /* wait for start parallel */
	while (1) {
		while (callback == NULL) {
      // COMPILER_FENCE();
      __sync_synchronize();
      callback = args->callback;
      if (HeTM_is_stop()) break;
      // pthread_yield();
    }
    if (HeTM_is_stop()) break;

    // Runs the corresponding thread type (worker or controller)
    if (threadId == idGPU) {
      HeTM_gpu_thread();
    } else {
      HeTM_cpu_thread();
    }
    // callback(threadId, args->args);
    args->callback = callback = NULL;
	}
  HETM_DEB_THREADING("Thread %i wait join barrier", threadId);
  barrier_cross(wait_callback); /* join threads also call the barrier */
  HETM_DEB_THREADING("Thread %i exit", threadId);

  return NULL;
}
