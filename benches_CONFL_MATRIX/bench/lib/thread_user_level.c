/* =============================================================================
 *
 * thread.c
 *
 * =============================================================================
 *
 * Copyright (C) Stanford University, 2006.  All Rights Reserved.
 * Author: Chi Cao Minh
 *
 * =============================================================================
 *
 * For the license of bayes/sort.h and bayes/sort.c, please see the header
 * of the files.
 * 
 * ------------------------------------------------------------------------
 * 
 * For the license of kmeans, please see kmeans/LICENSE.kmeans
 * 
 * ------------------------------------------------------------------------
 * 
 * For the license of ssca2, please see ssca2/COPYRIGHT
 * 
 * ------------------------------------------------------------------------
 * 
 * For the license of lib/mt19937ar.c and lib/mt19937ar.h, please see the
 * header of the files.
 * 
 * ------------------------------------------------------------------------
 * 
 * For the license of lib/rbtree.h and lib/rbtree.c, please see
 * lib/LEGALNOTICE.rbtree and lib/LICENSE.rbtree
 * 
 * ------------------------------------------------------------------------
 * 
 * Unless otherwise noted, the following license applies to STAMP files:
 * 
 * Copyright (c) 2007, Stanford University
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 * 
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the
 *       distribution.
 * 
 *     * Neither the name of Stanford University nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY STANFORD UNIVERSITY ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL STANFORD UNIVERSITY BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 *
 * =============================================================================
 */

#define _GNU_SOURCE
#include <sched.h>
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <setjmp.h>
#include <limits.h>
#include <err.h>
#include <unistd.h>
#include <sys/sysinfo.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <signal.h>
#include <string.h>
#include "thread.h"
#include "types.h"

#include <stdio.h>

#define STACK_SIZE 4194304 // 4MB

static THREAD_LOCAL_T    global_threadId;
static long              global_numThread       = 1;
static THREAD_BARRIER_T* global_barrierPtr      = NULL;
static long*             global_threadIds       = NULL;
static THREAD_ATTR_T     global_threadAttr;
static THREAD_T*         global_threads         = NULL;
static void            (*global_funcPtr)(void*) = NULL;
static void*             global_argPtr          = NULL;
static volatile bool_t   global_doShutdown      = FALSE;

typedef struct thr_info_ {
	void(*func)(void*);
	void *args;
	long id;
	sigjmp_buf ctx;
	void *stack;
	// void *stack_save; // when enqueing also save the stack
	size_t stack_size;
	intptr_t sp;
	intptr_t pc;
} thr_info_s;

#define JB_SP 6
#define JB_PC 7
/* A translation is required when using an address of a variable.
   Use this as a black box in your code. */
intptr_t translate_address(intptr_t addr)
{
	intptr_t ret;
	asm volatile("xor    %%fs:0x30,%0\n"
			"rol    $0x11,%0\n"
			: "=g" (ret)
			  : "0" (addr));
	return ret;
}

/* user level threads ---> issue there are thread_local data that no longer works */
#define INIT_YIELD_QUEUE_SIZE 8192
static thr_info_s *ctxThrs = NULL;
static __thread unsigned qSPtr = 0;
static __thread unsigned qEPtr = 0;
static __thread unsigned queueSize = 0;
static __thread int *qThrsReady = NULL;
static __thread int curTask = 0;
static __thread int curTaskInit = 0;
static __thread intptr_t saveSP = 0;
static int *thrsPerCore = NULL;
static void **thrArgs;

static int nprocs()
{
	// return 1;
  cpu_set_t cs;
  CPU_ZERO(&cs);
  sched_getaffinity(0, sizeof(cs), &cs);
  return CPU_COUNT(&cs);
}

static void assignToThisCore(int core_id)
{
	cpu_set_t  mask;
	CPU_ZERO(&mask);
	CPU_SET(core_id, &mask);
	sched_setaffinity(0, sizeof(mask), &mask);
}

static int qThrsReadyIsEmpty()
{
	return qEPtr == qSPtr;
}

static int
qThrsReadyEnqueue() { 
	int ptr = qEPtr;
	int res = 0;

	qEPtr = (qEPtr+1) & (queueSize-1);
	if (qSPtr == qEPtr) {
		qEPtr = (qEPtr+queueSize-1) & (queueSize-1);
		queueSize <<= 1;
		qEPtr = (qEPtr+1) & (queueSize-1);
		qThrsReady = (int*)realloc(qThrsReady, sizeof(int)*queueSize);
	}
	qThrsReady[ptr] = curTask;
	res = sigsetjmp(ctxThrs[curTask].ctx, 1);
	return res;
}

static int
qThrsReadyDequeue() {
	int ptr = qSPtr;
	if (!qThrsReadyIsEmpty()) {
		qSPtr = (qSPtr+1) & (queueSize-1);
		curTask = qThrsReady[ptr];
		siglongjmp(ctxThrs[curTask].ctx, 1);
	}
	return 0;
}

static void
thread_task_yield() {
	if (!qThrsReadyEnqueue()) {
		qThrsReadyDequeue();
	}
}

long thread_get_task_id()
{
	return ctxThrs[curTask].id;
}

static void
initUserLevelThreads(int nbCores, int numThread)
{
	int tasksPerCore = numThread / nbCores;
	int remThrs = numThread % nbCores;
	int i;
	
	thrsPerCore = (int*)malloc(sizeof(int)*nbCores);
	thrArgs = (void**)malloc(sizeof(void*)*nbCores);

	for (i = 0;       i < remThrs; ++i) { thrsPerCore[i] = tasksPerCore + 1; }
	for (i = remThrs; i < nbCores; ++i) { thrsPerCore[i] = tasksPerCore; }

	ctxThrs = (thr_info_s*)malloc(STACK_SIZE*sizeof(thr_info_s));
	for (i = 0; i < numThread; ++i) {
		//Stack
		ctxThrs[i].stack = malloc(STACK_SIZE);

		//Initializing stack pointer and stack size
		ctxThrs[i].stack_size = STACK_SIZE;
		ctxThrs[i].sp = (intptr_t)ctxThrs[i].stack + STACK_SIZE - sizeof(int);
	}
}


/* =============================================================================
 * threadWait
 * -- Synchronizes all threads to start/stop parallel section
 * =============================================================================
 */
static void
threadWait (void* argPtr)
{
	long threadId = *(long*)argPtr;
	long nbCores = nprocs();
	int i, j, tid, jmpDone = 0;
	intptr_t currSP;
	volatile int saveCurTask; // <-- also gives the current stack position

	THREAD_LOCAL_SET(global_threadId, (long)threadId);
	assignToThisCore(threadId % nbCores);

	queueSize = INIT_YIELD_QUEUE_SIZE;
	qSPtr = qEPtr = 0;
	qThrsReady = (int*)realloc(qThrsReady, sizeof(int)*queueSize);

	for (i = 0; i < thrsPerCore[threadId]; ++i) {
		tid = i + threadId*thrsPerCore[threadId == 0 ? 0 : threadId-1];
		curTask = tid;
		ctxThrs[tid].id = tid;
		ctxThrs[tid].func = global_funcPtr;
		ctxThrs[tid].args = global_argPtr;
		ctxThrs[tid].pc = (intptr_t)global_funcPtr;
		if (sigsetjmp(ctxThrs[tid].ctx, 1)) {
			if (!curTaskInit) break; // already init
			saveCurTask = curTask;
			tid = curTask;
			threadId = thread_getId();
			// enqueus all remaining tasks
			for (j = 0; j < thrsPerCore[threadId]-1; ++j) {
				int ptr = qEPtr;
				qEPtr = (qEPtr+1) & (queueSize-1);
				tid = j + threadId*thrsPerCore[threadId == 0 ? 0 : threadId-1];
				curTask = tid;
				qThrsReady[ptr] = curTask;
			}
			if (curTaskInit) {
				curTaskInit = 0;
				curTask = saveCurTask;
			}
			tid = curTask;
			jmpDone = 1;
			break;
		}
		saveSP = (ctxThrs[tid].ctx->__jmpbuf)[JB_SP]; // all tid's have the same SP
		(ctxThrs[tid].ctx->__jmpbuf)[JB_SP] = translate_address(ctxThrs[tid].sp);
		// (ctxThrs[tid].ctx->__jmpbuf)[JB_PC] = translate_address(ctxThrs[tid].pc);
		// sigemptyset(&ctxThrs[tid].ctx->__saved_mask);
	}
	// need jmp before start
	if (!jmpDone) {
		curTaskInit = 1;
		siglongjmp(ctxThrs[tid].ctx, 1);
	}
	threadId = thread_getId();
	while (1) {
		THREAD_BARRIER(global_barrierPtr, thread_get_task_id()); /* wait for start parallel */
		if (global_doShutdown) {
			break;
		}
		global_funcPtr(global_argPtr);
		THREAD_BARRIER(global_barrierPtr, thread_get_task_id()); /* wait for end parallel */
		if (threadId == 0) { // any task from thread 0
			break;
		}
	}
	// restores the previous stack
	tid = thread_get_task_id();
	if (!sigsetjmp(ctxThrs[tid].ctx, 1)) {
		currSP = (ctxThrs[tid].ctx->__jmpbuf)[JB_SP];
		(ctxThrs[tid].ctx->__jmpbuf)[JB_SP] = saveSP;
		saveSP = currSP;
		siglongjmp(ctxThrs[tid].ctx, 1);
	}
}


/* =============================================================================
 * thread_startup
 * -- Create pool of secondary threads
 * -- numThread is total number of threads (primary + secondaries)
 * =============================================================================
 */
void
thread_startup (long numThread)
{
	long i;
	int nbCores = nprocs();

	initUserLevelThreads(nbCores, numThread);

	global_numThread = numThread;
	global_doShutdown = FALSE;

	/* Set up barrier */
	assert(global_barrierPtr == NULL);
	global_barrierPtr = THREAD_BARRIER_ALLOC(numThread);
	assert(global_barrierPtr);
	THREAD_BARRIER_INIT(global_barrierPtr, numThread);

	/* Set up ids */
	THREAD_LOCAL_INIT(global_threadId);
	assert(global_threadIds == NULL);
	global_threadIds = (long*)malloc(nbCores/*numThread*/ * sizeof(long));
	assert(global_threadIds);
	for (i = 0; i < nbCores/*numThread*/; i++) {
		global_threadIds[i] = i;
	}

	/* Set up thread list */
	assert(global_threads == NULL);
	global_threads = (THREAD_T*)malloc(nbCores/*numThread*/ * sizeof(THREAD_T));
	assert(global_threads);

	/* Set up pool */
	THREAD_ATTR_INIT(global_threadAttr);
	for (i = 1; i < nbCores/*numThread*/; i++) {
		if (i >= numThread) break;
		THREAD_CREATE(global_threads[i],
					  global_threadAttr,
					  &threadWait,
					  &global_threadIds[i]);
	}

	/*
	 * Wait for primary thread to call thread_start
	 */
}


/* =============================================================================
 * thread_start
 * -- Make primary and secondary threads execute work
 * -- Should only be called by primary thread
 * -- funcPtr takes one arguments: argPtr
 * =============================================================================
 */
void
thread_start (void (*funcPtr)(void*), void* argPtr)
{
	global_funcPtr = funcPtr;
	global_argPtr = argPtr;

	long threadId = 0; /* primary */
	threadWait((void*)&threadId);
}


/* =============================================================================
 * thread_shutdown
 * -- Primary thread kills pool of secondary threads
 * =============================================================================
 */
void
thread_shutdown ()
{
	/* Make secondary threads exit wait() */
	global_doShutdown = TRUE;
	__atomic_store_n(&global_barrierPtr->isExit, 1, __ATOMIC_RELEASE);
	// THREAD_BARRIER(global_barrierPtr, 0);

	// long numThread = global_numThread;

	long i;
	for (i = 1; i < nprocs()/*numThread*/; i++) {
		THREAD_JOIN(global_threads[i]);
	}

	THREAD_BARRIER_FREE(global_barrierPtr);
	global_barrierPtr = NULL;

	free(global_threadIds);
	global_threadIds = NULL;

	free(global_threads);
	global_threads = NULL;

	global_numThread = 1;
}


/* =============================================================================
 * thread_barrier_alloc
 * =============================================================================
 */
thread_barrier_t*
thread_barrier_alloc (long numThread)
{
	unsigned count = numThread;
	thread_barrier_t* barrierPtr;

	assert(numThread > 0);
	// assert((numThread & (numThread - 1)) == 0); /* must be power of 2 */
	barrierPtr = (thread_barrier_t*)malloc(sizeof(thread_barrier_t));
	if (barrierPtr != NULL) {
		barrierPtr->total = count;
	}

	return barrierPtr;
}
// // old impl
// thread_barrier_t*
// thread_barrier_alloc (long numThread)
// {
// 	thread_barrier_t* barrierPtr;
// 	assert(numThread > 0);
// 	assert((numThread & (numThread - 1)) == 0); /* must be power of 2 */
// 	barrierPtr = (thread_barrier_t*)malloc(numThread * sizeof(thread_barrier_t));
// 	if (barrierPtr != NULL) {
// 		barrierPtr->numThread = numThread;
// 	}
// 	return barrierPtr;
// }


/* =============================================================================
 * thread_barrier_free
 * =============================================================================
 */
void
thread_barrier_free (thread_barrier_t* b)
{
	__atomic_store_n(&b->isExit, 1, __ATOMIC_RELEASE);

	free(b); // this may cause some crashes...
}
// // old impl
// void
// thread_barrier_free (thread_barrier_t* barrierPtr)
// {
// 	free(barrierPtr);
// }


/* =============================================================================
 * thread_barrier_init
 * =============================================================================
 */
void thread_barrier_init(thread_barrier_t *b)
{
	b->isExit = 0;
	b->count_in = 0;
	b->count_out = 0;
	b->count_next = -1;
}
// // old impl
// void
// thread_barrier_init (thread_barrier_t* barrierPtr)
// {
// 	long i;
// 	long numThread = barrierPtr->numThread;
// 	for (i = 0; i < numThread; i++) {
// 		barrierPtr[i].count = 0;
// 		THREAD_MUTEX_INIT(barrierPtr[i].countLock);
// 		THREAD_COND_INIT(barrierPtr[i].proceedCond);
// 		THREAD_COND_INIT(barrierPtr[i].proceedAllCond);
// 	}
// }


/* =============================================================================
 * thread_barrier
 * -- Simple logarithmic barrier
 * =============================================================================
 */
void
thread_barrier (thread_barrier_t* b, long threadId)
{
	unsigned oldBunch = __atomic_load_n(&b->count_next, __ATOMIC_ACQUIRE);
	unsigned nextBunch = oldBunch + b->total;
	volatile unsigned wait = __sync_fetch_and_add(&b->count_in, 1);
	volatile unsigned next;
	volatile unsigned long long temp;

//   __sync_add_and_fetch(&g_ticketBarTot, 1);
  // __sync_add_and_fetch(&g_ticketBarCount, 1);

  // if (nextBunch % b->total != 0) {
  //   // BUG: ...
  //   nextBunch = nextBunch + (b->total - (nextBunch % b->total));
  // }

	// int ret = 0;

	while (!__atomic_load_n(&b->isExit, __ATOMIC_ACQUIRE))
	{
		next = __atomic_load_n(&(b->count_next), __ATOMIC_ACQUIRE);

		/* Have the required number of threads entered? */
		if (wait - next == b->total /*TODO*/)
		{
			/* Move to next bunch */
			__atomic_store_n(&b->count_next, nextBunch, __ATOMIC_RELEASE);
			// ret = PTHREAD_BARRIER_SERIAL_THREAD;
			break;
		}

		/* Are we allowed to go? */
		if (wait - next >= (1UL << 31))
		{
			// ret = 0;
			break;
		}

		/* Go to sleep until our bunch comes up */
		while(__atomic_load_n(&b->count_next, __ATOMIC_ACQUIRE) == oldBunch && !__atomic_load_n(&b->isExit, __ATOMIC_ACQUIRE)) {
			thread_task_yield(); //asm("" ::: "memory");
			// if ((wait - next) + __atomic_load_n(&externalBarCount, __ATOMIC_ACQUIRE) == b->total) {
			// 	printf("[%i] found external barrier in use (wait=%d next=%d)\n", thread_get_task_id(), wait, next);
			// 	break;
			// }
		}

		// if ((wait - next) + __atomic_load_n(&externalBarCount, __ATOMIC_ACQUIRE) == b->total) {
		// 	printf("[%i] fixed count due to external barrier in use (wait=%d next=%d)\n", thread_get_task_id(), wait, next);
		// 	next = __atomic_load_n(&(b->count_next), __ATOMIC_ACQUIRE);
		// 	int countInVal = nextBunch + 1;
		// 	int r_count_in = __atomic_load_n(&b->count_in, __ATOMIC_ACQUIRE);
		// 	if (r_count_in < countInVal) {
		// 		__sync_val_compare_and_swap(&b->count_in, r_count_in, countInVal);
		// 	}
		// 	__atomic_store_n(&b->count_next, nextBunch, __ATOMIC_RELEASE);
		// 	break;
		// }
	}

	/* Add to count_out, simultaneously reading count_in */
	temp = __sync_fetch_and_add(&b->reset, 1ULL << 32);

	/* Does count_out == count_in? */
	if ((temp >> 32) == (unsigned) temp)
	{
		/* Notify destroyer */
		__sync_synchronize();
	}

	// __sync_add_and_fetch(&g_ticketBarCount, -1);

	// return ret;
}
// // old implementation
// void
// thread_barrier (thread_barrier_t* barrierPtr, long threadId)
// {
//     long i = 2;
//     long base = 0;
//     long index;
//     long numThread = barrierPtr->numThread;
//     if (numThread < 2) {
//         return;
//     }
//     do {
//         index = base + threadId / i;
//         if ((threadId % i) == 0) {
//             THREAD_MUTEX_LOCK(barrierPtr[index].countLock);
//             barrierPtr[index].count++;
//             while (barrierPtr[index].count < 2) {
//                 pthread_yield();
//                 // thread_yield();
//                 THREAD_COND_WAIT(barrierPtr[index].proceedCond,
//                                  barrierPtr[index].countLock);
//             }
//             THREAD_MUTEX_UNLOCK(barrierPtr[index].countLock);
//         } else {
//             THREAD_MUTEX_LOCK(barrierPtr[index].countLock);
//             barrierPtr[index].count++;
//             if (barrierPtr[index].count == 2) {
//                 THREAD_COND_SIGNAL(barrierPtr[index].proceedCond);
//             }
//             while (THREAD_COND_WAIT(barrierPtr[index].proceedAllCond,
//                                     barrierPtr[index].countLock) != 0)
//             {
//                 pthread_yield();
//                 // thread_yield();
//                 /* wait */
//             }
//             THREAD_MUTEX_UNLOCK(barrierPtr[index].countLock);
//             break;
//         }
//         base = base + numThread / i;
//         i *= 2;
//         pthread_yield();
//         // thread_yield();
//     } while (i <= numThread);
//     for (i /= 2; i > 1; i /= 2) {
//         base = base - numThread / i;
//         index = base + threadId / i;
//         THREAD_MUTEX_LOCK(barrierPtr[index].countLock);
//         barrierPtr[index].count = 0;
//         THREAD_COND_SIGNAL(barrierPtr[index].proceedAllCond);
//         THREAD_MUTEX_UNLOCK(barrierPtr[index].countLock);
//     }
// }


/* =============================================================================
 * thread_getId
 * -- Call after thread_start() to get thread ID inside parallel region
 * =============================================================================
 */
long
thread_getId()
{
	return (long)THREAD_LOCAL_GET(global_threadId);
}


/* =============================================================================
 * thread_getNumThread
 * -- Call after thread_start() to get number of threads inside parallel region
 * =============================================================================
 */
long
thread_getNumThread()
{
	return global_numThread;
}


/* =============================================================================
 * thread_barrier_wait
 * -- Call after thread_start() to synchronize threads inside parallel region
 * =============================================================================
 */
void
thread_barrier_wait()
{
	// printf("thread_barrier_wait\n");
#ifndef SIMULATOR
	long threadId = thread_get_task_id(); /*thread_getId();*/
#endif /* !SIMULATOR */
	// static int enterCount = 0;
	// extern int g_ticketBarCount;
	// extern int g_ticketBarTot;
	// extern int externalBarCount;
	// extern long externalBarTotal;
	// extern unsigned long long g_nbEnterThreads;

	// __sync_add_and_fetch(&enterCount, 1);
	// pthread_yield();
	// while (__atomic_load_n(&g_ticketBarTot, __ATOMIC_ACQUIRE) % g_nbEnterThreads != 0)
	// {
	//     // maybe there is some thread that is late in exiting the barrier ....
	//     handleMatrixLine(1); // exits if externalBarCount is set
	// }

	// __sync_add_and_fetch(&externalBarCount, 1);
	THREAD_BARRIER(global_barrierPtr, threadId);
	// __sync_add_and_fetch(&externalBarTotal, 1);
	// __sync_add_and_fetch(&externalBarCount, -1);
	// __sync_add_and_fetch(&enterCount, -1);
}


/* =============================================================================
 * TEST_THREAD
 * =============================================================================
 */
#ifdef TEST_THREAD


#include <stdio.h>
#include <unistd.h>


#define NUM_THREADS    (5)
#define NUM_ITERATIONS (3)



void
printId (void* argPtr)
{
	long threadId = thread_get_task_id()/*thread_getId()*/;
	long numThread = thread_getNumThread();
	long i;

	for ( i = 0; i < NUM_ITERATIONS; i++ ) {
		thread_barrier_wait();
		if (threadId == 0) {
			sleep(2);
		} else if (threadId == numThread-1) {
			sleep(1);
			// usleep(100); // TODO: crashes with userlevel threads... why?
		}
		printf("i = %li, tid = %li\n", i, threadId);
		if (threadId == 0) {
			puts("");
		}
		fflush(stdout);
	}
}


int
main ()
{
	puts("Starting...");

	/* Run in parallel */
	thread_startup(NUM_THREADS);
	/* Start timing here */
	thread_start(printId, NULL);
	// thread_start(printId, NULL);
	// thread_start(printId, NULL);
	/* Stop timing here */
	thread_shutdown();

	puts("Done.");

	return 0;
}


#endif /* TEST_THREAD */


/* =============================================================================
 *
 * End of thread.c
 *
 * =============================================================================
 */
