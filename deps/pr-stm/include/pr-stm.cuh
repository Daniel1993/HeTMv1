/**
 * TODO: PR-STM License
 *
 * For performance reasons all functions on PR-STM are implemented
 * in headers
 *
 * TODO: cite PR-STM paper [EuroPar15]
 */
#ifndef PR_STM_H_GUARD
#define PR_STM_H_GUARD

#ifdef __cplusplus // only works on c++

#include <cuda_runtime.h>
#include "cuda_util.h"
#include <stdint.h>

#include "pr-stm-track-rwset-structs.cuh"

 // TODO: needs to defined on compilation

#ifndef PR_LOCK_TABLE_SIZE
// must be power 2
#define PR_LOCK_TABLE_SIZE  0x80000
#endif

#ifndef PR_GRANULE_T
#define PR_GRANULE_T        int
#define PR_LOCK_GRANULARITY 4 /* size in bytes */
#define PR_LOCK_GRAN_BITS   2 /* log2 of the size in bytes */
#endif

#ifndef PR_MAX_RWSET_SIZE
#define PR_MAX_RWSET_SIZE   1024
#endif


// PR-STM extensions, in order to extend PR-STM implement the following
// ##########################################
// --------------------------------------------------------------------
// implement these or it will crash... if not need just copy paste this
// PR_DEVICE void PR_beforeKernel_EXT(PR_txCallDefArgs) { }
// PR_DEVICE void PR_afterKernel_EXT (PR_txCallDefArgs) { }
// PR_DEVICE void PR_beforeBegin_EXT (PR_txCallDefArgs) { }
// PR_DEVICE void PR_afterCommit_EXT (PR_txCallDefArgs) { }
// PR_DEVICE void PR_after_val_locks_EXT (PR_args_s *args) { }
// PR_DEVICE void PR_after_writeback_EXT (PR_args_s *args, i, addr, val) { }
// --------------------------------------------------------------------

// NOTE: you really need to implement these two, else it crashes
// (they come after the definition of pr_tx_args_s)
// extern void (*pr_clbk_before_run_ext)(pr_tx_args_s*);
// extern void (*pr_clbk_after_run_ext)(pr_tx_args_s*);
// ##########################################


#define PR_DEVICE       __device__
#define PR_HOST         __host__
#define PR_BOTH         __device__ __host__
#define PR_ENTRY_POINT  __global__

// This is in pr-stm-internal.cuh (Do not call it!)
#define PR_globalVars \
	__thread int PR_curr_dev = 0; \
	int PR_enable_auto_stats = 0; \
	PR_global_data_s PR_global[PR_MAX_DEVICES]; \
//

#define PR_globalKernelArgs \
	pr_tx_args_dev_host_s args

// if a function is called these must be sent (still compatible with STAMP)
#define PR_txCallDefArgs \
	pr_tx_args_dev_host_s &args, PR_args_s &pr_args \
//
#define PR_txCallArgs \
	args, pr_args \
//

// The application must "PR_globalVars;" somewhere in the main file
static const int PR_MAX_DEVICES = 16;
static const int PR_MAX_NB_STREAMS = 8;

typedef struct PR_init_params_ {
	int nbStreams;
	int lockTableSize; // must be a power2 equal to PR_LOCK_TABLE_SIZE used in pr-stm-internal.cuh
} PR_init_params_s;

typedef struct PR_global_data_ {
	int PR_blockNum;
	int PR_threadNum;
	volatile int PR_isStart[PR_MAX_NB_STREAMS];
	volatile int PR_isDone[PR_MAX_NB_STREAMS];
	long long PR_nbAborts;
	long long PR_nbCommits;
	long long PR_nbAbortsStrm [PR_MAX_NB_STREAMS];
	long long PR_nbCommitsStrm[PR_MAX_NB_STREAMS];
	long long PR_nbAbortsLastKernel ;
	long long PR_nbCommitsLastKernel;
	long long *PR_sumNbAborts ;
	long long *PR_sumNbCommits;
	long long *PR_sumNbAbortsDev;
	long long *PR_sumNbCommitsDev;
	long long PR_nbAbortsSinceCheckpoint ;
	long long PR_nbCommitsSinceCheckpoint;
	double PR_kernelTime;
	cudaStream_t *PR_streams;
	int PR_currentStream;
	int PR_streamCount;
	int *PR_lockTableHost;
	int *PR_lockTableDev;

	cudaEvent_t PR_eventKernelStart;
	cudaEvent_t PR_eventKernelStop;
} PR_global_data_s;

extern int PR_enable_auto_stats;
extern __thread int PR_curr_dev;
extern PR_global_data_s PR_global[PR_MAX_DEVICES];

#define PR_SET(addr, value)       (*((PR_GRANULE_T*)addr) = (PR_GRANULE_T)(value))
#define PR_MOD_ADDR(addr)         ((addr) & (PR_LOCK_TABLE_SIZE-1))
#define PR_GET_MTX(mtx_tbl, addr) ((mtx_tbl)[PR_MOD_ADDR(((uintptr_t)addr) >> PR_LOCK_GRAN_BITS)])

#define PR_LOCK_NB_LOCK_BITS     2
#define PR_LOCK_NB_OWNER_BITS    22
#define PR_LOCK_VERSION_BITS     24 // PR_LOCK_NB_LOCK_BITS + PR_LOCK_NB_OWNER_BITS
#define PR_LOCK_NB_VERSION_BITS  8 // 32 - PR_LOCK_VERSION_BITS
#define PR_LOCK_VERSION_OVERFLOW 256 // 32 - PR_LOCK_VERSION_BITS

#define PR_GET_VERSION(x)   ( ((x) >> 24) & 0xff    )
#define PR_CHECK_PRELOCK(x) (  (x)        & 0b1     )
#define PR_CHECK_LOCK(x)    ( ((x) >> 1)  & 0b1     )
#define PR_GET_OWNER(x)     ( ((x) >> 2)  & 0x3fffff)
#define PR_MASK_VERSION     0xff000000
#define PR_THREAD_IDX       (threadIdx.x + blockIdx.x * blockDim.x) // TODO: 3D thread-id grid

// TODO: maximum nb threads is 2048*1024
#define PR_PRELOCK_VAL(version, id) ((version << 24) | (id << 2) | 0b01)
#define PR_LOCK_VAL(version, id)    ((version << 24) | (id << 2) | 0b11)

#define PR_CHECK_CUDA_ERROR(func, msg) \
	CUDA_CHECK_ERROR(func, msg) \
//

#define PR_ALLOC(ptr, size) \
	CUDA_DEV_ALLOC(ptr, size) \
//

#define PR_CPY_TO_DEV(dev, host, size) \
	CUDA_CPY_TO_DEV(dev, host, size) \
//

#define PR_CPY_TO_HOST(host, dev, size) \
	CUDA_CPY_TO_HOST(host, dev, size) \
//

#define PR_CPY_TO_HOST_ASYNC(host, dev, size, stream) \
	CUDA_CPY_TO_HOST_ASYNC(host, dev, size, stream) \
//

typedef struct PR_rwset_ { // TODO: this should be a sort of a hashmap
	PR_GRANULE_T **addrs;
	PR_GRANULE_T  *values;
	int           *versions;
	int            size;
} PR_rwset_s;

typedef struct PR_args_ {
	int     tid;
	void   *pr_args_ext; /* can send pointers to extra datastructures via these */
	void   *pr_args_ext1;
	void   *pr_args_ext2;
	int    *mtx;
	int     is_abort;
	void   *inBuf;
	size_t  inBuf_size;
	void   *outBuf;
	size_t  outBuf_size;
	int     current_stream;
	// below is private
	PR_rwset_s rset;
	PR_rwset_s wset;
} PR_args_s;

typedef struct pr_buffer_ {
	void  *buf;
	size_t size;
} pr_buffer_s;

// mutex format: TODO: review this
// lock,version,owner in format version<<5|owner<<2|lock

typedef struct pr_tx_args_dev_host_ {
	void   *pr_args_ext; /* add struct of type PR_DEV_BUFF_S_EXT here */
	void   *pr_args_ext1;
	void   *pr_args_ext2;
	int    *mtx;
	int     devId;
	unsigned int *nbAborts[PR_MAX_DEVICES];
	unsigned int *nbCommits[PR_MAX_DEVICES];
	void   *inBuf;
	size_t  inBuf_size;
	void   *outBuf;
	size_t  outBuf_size;
	int     current_stream;
} pr_tx_args_dev_host_s;

typedef struct pr_tx_args_ {
	void                 (*callback)(pr_tx_args_dev_host_s);
	pr_tx_args_dev_host_s  dev;
	pr_tx_args_dev_host_s  host;
	void                  *stream;
} pr_tx_args_s;

// ##########################################
// NOTE: you really need to implement these two, else it crashes
extern void (*pr_clbk_before_run_ext)(pr_tx_args_s*);
extern void (*pr_clbk_after_run_ext)(pr_tx_args_s*);
// ##########################################

// defines local variables
// TODO: we need to have a very good estimate of PR_MAX_RWSET_SIZE
#define PR_allocRWset(set) ({ \
	PR_GRANULE_T *pr_internal_addrs[PR_MAX_RWSET_SIZE]; \
	PR_GRANULE_T pr_internal_values[PR_MAX_RWSET_SIZE]; \
	int pr_internal_versions[PR_MAX_RWSET_SIZE]; \
	set.addrs = pr_internal_addrs; \
	set.values = pr_internal_values; \
	set.versions = pr_internal_versions; \
}) \
//
#define PR_freeRWset(set) /* empty: local variables used */
//

#define PR_enterKernel(_tid) \
	PR_args_s pr_args; \
	PR_allocRWset(pr_args.rset); \
	PR_allocRWset(pr_args.wset); \
	pr_args.tid = _tid; \
	pr_args.mtx = args.mtx; \
	pr_args.inBuf = args.inBuf; \
	pr_args.inBuf_size = args.inBuf_size; \
	pr_args.outBuf = args.outBuf; \
	pr_args.outBuf_size = args.outBuf_size; \
	pr_args.pr_args_ext = args.pr_args_ext; \
	pr_args.pr_args_ext1 = args.pr_args_ext1; \
	pr_args.pr_args_ext2 = args.pr_args_ext2; \
	pr_args.current_stream = args.current_stream; \
	PR_beforeKernel_EXT(PR_txCallArgs); \
//
#define PR_exitKernel() \
	PR_freeRWset(pr_args.rset); \
	PR_freeRWset(pr_args.wset); \
	PR_afterKernel_EXT(PR_txCallArgs); \
//

// setjmp is not available --> simple while
#define PR_txBegin() \
do { \
	PR_beforeBegin_EXT(PR_txCallArgs); \
	pr_args.rset.size = 0; \
	pr_args.wset.size = 0; \
	pr_args.is_abort = 0 \
//

#define PR_txCommit() \
	PR_validateKernel(); \
	PR_commitKernel(); \
	PR_afterCommit_EXT(PR_txCallArgs); \
	if (!pr_args.is_abort) { \
		if (args.nbCommits[args.devId] != NULL) { \
			atomicInc(&(args.nbCommits[args.devId][pr_args.tid + blockDim.x*gridDim.x*pr_args.current_stream]), 0x7FFFFFFF); \
		} \
	} else { \
		if (args.nbAborts[args.devId] != NULL) { \
			atomicInc(&(args.nbAborts[args.devId][pr_args.tid + blockDim.x*gridDim.x*pr_args.current_stream]), 0x7FFFFFFF); \
		} \
	} \
} while (pr_args.is_abort); \
//

/**
 * Transactional Read.
 *
 * Reads data from global memory to local memory.
 */
#define PR_read(a) ({ \
	PR_GRANULE_T r; \
	r = PR_i_openReadKernel(&pr_args, (PR_GRANULE_T*)(a)); \
	r; \
})

/**
 * Transactional Write.
 *
 * Do calculations and write result to local memory.
 * Version will increase by 1.
 */
#define PR_write(a, v) ({ /*TODO: v cannot be a constant*/ \
	PR_i_openWriteKernel(&pr_args, (PR_GRANULE_T*)(a), v); \
})

/**
 * Validate function.
 *
 * Try to lock all memory this thread need to write and
 * check if any memory this thread read is changed.
 */
#define PR_validateKernel() \
	PR_i_validateKernel(&pr_args)

/**
 * Commit function.
 *
 * Copies results (both value and version) from local memory
 * (write set) to global memory (data and lock).
 */
#define PR_commitKernel() \
	PR_i_commitKernel(&pr_args) \

// Use these MACROs to wait execution (don't forget to put a lfence)
#define PR_IS_START          (__atomic_load_n(&PR_global[PR_curr_dev].PR_isStart[PR_global[PR_curr_dev].PR_currentStream], __ATOMIC_ACQUIRE))
#define PR_SET_IS_START(_i)  (__atomic_store_n(&PR_global[PR_curr_dev].PR_isStart[PR_global[PR_curr_dev].PR_currentStream], _i, __ATOMIC_RELEASE))
#define PR_IS_DONE           (__atomic_load_n(&PR_global[PR_curr_dev].PR_isDone[PR_global[PR_curr_dev].PR_currentStream], __ATOMIC_ACQUIRE))
#define PR_SET_IS_DONE(_i)   (__atomic_store_n(&PR_global[PR_curr_dev].PR_isDone[PR_global[PR_curr_dev].PR_currentStream], _i, __ATOMIC_RELEASE))
#define PR_WAIT_START_COND  (!PR_IS_START && !PR_IS_DONE)
#define PR_WAIT_FINISH_COND (!PR_IS_DONE)

// IMPORTANT: do not forget calling this after wait
#define PR_AFTER_WAIT(args) ({ \
	float kernelElapsedTime; \
	PR_SET_IS_DONE(0); \
	PR_SET_IS_START(0); \
	/* cudaEventSynchronize(PR_eventKernelStop); done by cudaStreamSynchronize */ \
	cudaEventElapsedTime(&kernelElapsedTime, PR_global[PR_curr_dev].PR_eventKernelStart, PR_global[PR_curr_dev].PR_eventKernelStop); \
	PR_global[PR_curr_dev].PR_kernelTime += kernelElapsedTime; \
	kernelElapsedTime; \
}) \
//

// TODO: put an _mpause
#define PR_waitKernel(args) \
while(PR_WAIT_START_COND) { \
	/*asm("" ::: "memory")*//*pthread_yield();*/ \
} \
PR_CHECK_CUDA_ERROR(cudaStreamSynchronize(PR_global[PR_curr_dev].PR_streams[PR_global[PR_curr_dev].PR_currentStream]), ""); \
while(PR_WAIT_FINISH_COND) { /* this should not be needed */ \
	/*asm("" ::: "memory")*//*pthread_yield();*/ \
} \
PR_AFTER_WAIT(args); \
//

// Important: Only one file in the project must include pr-stm-internal.cuh!
//            Put custom implementation by re-defining *_EXT MACROs there.
// Example:
//   #undef PR_AFTER_COMMIT_EXT
//   #define PR_AFTER_COMMIT_EXT typedef struct /* your custom implementation */
// Then do:
// #include "pr-stm-internal.cuh" // implementation using your MACROs

// wrapper functions
PR_HOST void PR_init(PR_init_params_s); // call this before anything else

PR_HOST void PR_noStatistics(pr_tx_args_s *args);
PR_HOST void PR_createStatistics(pr_tx_args_s *args);
PR_HOST void PR_resetStatistics(pr_tx_args_s *args);
PR_HOST void PR_prepareIO(pr_tx_args_s *args, pr_buffer_s inBuf, pr_buffer_s outBuf);
PR_HOST void PR_run(void(*callback)(pr_tx_args_dev_host_s), pr_tx_args_s *args);

PR_HOST cudaStream_t PR_getCurrentStream();

// Call this after retriving output to change to the next stream.
// This also calls PR_reduceCommitAborts to obtain statistics.
// It does not synchronize the stream, thus call it twice to force
// the copy of current data, e.g., before exiting.
PR_HOST void PR_useNextStream(pr_tx_args_s *args);

// resets aborts and commits into PR_nb*SinceCheckpoint
PR_HOST void PR_checkpointAbortsCommits();

PR_HOST void PR_retrieveIO(pr_tx_args_s *args); // updates PR_nbAborts and PR_nbCommits
PR_HOST void PR_disposeIO(pr_tx_args_s *args);

PR_HOST void PR_teardown(); // after you are done

PR_ENTRY_POINT void PR_reduceCommitAborts(
	int doReset,
	int targetStream,
	PR_globalKernelArgs,
	uint64_t *nbCommits,
	uint64_t *nbAborts);


PR_DEVICE void PR_beforeKernel_EXT(PR_txCallDefArgs);
PR_DEVICE void PR_afterKernel_EXT (PR_txCallDefArgs);
PR_DEVICE void PR_beforeBegin_EXT (PR_txCallDefArgs);
PR_DEVICE void PR_afterCommit_EXT (PR_txCallDefArgs);
PR_DEVICE void PR_after_val_locks_EXT (PR_args_s *args);
PR_DEVICE void PR_after_writeback_EXT (PR_args_s *args, int i, PR_GRANULE_T *addr, PR_GRANULE_T val);

// internal 
PR_DEVICE PR_GRANULE_T PR_i_openReadKernel(PR_args_s *args, PR_GRANULE_T *addr);
PR_DEVICE void PR_i_openWriteKernel(PR_args_s *args, PR_GRANULE_T *addr, PR_GRANULE_T wbuf);
PR_DEVICE void PR_i_validateKernel(PR_args_s *args);
PR_DEVICE void PR_i_commitKernel(PR_args_s *args);

#endif /** __cplusplus **/

#endif /* PR_STM_H_GUARD */
