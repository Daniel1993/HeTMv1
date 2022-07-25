/* Include this file in your pr-stm-implementation file,
 * then use the MACROs accordingly!
 * Do not forget to include pr-stm-internal.cuh after this
 */
#ifndef PR_STM_WRAPPER_H_GUARD_
#define PR_STM_WRAPPER_H_GUARD_

// #define PR_GRANULE_T        int // TODO
// #define PR_LOCK_GRANULARITY 4 /* size in bytes */
// #define PR_LOCK_GRAN_BITS   2 /* log2 of the size in bytes */
// #define PR_LOCK_TABLE_SIZE  0x800000

// TODO: this is benchmark dependent
// #ifdef BENCH_MEMCD
// #define PR_MAX_RWSET_SIZE   NUMBER_WAYS
// #else /* !BENCH_MEMCD */
// #define PR_MAX_RWSET_SIZE   BANK_NB_TRANSFERS
// #endif /* BENCH_MEMCD */

#include "pr-stm.cuh"

#include "bitmap.hpp"
#include "memman.hpp"
#include "hetm.cuh"

typedef struct HeTM_GPU_log_ {
	int devId;
	void *bmap_rset_devptr;
	void *bmap_wset_devptr;
	void *bmap_wset_cache_devptr;
	void *bmap_rset_cache_devptr;
	long *state; /* TODO: this is benchmark specific */
	void *devMemPoolBasePtr;
	void *hostMemPoolBasePtr;
	// memman_bmap_s *bmap;
	// memman_bmap_s *bmapBackup;
	long batchCount;
	int isGPUOnly;
} HeTM_GPU_log_s;

typedef struct HeTM_GPU_dbuf_log_ {
	HeTM_GPU_log_s gpuLog;
	long *state;
} HeTM_GPU_dbuf_log_s;

// TODO: move PR_BEFORE_RUN_EXT here and set the pointer before PR_INIT
// void impl_pr_clbk_before_run_ext(pr_tx_args_s*)
// {
// }
// extern void (*pr_clbk_after_run_ext)(pr_tx_args_s*);

#ifdef PR_BEFORE_RUN_EXT
#undef PR_BEFORE_RUN_EXT
#endif

// in hetm-prstm-aux-fn.cpp
typedef struct pr_tx_args_ pr_tx_args_s;
void hetm_impl_pr_clbk_before_run_ext(pr_tx_args_s *args);
void hetm_impl_pr_clbk_after_run_ext(pr_tx_args_s *args);

#ifdef BMAP_ENC_1BIT
#define SET_ON_RS_BMAP_AUX(_pos, _GPU_log) \
	BM_SET_POS_GPU(_pos, (_GPU_log)->bmap_rset_devptr); \
//
#define SET_ON_RS_BMAP_CACHE_AUX(_pos, _GPU_log) \
	/* printf("SET_ON_RS_BMAP_CACHE pos %lu\n", _rposCache); */ \
	BM_SET_POS_GPU(_pos, (_GPU_log)->bmap_rset_cache_devptr); \
//
#define SET_ON_WS_BMAP_AUX(_pos, _GPU_log) \
	BM_SET_POS_GPU(_pos, (_GPU_log)->bmap_wset_devptr) \
//
#define SET_ON_WS_BMAP_CACHE_AUX(_pos, _GPU_log) \
	BM_SET_POS_GPU(_pos, (_GPU_log)->bmap_wset_cache_devptr); \
//
#else
#define SET_ON_RS_BMAP_AUX(_pos, _GPU_log) \
	ByteM_SET_POS(_pos, (_GPU_log)->bmap_rset_devptr, (_GPU_log)->batchCount); \
//
#define SET_ON_RS_BMAP_CACHE_AUX(_pos, _GPU_log) \
	ByteM_SET_POS(_pos, (_GPU_log)->bmap_rset_cache_devptr, (_GPU_log)->batchCount); \
//
#define SET_ON_WS_BMAP_AUX(_pos, _GPU_log) \
	ByteM_SET_POS(_pos, (_GPU_log)->bmap_wset_devptr, (_GPU_log)->batchCount) \
//
#define SET_ON_WS_BMAP_CACHE_AUX(_pos, _GPU_log) \
	ByteM_SET_POS(_pos, (_GPU_log)->bmap_wset_cache_devptr, (_GPU_log)->batchCount); \
//
#endif

#ifndef HETM_REDUCED_RS
#define HETM_REDUCED_RS 0
#endif /* HETM_REDUCED_RS */

#define  SET_ON_WS_BMAP(_addr, _GPU_log) ({ \
	unsigned long pos = ((uintptr_t)(_addr) - (uintptr_t)((_GPU_log)->devMemPoolBasePtr)) >> (PR_LOCK_GRAN_BITS+HETM_REDUCED_RS); \
	unsigned long posCache = pos >> BMAP_GRAN_BITS; \
	/* printf("set on WRITE SET pos %li posCache %li in %p\n", pos, posCache, (_GPU_log)->bmap_wset_devptr); */ \
	SET_ON_WS_BMAP_AUX(pos, _GPU_log); \
	SET_ON_WS_BMAP_CACHE_AUX(posCache, _GPU_log) \
}) //

#define SET_ON_RS_BMAP(_addr, _GPU_log) ({ \
	unsigned long pos = ((uintptr_t)(_addr) - (uintptr_t)((_GPU_log)->devMemPoolBasePtr)) >> (PR_LOCK_GRAN_BITS+HETM_REDUCED_RS); \
	unsigned long posCache = pos >> BMAP_GRAN_BITS; \
	/* printf("set on READ SET pos %li posCache %li in %p\n", pos, posCache, (_GPU_log)->bmap_rset_devptr); */ \
	SET_ON_RS_BMAP_AUX(pos, _GPU_log); \
	SET_ON_RS_BMAP_CACHE_AUX(posCache, _GPU_log) \
}) //

#define PR_i_rand(args, n) ({ \
	HeTM_GPU_log_s *GPU_log = (HeTM_GPU_log_s*)args.pr_args_ext; \
	int id = PR_THREAD_IDX; \
	unsigned x; \
	long *state = GPU_log->state; \
	x = RAND_R_FNC(state[id]); \
	(unsigned) (x % n); \
}) \
//
#define PR_rand(n) \
	PR_i_rand(args, n) \
//

#ifdef HETM_DISABLE_RS
#define PR_AFTER_VAL_LOCKS_GATHER_READ_SET(i) /* empty */
#else /* !HETM_DISABLE_RS */
#define PR_AFTER_VAL_LOCKS_GATHER_READ_SET(_i) \
	for (_i = 0; _i < args->rset.size; _i++) { \
		SET_ON_RS_BMAP(args->rset.addrs[_i], GPU_log); \
	} \
//
#endif /* HETM_DISABLE_RS */

#ifdef HETM_DISABLE_WS
#define PR_AFTER_VAL_LOCKS_GATHER_WRITE_SET(i) /* empty */
#else /* !HETM_DISABLE_WS	 */
#define PR_AFTER_VAL_LOCKS_GATHER_WRITE_SET(_i) \
	for (_i = 0; _i < args->wset.size; _i++) { \
		/* this is avoided through a memcpy D->D after batch */ \
		/* memman_access_addr_dev(GPU_log->bmap, args->wset.addrs[_i], GPU_log->batchCount); */ /* TODO */ \
		SET_ON_RS_BMAP(args->wset.addrs[_i], GPU_log); \
		SET_ON_WS_BMAP(args->wset.addrs[_i], GPU_log); \
	} \
//
#endif /* HETM_DISABLE_WS */

__global__ void HeTM_setupCurand(void *args);

#endif /* PR_STM_WRAPPER_H_GUARD_ */
