#ifndef TSX_IMPL_GUARD_H_
#define TSX_IMPL_GUARD_H_

#define HTM_SGL_INIT_BUDGET 10

#include "htm_retry_template.h"
#include "rdtsc.h"
#include "hetm-log.h"

extern int errors[HETM_MAX_THREADS][HTM_NB_ERRORS];

// updates the statistics
#undef HTM_INC
#define HTM_INC(status) \
  TM_inc_error(HTM_SGL_tid, status) \
//

#define TM_inc_error(tid, error) \
  HTM_ERROR_INC(error, errors[tid])

#define TM_get_error(error) ({ \
  int res = errors[HTM_SGL_tid][error]; \
  res; \
}) \

// #undef AFTER_ABORT
// #define AFTER_ABORT(tid, budget, status) printf(" [%i] ", tid)

#undef BEFORE_TRANSACTION
#define BEFORE_TRANSACTION(tid, budget) \
  HeTM_ptr = 0; \
  HeTM_ptr_reads = 0; \
//

#undef BEFORE_COMMIT
#define BEFORE_COMMIT(tid, budget, status) ({ \
  HeTM_version = rdtscp(); \
})

#undef AFTER_TRANSACTION
#ifdef HETM_INSTRUMENT_CPU
#define AFTER_TRANSACTION(tid, budget) ({ \
  uintptr_t i; \
  for (i = 0; i < HeTM_ptr; ++i) { \
    /* printf("HeTM_bufAddrs[HeTM_ptr]=%p\n", HeTM_bufAddrs[i]); */\
    stm_log_newentry(HeTM_log, (long*)HeTM_bufAddrs[i], HeTM_bufVal[i], HeTM_version); \
  } \
  for (i = 0; i < HeTM_ptr_reads; ++i) { \
    /* printf("HeTM_bufAddrs_reads[HeTM_ptr_reads]=%p\n", HeTM_bufAddrs_reads[i]); */\
    stm_log_read_entry((long*)HeTM_bufAddrs_reads[i]); \
  } \
})
#else
#define AFTER_TRANSACTION(tid, budget) /* empty */
#endif

#undef HTM_SGL_after_write
#ifdef HETM_INSTRUMENT_CPU
#define HTM_SGL_after_write(addr, val) ({ \
  uintptr_t _val = (val); \
  HeTM_bufAddrs[HeTM_ptr] = addr; \
  HeTM_bufVal[HeTM_ptr]   = (uintptr_t)_val; \
  /*HeTM_bufVers[HeTM_ptr]  = 0;*/ \
  HeTM_ptr++; \
}) \
//
#else
#define HTM_SGL_after_write(addr, val) /* empty */
#endif

#undef HTM_SGL_before_read
#ifdef HETM_INSTRUMENT_CPU
#define HTM_SGL_before_read(addr) ({ \
  HeTM_bufAddrs_reads[HeTM_ptr_reads] = addr; \
  /*HeTM_bufVers[HeTM_ptr]  = 0;*/ \
  HeTM_ptr_reads++; \
}) \
//
#else
#define HTM_SGL_before_read(addr) /* empty */
#endif

#undef AFTER_SGL_BEGIN
#define AFTER_SGL_BEGIN(tid) \
  errors[tid][HTM_FALLBACK]++;

// not defined in the template
#define HTM_SGL_init_thr() ({ HTM_thr_init(-1); HeTM_log = stm_log_init(); HeTM_htmRndSeed *= HTM_SGL_tid + 1234; })
#define HTM_SGL_exit_thr() ({ HTM_thr_exit(); if (HeTM_log != NULL) { stm_log_free(HeTM_log); HeTM_log = NULL; } })

#define HeTM_get_log(ptr) ({ *(HeTM_CPULogNode_t **)ptr = stm_log_read(HeTM_log); })

#undef AFTER_ABORT
#define AFTER_ABORT(tid, budget, status) ({ \
  if (RAND_R_FNC(HeTM_htmRndSeed) % 10000 < 10) { pthread_yield(); budget = 0; } \
})

#endif /* TSX_IMPL_GUARD_H_ */
