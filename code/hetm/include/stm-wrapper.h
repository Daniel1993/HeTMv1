#ifndef STM_WRAPPER_H_GUARD_
#define STM_WRAPPER_H_GUARD_

/*
 * Useful macros to work with transactions. Note that, to use nested
 * transactions, one should check the environment returned by
 * stm_get_env() and only call sigsetjmp() if it is not null.
 */

 /*
  * When a thread enters its routine, one must call TM_INIT_THREAD with
  * the dataset base pointer and size (located in HeTM_shared_data.mempool_hostptr)
  */

// TODO: requires to compile HeTM with the correct set of flags
#ifndef USE_TSX_IMPL
// uses TinySTM
#define TM_START(tid, ro) { \
  stm_tx_attr_t _a = {{.id = (unsigned int)tid, .read_only = (unsigned int)ro}}; \
  sigjmp_buf *_e = stm_start(_a); \
  if (_e != NULL) sigsetjmp(*_e, 0); \
  HeTM_ptr = 0; \
  HeTM_ptr_reads = 0; \
//

// TODO: does assumptions on the machine --> PR-STM works with ints
#define TM_LOAD(addr) ({ \
  uintptr_t _mask64Bits = ((uintptr_t)(-1)) << 3; \
  uintptr_t _newAddr = (uintptr_t)(addr) & _mask64Bits; \
  uintptr_t _loaded, _high, _low; \
  int _res; \
  HeTM_bufAddrs_reads[HeTM_ptr_reads] = addr; \
  /*HeTM_bufVers[HeTM_ptr]  = 0;*/ \
  HeTM_ptr_reads++; \
  _loaded = stm_load((stm_word_t *)_newAddr); \
  _low = _loaded & (0xFFFFFFFFL); \
  _high = (_loaded & (0xFFFFFFFFL << 32)) >> 32; \
  if ((uintptr_t)(addr) & 0x4) \
    _res = _high; \
  else \
    _res = _low; \
  _res; \
})

// TODO: TinySTM bug on 32 bit addresses
#define TM_STORE(addr, value) ({ \
  uintptr_t _mask64Bits = ((uintptr_t)(-1)) << 3; \
  uintptr_t _newAddr = (uintptr_t)(addr) & _mask64Bits; \
  uintptr_t _toStore, _high, _low, _value = (uintptr_t)value; \
  HeTM_bufAddrs[HeTM_ptr] = addr; \
  HeTM_bufVal[HeTM_ptr]   = _value; \
  /*HeTM_bufVers[HeTM_ptr]  = 0;*/ \
  HeTM_ptr++; \
  if ((uintptr_t)(addr) & 0x4) { \
    _high = _value & (0xFFFFFFFFL); \
    _low = *((stm_word_t*)(_newAddr)) & (0xFFFFFFFFL); \
  } else { \
    _high = (*((stm_word_t*)(_newAddr)) & (0xFFFFFFFFL << 32)) >> 32; \
    _low = _value & (0xFFFFFFFFL); \
  } \
  _toStore = (_high << 32) | _low; \
  stm_store((stm_word_t *)(_newAddr), (stm_word_t)_toStore); \
})

// TODO: tinySTM does not store read addresses 
// #ifdef HETM_INSTRUMENT_CPU
//   r_entry_t *r = tx->r_set.entries;
//   for (i = tx->r_set.nb_entries; i > 0; i--, r++) {
//     stm_log_read_entry((long*)r->addr);
//   }
// #endif /*HETM_INSTRUMENT_CPU*/

#define TM_COMMIT             stm_commit(); \
  for (size_t i = 0; i < HeTM_ptr; ++i) { \
    /* printf("HeTM_bufAddrs[HeTM_ptr]=%li\n", (int*)HeTM_bufAddrs[i] - (int*)HeTM_shared_data[0].mempool_hostptr); */\
    stm_log_newentry(HeTM_log, (long*)HeTM_bufAddrs[i], HeTM_bufVal[i], HeTM_version); \
  } \
  for (size_t i = 0; i < HeTM_ptr_reads; ++i) { \
    /* printf("HeTM_bufAddrs_reads[HeTM_ptr_reads]=%p\n", HeTM_bufAddrs_reads[i]); */\
    stm_log_read_entry((long*)HeTM_bufAddrs_reads[i]); \
  } \
  HeTM_version = rdtsc() /* NOT USED*/; }

#define TM_GET_LOG(p)         p = stm_thread_local_log
#define TM_LOG(val)           stm_log_add(0, val)
#define TM_LOG2(pos,val)      stm_log_add(pos, val)
#define TM_FREE(point)        free(point)

#define TM_INIT(nb_threads)   stm_init(); mod_ab_init(0, NULL)
#define TM_EXIT()             stm_exit()
#define TM_INIT_THREAD(p,s)   stm_init_thread(); /*stm_log_init_bm(p,s)*/
#define TM_EXIT_THREAD()      stm_exit_thread()
#else /* USE_TSX_IMPL */
// redefine with TSX
#ifdef GRANULE_TYPE
#undef GRANULE_TYPE
#define GRANULE_TYPE int
#endif
#include "tsx_impl.h"
#define TM_START(tid, ro) 		HTM_SGL_begin();
#define TM_COMMIT 		        HTM_SGL_commit();
#define TM_LOAD(addr)         HTM_SGL_read(addr)
#define TM_STORE(addr, value) HTM_SGL_write(addr, value)

#define TM_GET_LOG(p)         HeTM_get_log(&p)
#define TM_LOG(val)           /*stm_log_add(0,val)*/
#define TM_LOG2(pos,val)      /*stm_log_add(pos,val)*/
#define TM_FREE(point)        free(point)

#define TM_INIT(nb_threads)		HTM_init(nb_threads) /*stm_init(); mod_ab_init(0, NULL)*/
#define TM_EXIT()             HTM_exit(); stm_exit()
#define TM_INIT_THREAD(p,s)   HTM_SGL_init_thr(); stm_init_thread(); /*stm_log_init_bm(p, s)*/
#define TM_EXIT_THREAD()      HTM_SGL_exit_thr() /*stm_exit_thread()*/

#endif /* USE_TSX_IMPL */

#include "hetm-log.h"

#endif /* STM_WRAPPER_H_GUARD_ */
