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
  if (_e != NULL) sigsetjmp(*_e, 0)

// TODO: does assumptions on the machine --> PR-STM works with ints
#define TM_LOAD(addr)         ({ \
  stm_log_read_entry((long*)(addr)); \
  uintptr_t _mask64Bits = ((uintptr_t)(-1)) << 3; \
  uintptr_t _newAddr = (uintptr_t)(addr) & _mask64Bits; \
  uintptr_t _loaded, _high, _low; \
  int _res; \
  _loaded = stm_load((stm_word_t *)_newAddr); \
  _low = _loaded & (0xFFFFFFFFL); \
  _high = (_loaded & (0xFFFFFFFFL << 32)) >> 32; \
  if ((uintptr_t)(addr) & 0x4) \
    _res = _high; \
  else \
    _res = _low; \
  _res; \
})

#define TM_STORE(addr, value) ({ \
  uintptr_t _mask64Bits = ((uintptr_t)(-1)) << 3; \
  uintptr_t _newAddr = (uintptr_t)(addr) & _mask64Bits; \
  uintptr_t _toStore, _high, _low, _value = (uintptr_t)value; \
  if ((uintptr_t)(addr) & 0x4) { \
    _high = _value & (0xFFFFFFFFL); \
    _low = *((stm_word_t*)(_newAddr)) & (0xFFFFFFFFL); /* 32 bits */ \
  } else { \
    _high = (*((stm_word_t*)(_newAddr)) & (0xFFFFFFFFL << 32)) >> 32; \
    _low = _value & (0xFFFFFFFFL); \
  } \
  _toStore = (_high << 32) | _low; \
  stm_store((stm_word_t *)(_newAddr), (stm_word_t)_toStore); \
})

#define TM_COMMIT             stm_commit(); }

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
#define GRANULE_TYPE int
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


#endif /* STM_WRAPPER_H_GUARD_ */
