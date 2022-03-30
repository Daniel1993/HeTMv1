#ifndef STM_H
#define STM_H 1

#  include <tinystm.h>
#  include "thread.h"

// #    define TM_ARG                        STM_SELF,
// #    define TM_ARG_ALONE                  STM_SELF
// #    define TM_ARGDECL                    STM_THREAD_T* TM_ARG
// #    define TM_ARGDECL_ALONE              STM_THREAD_T* TM_ARG_ALONE
// #    define TM_CALLABLE                   /* nothing */

#define STM_SELF         tinySTM_self
#define STM_THREAD_T     struct stm_tx

#define STM_STARTUP()               /* empty */
#define STM_NEW_THREADS(numThread)  if (sizeof(long) != sizeof(void *)) { \
                                      fprintf(stderr, "Error: unsupported long and pointer sizes\n"); \
                                      exit(1); \
                                    } \
                                    stm_init();
#define STM_SHUTDOWN()              stm_exit()

#define STM_NEW_THREAD()            stm_init_thread()
#define STM_INIT_THREAD(_arg, _id)  /* empty */
#define STM_FREE_THREAD(_arg)       stm_exit_thread()
#define STM_MALLOC(size)            malloc(size)
#define STM_FREE(ptr)               free(ptr)

#define STM_BEGIN_WR()              do { \
                                      stm_tx_attr_t _a = {{.read_only = 0}}; \
                                      sigjmp_buf _e; \
                                      stm_start(_a, &_e); \
                                      if (_e != NULL) sigsetjmp(_e, 0); \
                                    } while (0)
#define STM_BEGIN_RD()              do { \
                                      stm_tx_attr_t _a = {{.read_only = 1}}; \
                                      sigjmp_buf _e; \
                                      stm_start(_a, &_e); \
                                      if (_e != NULL) sigsetjmp(_e, 0); \
                                    } while (0)
#define STM_END()                   stm_commit()
#define STM_RESTART()               stm_abort(0)

#  include <wrappers.h>

#  define STM_READ(var)           stm_load((volatile stm_word_t *)(void *)&(var))
#  define STM_READ_P(var)         stm_load_ptr((volatile void **)(void *)&(var))
#  define STM_READ_D(var)         stm_load_double((volatile float *)(void *)&(var))
#  define STM_READ_F(var)         stm_load_float((volatile float *)(void *)&(var))

#  define STM_WRITE(var, val)     stm_store((volatile stm_word_t *)(void *)&(var), (stm_word_t)val)
#  define STM_WRITE_P(var, val)   stm_store_ptr((volatile void **)(void *)&(var), val)
#  define STM_WRITE_D(var, val)   stm_store_double((volatile float *)(void *)&(var), val)
#  define STM_WRITE_F(var, val)   stm_store_float((volatile float *)(void *)&(var), val)

#  define STM_LOCAL_WRITE(var, val)      ({var = val; var;})
#  define STM_LOCAL_WRITE_P(var, val)    ({var = val; var;})
#  define STM_LOCAL_WRITE_D(var, val)    ({var = val; var;})
#  define STM_LOCAL_WRITE_F(var, val)    ({var = val; var;})

#endif /* STM_H */

