#ifndef STM_H_GUARD
#define STM_H_GUARD

#include <setjmp.h>

#define GRAPH_TM_INIT_SIZE 256
#define GRAPH_TM_FILENAME  "./graph_tm_%i_thrs/B%i_G%i.in"

#ifdef __cplusplus
extern "C" {
#endif

/* TM metadata */
#  define STM_INT          long
#  define STM_FLT          float
#  define STM_PTR          void*
typedef enum { 
  STM_NON_CODE = 0,
  STM_INT_CODE = 1,
  STM_FLT_CODE = 2,
  STM_PTR_CODE = 3,
  STM_FRE_CODE = 4 // mallocs are ok, but frees may cause invalid accesses
} STM_TYPE_CODE;

#  define STM_SELF         graphTM_self
#  define STM_THREAD_T     graphTM_thr

/* thread setup and memory allocation */
#  define STM_STARTUP()               stm_startup()
#  define STM_NEW_THREADS(numThread)  stm_new_threads(numThread)
#  define STM_SHUTDOWN()              stm_shutdown()
#  define STM_NEW_THREAD()            stm_new_thread()
#  define STM_INIT_THREAD(tmArg, tid) stm_init_thread(tmArg, tid)
#  define STM_FREE_THREAD(tmArg)      stm_free_thread(tmArg)
#  define STM_GET_THREAD()            stm_get_thread()
#  define STM_SET_SELF(tmArg)         stm_set_self(tmArg)
#  define STM_MALLOC(size)            stm_malloc((long unsigned)size)
#  define STM_FREE(ptr)               stm_free(ptr)

/* TX manipulation */
#  define STM_BEGIN_WR()              setjmp(graphTM_self->jumpEnv); stm_begin_wr()
#  define STM_BEGIN_RD()              setjmp(graphTM_self->jumpEnv); if (graphTM_self->isRO) stm_begin_rd(); else stm_begin_wr()
#  define STM_END()                   if (!stm_end()) siglongjmp(graphTM_self->jumpEnv, -1)
#  define STM_RESTART()               stm_restart(); siglongjmp(graphTM_self->jumpEnv, 1)

/* TX instrumentation */
#  define STM_READ(var)               stm_read((STM_INT*)&(var))
#  define STM_READ_P(var)             stm_read_P((STM_PTR*)&(var))
#  define STM_READ_F(var)             stm_read_F((STM_FLT*)&(var))

#  define STM_WRITE(var, val)         stm_write((STM_INT*)&(var), val)
#  define STM_WRITE_P(var, val)       stm_write_P((STM_PTR*)&(var), val)
#  define STM_WRITE_F(var, val)       stm_write_F((STM_FLT*)&(var), val)

// non-conflicting accesses
#  define STM_LOCAL_WRITE(var, val)   ((var) = (val))
#  define STM_LOCAL_WRITE_P(var, val) ((var) = (val))
#  define STM_LOCAL_WRITE_F(var, val) ((var) = (val))

typedef struct graphTM_g_ {
  int nbVertices;
  int nbEdges;
  int *edges[2];
} graphTM_g;

typedef union {
  STM_INT valInt;
  STM_FLT valFloat;
  STM_PTR valPtr;
} graphTM_typesSize;

typedef struct graphTM_rwset_entry_ {
  void *addr;
  STM_TYPE_CODE code;
  graphTM_typesSize v;
} graphTM_rwset_entry;

typedef struct graphTM_rwset_ {
  graphTM_rwset_entry *d;
  int size;
  int allocSize;
} graphTM_rwset;

typedef struct graphTM_ {
  graphTM_rwset *rs_entries; // 1 per thread
  graphTM_rwset *ws_entries;
  int nbThreads;
} graphTM;

typedef struct graphTM_thr_ {
  graphTM_rwset *rs_entries; // local rwset thread
  graphTM_rwset *ws_entries;
  int tid;
  int startedAsRO;
  int isRO;
  jmp_buf jumpEnv;
} graphTM_thr;

void stm_startup();
void stm_new_threads(int numThread);
void stm_shutdown();
STM_THREAD_T* stm_new_thread();
STM_THREAD_T* stm_get_thread();
void stm_set_self(STM_THREAD_T*);
void stm_init_thread(STM_THREAD_T*, int tid);
void stm_free_thread(STM_THREAD_T*);

void stm_begin_wr();
void stm_begin_rd();
int stm_end();
void stm_restart();

void *stm_malloc(long unsigned size);
void stm_free(void* addr);

int handleMatrixLine(int isRestart);

STM_INT stm_read(STM_INT*);
STM_FLT stm_read_F(STM_FLT*);
STM_PTR stm_read_P(STM_PTR*);

STM_INT stm_write(STM_INT*, STM_INT);
STM_FLT stm_write_F(STM_FLT*, STM_FLT);
STM_PTR stm_write_P(STM_PTR*, STM_PTR);

#ifdef __cplusplus
}
#endif

#endif /* STM_H_GUARD */