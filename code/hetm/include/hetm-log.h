#ifndef HETM_LOG_H_GUARD_
#define HETM_LOG_H_GUARD_

#include <stdio.h>
#include <stdlib.h>
// #include "pr-stm-wrapper.cuh"

#include "chunked-log.h"

// TODO: these don't go through
#define HETM_VERS_LOG  1
#define HETM_ADDR_LOG  2 // not used anymore
#define HETM_BMAP_LOG  3
#define HETM_VERS2_LOG 4 // not used anymore

#ifndef LOG_SIZE
// #define LOG_SIZE             131072
#define LOG_SIZE             32768 // 32768 // 65536 // 131072 /* Initial log size */
#endif /* LOG_SIZE */

#ifndef STM_LOG_BUFFER_SIZE
// #define STM_LOG_BUFFER_SIZE  1
#define STM_LOG_BUFFER_SIZE  8 // 4 /* buffer capacity (times LOG_SIZE) */
#endif /* STM_LOG_BUFFER_SIZE */

// CPU rs cache
#ifndef BMAP_GRAN_BITS
#define BMAP_GRAN_BITS (16) // 1kB --> then I use a smart copy
#endif /* BMAP_GRAN_BITS */

#ifndef VALUE_GRAN_BITS
#define VALUE_GRAN_BITS (2) // 4B
#endif /* BMAP_GRAN_BITS */

#define BMAP_GRAN (0x1<<BMAP_GRAN_BITS)
#define BitMAP_GRAN (0x1<<(BMAP_GRAN_BITS>>8))
#define CHUNK_GRAN (BMAP_GRAN<<VALUE_GRAN_BITS)

#ifndef VALUE_GRAN_BITS
#define VALUE_GRAN_BITS (2) // 4B
#endif /* BMAP_GRAN_BITS */



// #define BMAP_GRAN_BITS (12) // 4kB --> then I use a smart copy
// #define BMAP_GRAN (0x1<<12)
// #define BMAP_GRAN_BITS (14) // 16kB --> then I use a smart copy
// #define BMAP_GRAN (0x1<<14)

#ifndef HETM_LOG_TYPE
#define HETM_LOG_TYPE HETM_VERS_LOG
#endif /* HETM_LOG_TYPE */

#define HETM_LOG_T chunked_log_s

#ifndef HETM_NB_DEVICES
#define HETM_NB_DEVICES 2
#endif /* HETM_NB_DEVICES */

extern __thread HETM_LOG_T *stm_thread_local_log;

// extern void *stm_devMemPoolBackupBmap[HETM_NB_DEVICES];
// extern void *stm_devMemPoolBmap[HETM_NB_DEVICES];
extern void *stm_baseMemPool[HETM_NB_DEVICES];
extern void *stm_rsetCPU;
extern void *stm_rsetCPUCache;
extern void *stm_wsetCPU; // only for HETM_BMAP_LOG
extern void *stm_wsetCPUCache; // only for HETM_BMAP_LOG
extern void *stm_wsetCPUCacheConfl; // only for HETM_BMAP_LOG
extern void *stm_wsetCPUCacheConfl3; // only for HETM_BMAP_LOG

// TODO: what is this?
extern void *stm_wsetCPUCache_x64[HETM_NB_DEVICES]; // bytes are in different cache lines

extern size_t stm_wsetCPUCacheBits; // only for HETM_BMAP_LOG
extern size_t stm_rsetCPUCacheBits;
extern long *hetm_batchCount;

/* ################################################################### *
 * inline FUNCTIONS
 * ################################################################### */

static inline size_t stm_log_size(chunked_log_node_s *node)
{
  size_t res = 0;
  chunked_log_node_s *tmp = node;

  if (tmp == NULL) return res;

  while (tmp->next != NULL) {
    res++;
    tmp = tmp->next;
  }

  return res;
}

static inline void stm_log_truncate(HETM_LOG_T *log, int *nbChunks)
{
  CHUNKED_LOG_TRUNCATE(0, log, STM_LOG_BUFFER_SIZE, nbChunks);
}

static inline void stm_log_node_free(chunked_log_node_s *node)
{
  CHUNKED_LOG_FREE(node); // recycles
}

static inline void stm_log_free(HETM_LOG_T *log)
{
  if (log != NULL) {
    CHUNKED_LOG_DESTROY(0, log);
  }
  // CHUNKED_LOG_TEARDOWN(); // TODO: implement thread-local chunk_nodes
  stm_thread_local_log = NULL;
}

#ifdef __cplusplus
extern "C" {
#endif

/* Init log: must be called per thread (thread local log) */
/*static inline*/ HETM_LOG_T* stm_log_init();

// TODO: keep this inline for performance reasons
/* Add value to log */
/*static inline*/ void
stm_log_newentry(HETM_LOG_T *log, long* pos, int val, long vers);

/* Add value to log */
/*static inline*/ void
stm_log_read_entry(long* pos);

#ifdef __cplusplus
}
#endif

#endif /* HETM_LOG_H_GUARD_ */
