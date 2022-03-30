#ifndef CHUNKED_LOG_H_GUARD_
#define CHUNKED_LOG_H_GUARD_

/**
 * Chunked log library.
 *
 * This is a thread local library that aims in providing a very efficient log
 * implementation.
 */

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "chunked-log-aux.h"
#include <assert.h>
#include <errno.h>

#ifndef HETM_NB_DEVICES
#define HETM_NB_DEVICES 2
#endif

#ifndef malloc_or_die
#define malloc_or_die(var, nb) \
if (((var) = (__typeof__((var)))malloc((nb) * sizeof(__typeof__(*(var))))) == NULL) { \
  fprintf(stderr, "malloc error \"%s\" at " __FILE__":%i\n", \
  strerror(errno), __LINE__); \
  exit(EXIT_FAILURE); \
}
#endif /* malloc_or_die */


typedef struct chunked_log_node_ {
  char* volatile chunk; // assuming sizeof(char) == 1 Byte (8 bits)
  size_t size; // total size in Bytes of the chunk
  size_t gran; // granularity of the elements in Bytes
  size_t nb_buckets;
  union {
    size_t pos;  // current position in Bytes
    size_t *posArray; // current position in Bytes (mod)
  } p;
  struct chunked_log_node_ *next, *prev;
} chunked_log_node_s;

typedef struct truncated_chunked_log_node_ {
  chunked_log_node_s *curr;
  struct truncated_chunked_log_node_ *next, *prev;
} truncated_chunked_log_node_s;

typedef struct chunked_log_ {
  size_t size; // total number of nodes
  size_t gran; // granularity of the elements in Bytes
  size_t nb_elements; // number of elements with a given granularity
  size_t pos;  // position of the current (curr) node
  int tid;
  chunked_log_node_s *first, *curr, *last;
  chunked_log_node_s *nextTruncated[HETM_NB_DEVICES];
  truncated_chunked_log_node_s *trunc[HETM_NB_DEVICES], *truncLast[HETM_NB_DEVICES];
} chunked_log_s;

typedef struct mod_chunked_log_ {
  size_t nb_buckets; // total number of buckets
  size_t gran; // granularity of the elements in Bytes
  size_t nb_elements; // number of elements with a given granularity
  chunked_log_node_s *buckets, *bucketsEnd;
} mod_chunked_log_s;

/**
 * Thread local pool of log nodes.
 *
 * Each time a thread frees a node, rather than releasing the memory, the node
 * is placed here. Upon alloc, the system checks if there are freed nodes.
 */
#define SIZE_OF_FREE_NODES 0x1000L // needs to be REALLY big!

// TODO: cannot use __thread because the thread that FREEs is different!
// use some sort of array[nb_threads]
extern __thread chunked_log_node_s *chunked_log_node_recycled[SIZE_OF_FREE_NODES];
extern __thread unsigned long chunked_log_free_ptr;
extern __thread unsigned long chunked_log_alloc_ptr;
extern __thread unsigned long chunked_log_free_r_ptr;
extern __thread unsigned long chunked_log_alloc_r_ptr;

#define CHUNKED_LOG_INIT_AUX(log, granularity, nbElements) ({ \
  (log)->gran = granularity; \
  (log)->nb_elements = nbElements; \
})

#define CHUNKED_LOG_INIT(log, granularity, nbElements) ({ \
  CHUNKED_LOG_INIT_AUX(log, granularity, nbElements); \
  for (int j = 0; j < HETM_NB_DEVICES; ++j) { \
    (log)->trunc[j] = 0; \
    (log)->truncLast[j] = 0; \
    (log)->nextTruncated[j] = NULL; \
  } \
  (log)->first = (log)->last = (log)->curr = NULL; \
  (log)->size = 0; \
  (log)->pos = 0; \
})

#define MOD_CHUNKED_LOG_INIT(log, granularity, nbElements, nbBuckets) ({ \
  CHUNKED_LOG_INIT_AUX(log, granularity, nbElements); \
  (log)->nb_buckets = nbBuckets; \
  (log)->buckets = NULL; \
  (log)->bucketsEnd = NULL; \
})

// TODO: if out of space to free --> this will crash
// TODO: just use mallocs is much faster
// static inline void CHUNKED_LOG_FREE(chunked_log_node_s *node)
// {
//   if (node->nb_buckets > 0){
//     free(node->p.posArray);
//   }
//   free(node->chunk);
//   free(node);
// }
#define CHUNKED_LOG_FREE(node) ({ \
  /*unsigned long i = __sync_fetch_and_add(&chunked_log_free_r_ptr, 1); */\
  assert(node != NULL); \
  for (uintptr_t j = chunked_log_alloc_ptr; j < (uintptr_t)chunked_log_free_ptr; ++j) \
    if (node == chunked_log_node_recycled[j % SIZE_OF_FREE_NODES]) assert(0); \
  unsigned long i = chunked_log_free_ptr; \
  chunked_log_node_recycled[i % SIZE_OF_FREE_NODES] = node; /* barrier needed? */ \
  node->next = node->prev = NULL; /* TODO: this is fixing some bug in VERS */ \
  chunked_log_free_ptr++; \
  /* -------------------- */ \
  /*if (node->nb_buckets > 0) free(node->p.posArray); \
  free(node->chunk); \
  free(node);*/ \
}) \

/**
 * Destroy all thread local nodes.
 */
 // static inline chunked_log_node_s* CHUNKED_LOG_POP(chunked_log_s *log);
 // static inline void CHUNKED_LOG_DESTROY(chunked_log_s *log)
 // {
 //   chunked_log_node_s *node = CHUNKED_LOG_POP(log);
 //   while (node != NULL) {
 //     CHUNKED_LOG_FREE(node);
 //     node = CHUNKED_LOG_POP(log);
 //   }
 //   (log)->curr = (log)->last = (log)->first = NULL;
 //   (log)->size = 0;
 // }
#define CHUNKED_LOG_CURR_TRUNC(devId, log) ({ \
  (log)->trunc[devId]; \
})

#define CHUNKED_LOG_DESTROY_TRUNC(devId, log) ({ \
  truncated_chunked_log_node_s *_node = (log)->trunc[devId]; \
  while(_node != NULL) { \
    _node = _node->next; \
    free((log)->trunc[devId]); \
    (log)->trunc[devId] = _node; \
  } \
  (log)->trunc[devId] = (log)->truncLast[devId] = NULL; \
  if ((log)->nextTruncated[devId] != NULL && (log)->nextTruncated[devId]->next == NULL \
      && (log)->nextTruncated[devId] != (log)->last) { \
    (log)->nextTruncated[devId] = (log)->last; /*BUG*/ \
  } \
})

#define CHUNKED_LOG_DESTROY(devId, log) ({ \
  chunked_log_node_s *node = CHUNKED_LOG_POP(log); \
  CHUNKED_LOG_DESTROY_TRUNC(devId, log); \
  while (node != NULL) { \
    CHUNKED_LOG_FREE(node); \
    node = CHUNKED_LOG_POP(log); \
  } \
  (log)->nextTruncated[devId] = (log)->curr = (log)->last = (log)->first = NULL; \
  (log)->size = 0; \
})

#define MOD_CHUNKED_LOG_DESTROY(log) ({ \
  chunked_log_node_s *node = (log)->buckets; \
  while ((log)->buckets != NULL) { \
    node = (log)->buckets; \
    (log)->buckets = node->next; \
    CHUNKED_LOG_FREE(node); \
  } \
  (log)->buckets = (log)->bucketsEnd = NULL; \
})

// TODO: threads are still freeing the logs while this runs
#define CHUNKED_LOG_TEARDOWN() ({ \
  unsigned long k = chunked_log_free_ptr; \
  unsigned long i = chunked_log_alloc_ptr; \
  while (i < k || i > k - (SIZE_OF_FREE_NODES+1)) { \
    chunked_log_node_s *node = chunked_log_node_recycled[i % SIZE_OF_FREE_NODES]; \
    if (node->nb_buckets > 0) free(node->p.posArray); \
    free(node->chunk); \
    free(node); \
    ++i; \
  } \
  chunked_log_free_ptr = chunked_log_alloc_ptr = 0; \
  __sync_synchronize(); \
})

/**
 * Truncates a portion of the log.
 */
#define CHUNKED_LOG_TRUNCATE(_devId, _log, _nb_chunks, _nbChunks) ({ \
  chunked_log_node_s *_Cnode; \
  int i; \
  _Cnode = CHUNKED_LOG_TRUNC_PREPARE(_devId, (_log)); \
  for (i = 0; i < _nb_chunks; ++i) { \
    if (_Cnode == NULL) { /* no more available */ \
      break; \
    } \
    _Cnode = _Cnode->next; /* TODO: this may cause SIGSEGV !!! */ \
    CHUNKED_LOG_TRUNC_EXTEND(_devId, (_log)); \
  } \
  *_nbChunks = i; /* this may be different from _nb_chunks */ \
})

/**
 * Truncates a portion of the log. TODO: use truncate
 */
#define MOD_CHUNKED_LOG_TRUNCATE(log, nb_chunks) ({ \
  CHUNKED_LOG_LOCAL_INST(res); \
  int i; \
  for (i = 0; i < nb_chunks; ++i) { \
    if ((log)->buckets == NULL) { \
      break; \
    } \
    (log)->buckets->prev = NULL; /* invariant */ \
    chunked_log_node_s *node = (log)->buckets; \
    (log)->buckets = (log)->buckets->next; \
    node->next = NULL; \
    CHUNKED_LOG_EXTEND(&res, node); \
  } \
  if ((log)->buckets == NULL) { \
    (log)->bucketsEnd = NULL; \
  } \
  res; \
})

#define CHUNKED_LOG_IS_EMPTY(log) ({ \
  chunked_log_node_s* volatile first = (log)->first; \
  ((log)->size == 0 || ((log)->size == 1 && first != NULL && first->p.pos == 0)); \
})

#define MOD_CHUNKED_LOG_IS_EMPTY(log) ({ \
  int res = 1; \
  int i; \
  for (i = 0; i < (log)->nb_buckets; ++i) { \
    if ((log)->buckets != NULL) { \
      if ((log)->buckets->p.posArray[i] != 0) { \
        res = 0; \
        break; \
      } \
    } \
  } \
  res; \
})

#define MOD_CHUNKED_LOG_NODE_FREE_SPACE(node) ({ \
  int res = 0; \
  int i; \
  for (i = 0; i < node->nb_buckets; ++i) { \
    res += node->size - node->p.posArray[i]; \
  } \
  res; \
})

#define MOD_CHUNKED_LOG_IS_FLAT(log) ({ \
  int res = 1; \
  int i; \
  for (i = 0; i < (log)->nb_buckets; ++i) { \
    if ((log)->buckets != NULL && (log)->buckets != (log)->bucketsEnd) { \
      res = 0; \
      break; \
    } \
  } \
  res; \
})

/**
 * Allocs a new node with a given chunk size.
 */
#define CHUNKED_LOG_ALLOC(size_one_element, nb_elements) ({ \
  size_t size = size_one_element*nb_elements; \
  chunked_log_node_s *res = CHUNKED_LOG_FIND_FREE(size, 0); \
  if (res == NULL) { \
    malloc_or_die(res, sizeof(chunked_log_node_s)); \
    malloc_or_die(res->chunk, size); \
  } \
  res->nb_buckets = 0; \
  res->gran  = size_one_element; \
  res->size  = size; \
  res->p.pos = 0; \
  res->next  = res->prev = NULL; \
  res; \
})

/**
 * Allocs a new node with a given chunk size.
 */
#define MOD_CHUNKED_LOG_ALLOC(size_one_element, nb_elements, nbBuckets) ({ \
  size_t size = size_one_element*nb_elements; \
  /* TODO: must look for the same number of buckets also */ \
  chunked_log_node_s *res = CHUNKED_LOG_FIND_FREE(size, nbBuckets); \
  if (res == NULL) { \
    malloc_or_die(res, sizeof(chunked_log_node_s)); \
    malloc_or_die(res->chunk, size*nbBuckets); \
    malloc_or_die(res->p.posArray, sizeof(size_t)*nbBuckets); \
  } \
  memset(res->p.posArray, 0, sizeof(size_t)*nbBuckets); \
  memset(res->chunk, 0, size*nbBuckets); \
  res->nb_buckets = nbBuckets; \
  res->gran = size_one_element; \
  res->size = size; \
  res->next = res->prev = NULL; \
  res; \
})

// static inline chunked_log_node_s* CHUNKED_LOG_APPEND_AUX(chunked_log_s *log, chunked_log_node_s *node, void *el)
// {
//   chunked_log_node_s *newNode = node;
//   void *addr = node->chunk + node->p.pos;
//   memcpy(addr, el, node->gran);
//   node->p.pos += node->gran;
//   if (node->p.pos >= node->size) {
//     newNode = CHUNKED_LOG_ALLOC((log)->gran, (log)->nb_elements);
//     newNode->prev = node;
//     node->next = newNode;
//   }
//   return newNode;
// }
#define CHUNKED_LOG_APPEND_AUX(log, node, el) ({ \
  chunked_log_node_s *newNode = node; \
  void *addr = node->chunk + node->p.pos; \
  memcpy(addr, el, node->gran); \
  node->p.pos += node->gran; \
  if (node->p.pos >= node->size) { \
    newNode = CHUNKED_LOG_ALLOC((log)->gran, (log)->nb_elements); \
    newNode->prev = node; \
    node->next = newNode; \
  } \
  newNode; \
})

#define MOD_CHUNKED_LOG_APPEND_AUX(log, node, el, bucket) ({ \
  chunked_log_node_s *newNode = node; \
  size_t bucketPos = bucket*node->gran; \
  size_t addrPos = node->p.posArray[bucket]*node->nb_buckets + bucketPos; \
  void *addr = (char*)node->chunk + addrPos; \
  memcpy(addr, el, node->gran); \
  node->p.posArray[bucket] += node->gran; \
  if (node->p.posArray[bucket] >= node->size && node->next == NULL) { \
    newNode = MOD_CHUNKED_LOG_ALLOC((log)->gran, (log)->nb_elements, node->nb_buckets); \
    newNode->prev = node; \
    newNode->next = NULL; \
    node->next = newNode; \
  } \
  newNode; \
})

/**
 * Appends an element in the log. The element must be a pointer.
 */
// static inline void CHUNKED_LOG_APPEND(chunked_log_s *log, void *el)
// {
//   chunked_log_node_s *node = (log)->last;
//   if (node == NULL) {
//     node = CHUNKED_LOG_ALLOC((log)->gran, (log)->nb_elements);
//     CHUNKED_LOG_EXTEND((log), node);
//   }
//   chunked_log_node_s *newNode = CHUNKED_LOG_APPEND_AUX(log, node, el);
//   if (node != newNode) {
//     CHUNKED_LOG_EXTEND((log), newNode);
//   }
// }
#define CHUNKED_LOG_APPEND(log, el) ({ \
  chunked_log_node_s *node = (log)->last; \
  if (node == NULL) { \
    node = CHUNKED_LOG_ALLOC((log)->gran, (log)->nb_elements); \
    CHUNKED_LOG_EXTEND((log), node); \
  } \
  chunked_log_node_s *newNode = CHUNKED_LOG_APPEND_AUX(log, node, el); \
  if (node != newNode) { \
    CHUNKED_LOG_EXTEND((log), newNode); \
  } \
})

#define CHUNKED_LOG_EXTEND_FORCE(log) ({ \
  chunked_log_node_s *node = CHUNKED_LOG_ALLOC((log)->gran, (log)->nb_elements); \
  CHUNKED_LOG_EXTEND((log), node); \
})

// TODO: FIXME --> (log)->buckets != node && node->prev != NULL
// means that a prev pointer is not being init to ZERO properly
#define MOD_CHUNKED_LOG_APPEND(log, el, addr) ({ \
  uintptr_t pos = (uintptr_t)(addr) % (log)->nb_buckets; /* TODO: use mod2 */ \
  chunked_log_node_s *node = (log)->bucketsEnd; \
  if (node == NULL) { \
    node = MOD_CHUNKED_LOG_ALLOC((log)->gran, (log)->nb_elements, (log)->nb_buckets); \
    (log)->buckets = (log)->bucketsEnd = node; \
  } \
  while ((log)->buckets != node && node->prev != NULL && node->p.posArray[pos] == 0) { \
    /* check if previous has space */ \
    if (node->prev->p.posArray[pos] < node->prev->size) { \
      node = node->prev; \
    } else { \
      break; \
    } \
  } \
  chunked_log_node_s *newNode = MOD_CHUNKED_LOG_APPEND_AUX(log, node, el, pos); \
  if (newNode != node) { \
    newNode->prev = node; \
    node->next = newNode; \
    (log)->bucketsEnd = newNode; \
  } \
})

/**
 * Removes the first node in the log. Does not free.
 */
// static inline chunked_log_node_s* CHUNKED_LOG_POP(chunked_log_s *log)
// {
//   chunked_log_node_s *node = log->first;
//   if (node != NULL) {
//     CHUNKED_LOG_REMOVE_FRONT(log);
//     if (log->size > 0) log->size--;
//   }
//   return node;
// }
#define CHUNKED_LOG_POP(log) ({ \
  chunked_log_node_s *node = (log)->first; \
  if (node != NULL) { \
    CHUNKED_LOG_REMOVE_FRONT(log); \
    if ((log)->size > 0) (log)->size--; \
  } \
  node; \
})

#endif /* CHUNKED_LOG_H_GUARD_ */
