#ifndef CP_H_GUARD
#define CP_H_GUARD

#include <stdlib.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

typedef struct prod_cons_ prod_cons_s;

typedef void(*prod_cons_req_fn)(void*);

typedef struct prod_cons_async_req_ {
  void *args;
  prod_cons_req_fn fn;
} prod_cons_async_req_s;

#define thread_create_or_die(thr, attr, callback, arg) \
  if (pthread_create(thr, attr, callback, (void *)(arg)) != 0) { \
    fprintf(stderr, "Error creating thread at " __FILE__ ":%i\n", __LINE__); \
    exit(EXIT_FAILURE); \
  } \
//

#define thread_join_or_die(thr, res) \
  if (pthread_join(thr, res)) { \
    fprintf(stderr, "Error joining thread at " __FILE__ ":%i\n", __LINE__); \
    exit(EXIT_FAILURE); \
  } \
//

// Only work for base 2 nb_items
#define PROD_CONS_ATOMIC_INC_PTR(ptr, nb_items) ({ \
	__typeof__(ptr) readPtr;\
	readPtr = __sync_fetch_and_add(&(ptr), 1); \
	readPtr = LOG_MOD2(readPtr, nb_items); \
	readPtr; \
})

// Only works with mod base 2
#define LOG_MOD2(idx, mod)     ((long long)(idx) & ((mod) - 1))
#define PTR_MOD(ptr, inc, mod) LOG_MOD2((long long)ptr + (long long)inc, mod)
#define BUFFER_ADDR(pc, ptr)   &(pc->buffer[ptr])
#define IS_EMPTY(pc)           (pc->c_ptr == pc->p_ptr)
#define IS_FULL(pc)            (PTR_MOD(pc->c_ptr, -pc->p_ptr, pc->nb_items) == 1)
#define CAS(ptr, old, new)     __sync_bool_compare_and_swap(ptr, old, new)

#define malloc_or_die(var, nb) \
if (((var) = (__typeof__((var)))malloc((nb) * sizeof(__typeof__(*(var))))) == NULL) { \
  fprintf(stderr, "malloc error \"%s\" at " __FILE__":%i\n", \
  strerror(errno), __LINE__); \
  exit(EXIT_FAILURE); \
} \

/**
 * Allocates a producer-consumer buffer.
 * nb_items MUST be a power 2.
 */
prod_cons_s* prod_cons_init(size_t nb_item);

/**
 * Teardown the producer-consumer buffer.
 */
void prod_cons_destroy(prod_cons_s*);

/**
 * Adds the item pointer to the consumer-producer buffer.
 *
 * Blocks if buffer is full, returns the position in the buffer.
 * IMPORTANT: item CANNOT be NULL!
 */
long prod_cons_produce(prod_cons_s*, void *item);

/**
 * Removes the last item pointer from the consumer-producer buffer.
 *
 * Blocks if buffer is empty, returns the position in the buffer.
 */
long prod_cons_consume(prod_cons_s*, void **item);

int prod_cons_count_items(prod_cons_s*);
int prod_cons_is_full(prod_cons_s*);
int prod_cons_is_empty(prod_cons_s*);

int prod_cons_start_thread();
int prod_cons_join_thread();
void prod_cons_async_request(prod_cons_async_req_s req);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* CP_H_GUARD */
