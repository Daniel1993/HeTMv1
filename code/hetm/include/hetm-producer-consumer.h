#ifndef CP_H_GUARD
#define CP_H_GUARD

#include <stdlib.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

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

typedef struct hetm_producer_consumer_ hetm_pc_s;

// TODO: should only work for base 2 nb_items (is this better?)
#define HETM_PC_ATOMIC_INC_PTR(ptr, nb_items) ({ \
	__typeof__(ptr) readPtr;\
	readPtr = __sync_fetch_and_add(&(ptr), 1); \
	readPtr = LOG_MOD2(readPtr, nb_items); \
	readPtr; \
})

/**
 * Allocates a producer-consumer buffer.
 * nb_items MUST be a power 2.
 */
hetm_pc_s* hetm_pc_init(size_t nb_item);

/**
 * Teardown the producer-consumer buffer.
 */
void hetm_pc_destroy(hetm_pc_s*);

/**
 * Adds the item pointer to the consumer-producer buffer.
 *
 * Blocks if buffer is full, returns the position in the buffer.
 * IMPORTANT: item CANNOT be NULL!
 */
long hetm_pc_produce(hetm_pc_s*, void *item);

/**
 * Removes the last item pointer from the consumer-producer buffer.
 *
 * Blocks if buffer is empty, returns the position in the buffer.
 */
long hetm_pc_consume(hetm_pc_s*, void **item);

int hetm_pc_count_items(hetm_pc_s*);
int hetm_pc_is_full(hetm_pc_s*);
int hetm_pc_is_empty(hetm_pc_s*);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* CP_H_GUARD */
