#include "hetm-utils.h"
#include "hetm-producer-consumer.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>

// TODO: only works with modulo base 2
#define LOG_MOD2(idx, mod)     ((long long)(idx) & ((mod) - 1))
#define PTR_MOD(ptr, inc, mod) LOG_MOD2((long long)ptr + (long long)inc, mod)
#define BUFFER_ADDR(pc, ptr)   &(pc->buffer[ptr])
#define IS_EMPTY(pc)           (pc->c_ptr == pc->p_ptr)
#define IS_FULL(pc)            (PTR_MOD(pc->c_ptr, -pc->p_ptr, pc->nb_items) == 1)
#define CAS(ptr, old, new)     __sync_bool_compare_and_swap(ptr, old, new)

// TODO: allow more than one buffer
struct hetm_producer_consumer_ {
	void      **buffer;
	long long   c_ptr, p_ptr;
	semaphore_t c_sem, p_sem;
	size_t      nb_items;
};

hetm_pc_s* hetm_pc_init(size_t nb_items)
{
	hetm_pc_s *res;
	size_t nbItems = nb_items;

	if (__builtin_popcountll(nb_items) != 1) {
		// not power 2 (more than 1 bit set to 1)
		nbItems = 1 << __builtin_lroundf(__builtin_ceill(__builtin_log2l(nb_items)));
	}

	malloc_or_die(res, 1);
	malloc_or_die(res->buffer, nbItems);
	memset(res->buffer, 0, nbItems*sizeof(void*));
	semaphore_init(res->c_sem, 0); // consumer blocks at the begining
	semaphore_init(res->p_sem, nbItems); // producer is free
	res->nb_items = nbItems;
	res->c_ptr = 0;
	res->p_ptr = 0;
	return res;
}

void hetm_pc_destroy(hetm_pc_s *pc)
{
	semaphore_destroy(pc->c_sem);
	semaphore_destroy(pc->p_sem);
	free(pc->buffer);
	free(pc);
}

long hetm_pc_produce(hetm_pc_s *pc, void *i)
{
	long readPtr;
	if (i == NULL) return -1; // error
	semaphore_wait(pc->p_sem);
	readPtr = HETM_PC_ATOMIC_INC_PTR(pc->p_ptr, pc->nb_items);
	COMPILER_FENCE();
	while (*BUFFER_ADDR(pc, readPtr) != NULL) pthread_yield();
	*BUFFER_ADDR(pc, readPtr) = i;
	semaphore_post(pc->c_sem); // memory barrier
	return readPtr;
}

long hetm_pc_consume(hetm_pc_s *pc, void **i)
{
	long readPtr;
	semaphore_wait(pc->c_sem);
	readPtr = HETM_PC_ATOMIC_INC_PTR(pc->c_ptr, pc->nb_items);
	COMPILER_FENCE();
	while ((*i = *BUFFER_ADDR(pc, readPtr)) == NULL) pthread_yield();
	*BUFFER_ADDR(pc, readPtr) = NULL; // reset value
	semaphore_post(pc->p_sem); // memory barrier
	return readPtr;
}

int hetm_pc_count_items(hetm_pc_s *pc) {
	return PTR_MOD(pc->p_ptr, -pc->c_ptr, pc->nb_items); // TODO
}

int hetm_pc_is_full(hetm_pc_s *pc) {
	return IS_FULL(pc);
}

int hetm_pc_is_empty(hetm_pc_s *pc) {
	return IS_EMPTY(pc);
}
