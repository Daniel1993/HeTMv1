#include "prod-cons.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <semaphore.h>

#define semaphore_t           sem_t
#define semaphore_init(s, n)  sem_init(&s, 0, n)
#define semaphore_destroy(s)  sem_destroy(&s)
#define semaphore_post(s)     sem_post(&s)
#define semaphore_wait(s)     sem_wait(&s)

// TODO: allow more than one buffer
struct prod_cons_ {
	void      **buffer;
	long long   c_ptr, p_ptr;
	semaphore_t c_sem, p_sem;
	size_t      nb_items;
};

prod_cons_s* prod_cons_init(size_t nb_items)
{
	prod_cons_s *res;
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

void prod_cons_destroy(prod_cons_s *pc)
{
	semaphore_destroy(pc->c_sem);
	semaphore_destroy(pc->p_sem);
	free(pc->buffer);
	free(pc);
}

long prod_cons_produce(prod_cons_s *pc, void *i)
{
	long readPtr;
	if (i == NULL) return -1; // error
	semaphore_wait(pc->p_sem);
	readPtr = PROD_CONS_ATOMIC_INC_PTR(pc->p_ptr, pc->nb_items);
	while (*BUFFER_ADDR(pc, readPtr) != NULL);
	*BUFFER_ADDR(pc, readPtr) = i;
	semaphore_post(pc->c_sem); // memory barrier
	return readPtr;
}

long prod_cons_consume(prod_cons_s *pc, void **i)
{
	long readPtr;
	semaphore_wait(pc->c_sem);
	readPtr = PROD_CONS_ATOMIC_INC_PTR(pc->c_ptr, pc->nb_items);
	while ((*i = *BUFFER_ADDR(pc, readPtr)) == NULL);
	*BUFFER_ADDR(pc, readPtr) = NULL; // reset value
	semaphore_post(pc->p_sem); // memory barrier
	return readPtr;
}

int prod_cons_count_items(prod_cons_s *pc) {
	return PTR_MOD(pc->p_ptr, -pc->c_ptr, pc->nb_items); // TODO
}

int prod_cons_is_full(prod_cons_s *pc) {
	return IS_FULL(pc);
}

int prod_cons_is_empty(prod_cons_s *pc) {
	return IS_EMPTY(pc);
}
