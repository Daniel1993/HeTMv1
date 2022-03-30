#ifndef HETM_UTILS_H_GUARD_
#define HETM_UTILS_H_GUARD_

// ------------- TODO: don't like this API for the malloc
#define malloc_or_die(var, nb) \
if (((var) = (__typeof__((var)))malloc((nb) * sizeof(__typeof__(*(var))))) == NULL) { \
  fprintf(stderr, "malloc error \"%s\" at " __FILE__":%i\n", \
  strerror(errno), __LINE__); \
  exit(EXIT_FAILURE); \
} \
//
// -------------

/* ################################################################### *
 * BARRIER
 * ################################################################### */
#include <pthread.h> // on barrier.h

#include "ticket-barrier.h"

#define barrier_t          ticket_barrier_t
#define barrier_init(b, n) ticket_barrier_init(&b, (unsigned)n)
#define barrier_destroy(b) ticket_barrier_destroy(&b)
#define barrier_cross(b)   ticket_barrier_cross(&b)
#define barrier_reset(b)   ticket_barrier_reset(&b)

#define COMPILER_FENCE() asm("" ::: "memory")

/* ################################################################### *
 * SEMAPHORES
 * ################################################################### */
#include <semaphore.h> // linux only
#define semaphore_t           sem_t
#define semaphore_init(s, n)  sem_init(&s, 0, n)
#define semaphore_destroy(s)  sem_destroy(&s)
#define semaphore_post(s)     sem_post(&s)
#define semaphore_wait(s)     sem_wait(&s)
/* ################################################################### */

#ifdef USE_NVTX
#include "nvToolsExt.h"

const static uint32_t NVTX_colors[] = {
  0x0000ff00,
  0x000000ff,
  0x00ffff00,
  0x00ff00ff,
  0x0000ffff,
  0x00ff0000,
  0x00cccccc,
  0x00888888,
  0x00444444,
  0x00111111,
  0x001111cc,
  0x0011cc11,
  0x00cc1111,
  0x00cc11cc
};
const static int NVTX_num_colors = sizeof(NVTX_colors)/sizeof(uint32_t);

// TODO: do the same to cudaEvent
#define CUDA_EVENT_RECORD(ev, stream)        if (ev) { CUDA_CHECK_ERROR(cudaEventRecord(ev, stream), ""); }
#define CUDA_EVENT_ELAPSED_TIME(reg, e1, e2) if (e1 && e2) { CUDA_CHECK_ERROR(cudaEventElapsedTime(reg, e1, e2), ""); } else { *(reg) = 0; }
#define CUDA_EVENT_SYNCHRONIZE(ev)           if (ev) { CUDA_CHECK_ERROR(cudaEventSynchronize(ev), ""); }

#define NVTX_PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%NVTX_num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = NVTX_colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define NVTX_POP_RANGE() nvtxRangePop();
#else
#define CUDA_EVENT_RECORD(ev, stream)        /* empty */
#define CUDA_EVENT_ELAPSED_TIME(reg, e1, e2) /* empty */
#define CUDA_EVENT_SYNCHRONIZE(ev)           /* empty */
#define NVTX_PUSH_RANGE(name,cid)            /* empty */
#define NVTX_POP_RANGE()                     /* empty */
#endif

#endif /* HETM_UTILS_H_GUARD_ */
