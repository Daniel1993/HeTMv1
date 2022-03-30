#ifndef BB_LIB_H_GUARD
#define BB_LIB_H_GUARD

#ifdef __cplusplus
extern "C" {
#endif

#include "graph.h"

#include <time.h>

// typedef struct graph_ *graph;
// typedef struct state_ *state;

struct GREEDY_ {
  graph G; /* Current graph */
  int *P; /* Copy of the best solution */

  struct timespec startTime;
  unsigned long iter;
  int *fvs;
  int *curr_fvs;
};

typedef struct GREEDY_ *GREEDY_s;

typedef struct GREEDY_parameters_ {
  int none;
} GREEDY_parameters_s;

void // also stores internally
GREEDY_init_G(GREEDY_parameters_s, graph);

void
GREEDY_init_F(GREEDY_parameters_s, const char *filename);

void
GREEDY_run();

int*  // loop with while(-1 != *bestSolution)
GREEDY_getBestSolution();

void
GREEDY_destroy();

#ifdef __cplusplus
}
#endif

#endif /* BB_LIB_H_GUARD */
