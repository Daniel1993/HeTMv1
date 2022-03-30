#ifndef BB_LIB_H_GUARD
#define BB_LIB_H_GUARD

#ifdef __cplusplus
extern "C" {
#endif

#include "graph.h"

#include <time.h>

struct GSORT_ {
  int batchSize; /* How many nodes to put into FVS at each iteration */

  graph G; /* Current graph */
  int *d; /* Scores according to policy */
  int *P; /* Copy of the best solution */
  int *trimmed;
  int procVerts;

  struct timespec startTime;
};

typedef struct GSORT_ *GSORT_s;

typedef struct GSORT_parameters_ {
  int batchSize; /* How many nodes to put into FVS at each iteration */
} GSORT_parameters_s;

void // also stores internally
GSORT_init_G(GSORT_parameters_s, graph);

void
GSORT_init_F(GSORT_parameters_s, const char *filename);

void
GSORT_reset(graph);

void
GSORT_run();

int*  // loop with while(-1 != *bestSolution)
GSORT_getBestSolution();

void
GSORT_destroy();

#ifdef __cplusplus
}
#endif

#endif /* BB_LIB_H_GUARD */
