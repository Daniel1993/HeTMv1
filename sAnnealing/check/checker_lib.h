#ifndef CHECKER_LIB_H_GUARD
#define CHECKER_LIB_H_GUARD

#ifdef __cplusplus
extern "C" {
#endif

#include "state.h"
#include "graph.h"

#include <time.h>

// typedef struct graph_ *graph;
// typedef struct state_ *state;

struct CHECKER_ {
  unsigned long long n;// = 1<<20; /* Limit number of ops */
  unsigned long int ops;// = 0; /* Count the number of iterations */
  unsigned long int repOps;// = 0; /* When to report information */

  graph G; /* Current graph */
  graph rG; /* Restricted graph */
  int *d; /* Array for storing product degree */
  int *order; /* Array of vertexes by ascending
        product degree. */
  int *P; /* Copy of the best solution */
  int *firstOrder;
  int *SCCsize;
  int *SCCnbEdges;
  int **vertsPerSCC;
  long *budgetPerSCC;
  int currSCC;
  int currSCCsize;
  struct timespec startTime;

  int bound; /* Current best value */
  int sbound; /* Stored best bound */
  int accSol;
  int *accSolPerSCC;
};

typedef struct CHECKER_ *CHECKER_s;

typedef struct CHECKER_parameters_ {
  int none; /* Limit number of ops */
} CHECKER_parameters_s;

void // also stores internally
CHECKER_init_G(CHECKER_parameters_s, graph);

void
CHECKER_init_F(CHECKER_parameters_s, const char *filename);

void
CHECKER_run();

int*  // loop with while(-1 != *bestSolution)
CHECKER_getBestSolution();

void
CHECKER_destroy();

#ifdef __cplusplus
}
#endif

#endif /* CHECKER_LIB_H_GUARD */
