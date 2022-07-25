#ifndef BB_LIB_H_GUARD
#define BB_LIB_H_GUARD

#ifdef __cplusplus
extern "C" {
#endif

#include "state.h"
#include "graph.h"

#include <time.h>

// typedef struct graph_ *graph;
// typedef struct state_ *state;

struct BB_ {
  graph G; /* Current graph */
  graph rG; /* Restricted graph */
  state *s; /* Current state per SCC */
  int *d; /* Array for storing product degree */
  int *order; /* Array of vertexes by ascending
        product degree. */
  int *orderL;
  int *orderS;
  int *orderE;
  int *P; /* Copy of the best solution */
  int *FVS;
  int maxDagSize;
  int *firstOrder;
  int *SCCsize;
  int *SCCnbEdges;
  int *complexSCCs;
  graph *SCCgraphs;
  int **vertsPerSCC;
  long *weights;
  long *accWeights;
  int currSCC;
  int currSCCsize;
  unsigned char *inMaxDag;

  int bound; /* Current best value */
  int sbound; /* Stored best bound */
};

typedef struct BB_ *BB_s;

typedef struct BB_parameters_ {
  unsigned long long n; /* Limit number of ops */
  unsigned long long repOps; /* When to report information */
} BB_parameters_s;

void // also stores internally
BB_init_G(BB_parameters_s, graph);

void
BB_init_F(BB_parameters_s, const char *filename);

void
BB_reset(graph G);

void
BB_setWeights(long *weights);

void
BB_run();

int*
BB_getFVS();

int*  // loop with while(-1 != *bestSolution)
BB_getBestSolution();

void
BB_destroy();

int
BB_get_nbVerts();

int
BB_get_nbEdges();

void
BB_set_prof_file(FILE *fp);

void
BB_printHeader();

void
BB_printLine(int scc_id, int depth);

void
BB_printBestSolution();

#ifdef __cplusplus
}
#endif

#endif /* BB_LIB_H_GUARD */
