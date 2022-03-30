#ifndef SA_LIB_H_GUARD
#define SA_LIB_H_GUARD

#ifdef __cplusplus
extern "C" {
#endif

#include "sa_state.h"
#include "graph.h"
#include <stdlib.h>
#include <time.h>

// typedef struct graph_ *graph;
// typedef struct state_ *sa_state;

typedef struct SA_ *SA_s;
struct SA_ { // TODO: hide the struct
  unsigned long int n;// = 1<<10; /* Limit number of ops */
  unsigned long int ops;
  unsigned long int repOps;// = 0; /* When to report information */
  int *maxE; /* Maximum obtained thus far */
  double hot;// = 0.10;
  int hotD;// = -5;
  double cold;// = 0.10;
  int coldD;// = -1;

  // break the execution in batches
  int numberOfBatch;
  // double *hotArray; // size of numberOfBatch with linear decay
  // double *coldArray;

  struct timespec startTime;
  graph G;
  graph rG;
  sa_state *s; // per SCC
  sa_state *bestState; // per SCC
  int accScore;
  int **vertsPerSCC;
  long *budgetPerSCC;
  long *initBudgetPerSCC;
  int currSCC;
  int currSCCsize;
  int *accSolPerSCC;
  int *order;
  int *SCCsize;
  int *SCCnbEdges;
};

typedef struct SA_parameters_ {
  int n; /* Limit number of ops */
  int repOps; /* When to report information */
  double hot;// = 0.10;
  int hotD;// = -5;
  double cold;// = 0.10;
  int coldD;// = -1;
} SA_parameters_s;

int
SA_init_G(SA_parameters_s, graph);

int
SA_init_F(SA_parameters_s, const char *filename);

int
SA_reset();

int
SA_get_nbVerts();

int
SA_get_nbEdges();

void
SA_run();

int*  // loop with while(-1 != *bestSolution)
SA_getBestSolution();

void
SA_destroy();

void
SA_set_prof_file(FILE *fp);

void
SA_printHeader();

void
SA_printLine();

struct timespec
SA_getStartTime();

#ifdef __cplusplus
}
#endif

#endif /* SA_LIB_H_GUARD */