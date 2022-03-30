#include <stdlib.h>
#include <bsd/stdlib.h>
#include <stdio.h>
#include <math.h>
#include <fenv.h>
#include <assert.h>
#include <limits.h>
#include <string.h>
#include <time.h>

#include "graph.h"
#include "aux.h"
#include "greedy_lib.h"

// #if defined(USE_ROLLBACK)
// #define BATCH_FACTOR 4
// #else 
#define BATCH_FACTOR 0
// #endif

#define MAX_SCC_FILE "scc%i.in.max_scc"

/* Run a simple branch and bound algorithm */

/* Sorted should be in descending order */
__thread graph internal_G_;
__thread GREEDY_s internal_greedy_;
char graph_filename[128];
unsigned long notUsedBudgetGREEDY = 0;

static int policy_func(int inDeg, int outDeg)
{
  // TODO: create other policies
  return inDeg * outDeg;
}

static int
pcmp(
  const void *p1, 
  const void *p2
) {
  int i1 = *(int *)p1;
  int i2 = *(int *)p2;

  return policy_func(internal_G_->inDeg[i2], internal_G_->outDeg[i2])
    - policy_func(internal_G_->inDeg[i1], internal_G_->outDeg[i1]);
}

static int
fvs_rem(
  const void *p1, 
  const void *p2
) {
  int i1 = *(int *)p1;
  int i2 = *(int *)p2;

  return i2 - i1;
}

// TODO: BUG --> it gives all the nodes
static int* // returns current fvs position
applyFVSPolicy(
  graph G,
  int *vertList,
  int nbVerts,
  int *fvs
) {
  int *l = vertList;
  int *f = fvs;

  internal_G_ = G; // used by pcmp
  // TODO: does not need to sort the whole array all the time!
  //  as vertexes get trimmed or selected to FVS, they go to end of the list
  qsort(l, nbVerts, sizeof(int), pcmp); 

  // TODO: the list is sorted, so remove the first one (largest score by policy)
  *f = *l;
  f++;
  *f = -1;
  updateRemFromAdjacencyList(G, *l, NULL);
  return f;
}

static int*
GreedyComponent(graph G, int *fvs, int *orderP, int *sccSizeP, int *sccEdgesP)
{  
  int *scc;
  int *sccSizePtr;
  int *curr_fvs;
  // struct timespec curTime;

  // memcpy(fvs, fvsP);
  curr_fvs = fvs;
  *curr_fvs = -1;
  scc = &(orderP[1]);
  sccSizePtr = sccSizeP;
  while (*sccSizePtr != -1) {
    int reorder = 0;
#ifndef NDEBUG
    int scc_size = *sccSizePtr;
#endif
    // clock_gettime(CLOCK_MONOTONIC_RAW, &curTime);
    // printf("%lu\t%.3f\t%i\n",
    //   greedy->iter++,
    //   CLOCK_DIFF_MS(greedy->startTime, curTime),
    //   i
    // );

    while (*sccSizePtr > 1) {
      // remove 1 vertex from SCC
      curr_fvs = applyFVSPolicy(G, scc, *sccSizePtr, curr_fvs);
      assert(curr_fvs-fvs <= G->v+1);

      // try to find more SCCs
      // printf("restrict scc %d\n", sccSizePtr-sccSizeP);
      reorder = sccRestrictInPlace(G, sccSizePtr-sccSizeP, orderP, sccSizeP, sccEdgesP);

      if (reorder) break;

#ifndef NDEBUG
      if (!reorder) {
        assert(*sccSizePtr < scc_size);
        scc_size = *sccSizePtr;
      }
#endif
    }
    if (reorder) {
      sccSizePtr = sccSizeP;
      scc = &(orderP[1]);
    } else {
      scc += *sccSizePtr;
      sccSizePtr++;
    }
  }
  return fvs;
}

void
GREEDY_init_G(GREEDY_parameters_s p /* TODO: not used */, graph G)
{
  GREEDY_s greedy = (GREEDY_s)malloc(sizeof(struct GREEDY_));

  greedy->iter = 0;
  greedy->G = G;
  greedy->G->t = NULL;
  greedy->G->policy_func = policy_func;
  clock_gettime(CLOCK_MONOTONIC_RAW, &greedy->startTime);

  greedy->G->d = malloc(G->v*sizeof(int));
  greedy->P = malloc((1+G->v)*sizeof(int));
  *(greedy->P) = -1; /* Array where the solution is stored. */
  internal_greedy_ = greedy;

  // TODO: skipped trim from the algorithm
#ifndef NDEBUG
  for (int i = 0; i < greedy->G->v; i++) {
    assert(greedy->G->inDeg[i] == degree(greedy->G, i, in));
    assert(greedy->G->outDeg[i] == degree(greedy->G, i, out));
  }
#endif
}

void
GREEDY_init_F(GREEDY_parameters_s p, const char *fileName)
{
  FILE *stream = fopen(fileName, "r");
  graph G = loadG(stream);
  fclose(stream);

  /*return */GREEDY_init_G(p, G);
}

void
GREEDY_destroy()
{
  GREEDY_s greedy = internal_greedy_;
  // ... free everything
  free(greedy->P);
  freeG(greedy->G); // also frees rG
  free(greedy);
}

void
GREEDY_run()
{
  GREEDY_s greedy = internal_greedy_;
  int v, iter = 0;
  struct timespec curTime, startTime;
  int fvs_size;
  int *kP;

  printf("# nbVerts\ttime_ms\tFVSsize\n");
  // clock_gettime(CLOCK_MONOTONIC_RAW, &greedy->startTime);
  startTime = greedy->startTime;

  clock_gettime(CLOCK_MONOTONIC_RAW, &curTime);
  printf("%d\t%.3f\t%d\n",
    iter++,
    CLOCK_DIFF_MS(startTime, curTime),
    0
  );

  // complete graph
  if (greedy->G->e == greedy->G->v*greedy->G->v-greedy->G->v) {
    for (v = 0; v < greedy->G->v-1; v++) {
      greedy->fvs[v] = v;
    }
    goto END;
  }

  int *order = (int*)malloc((greedy->G->v+2)*sizeof(int));
  int *sccSize = (int*)malloc((greedy->G->v+1)*sizeof(int));
  int *sccEdges = (int*)malloc((greedy->G->v+1)*sizeof(int));
  sccRestrictInPlace(
    greedy->G,
    -1,
    order,
    sccSize,
    sccEdges
  );
  greedy->fvs = (int*)malloc((greedy->G->v+1)*sizeof(int));
  GreedyComponent(greedy->G, greedy->fvs, order, sccSize, sccEdges);
  greedy->curr_fvs = greedy->fvs;
  while (-1 != *greedy->curr_fvs) greedy->curr_fvs++;

  free(sccEdges);
  free(sccSize);
  free(order);

  // sort FVS (makes easier to gather the solution)
  qsort(greedy->fvs, greedy->curr_fvs - greedy->fvs, sizeof(int), fvs_rem);

END:
  fvs_size = greedy->curr_fvs - greedy->fvs;
  greedy->curr_fvs = &(greedy->fvs[0]);

  kP = greedy->P; /* Keep P */
  for (v = greedy->G->v-1; v >= 0; v--) {
    if (v == *greedy->curr_fvs) {
      greedy->curr_fvs++;
    } else {
      *kP = v;
      kP++;
    }
  }
  *kP = -1;
  clock_gettime(CLOCK_MONOTONIC_RAW, &curTime);
  printf("%d\t%.3f\t%d\n",
    iter++,
    CLOCK_DIFF_MS(startTime, curTime),
    fvs_size
  );
  printf("# best solution: ");
  kP = greedy->P;
  while (*kP != -1) printf("%d ", *(kP++));
  printf("(size %ld)\n", kP - greedy->P);
  free(greedy->fvs);
}

int*
GREEDY_getBestSolution()
{
  GREEDY_s greedy = internal_greedy_;
  return greedy->P;
}
