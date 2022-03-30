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
#include "gsort_lib.h"

// #if defined(USE_ROLLBACK)
// #define BATCH_FACTOR 4
// #else 
#define BATCH_FACTOR 0
// #endif

#define MAX_SCC_FILE "scc%i.in.max_scc"

/* Run a simple branch and bound algorithm */

/* Sorted should be in descending order */
// TODO: threading
/*__thread */GSORT_s internal_tmp_gsort_;
/*__thread */GSORT_s internal_gsort_;
char graph_filename[128];
unsigned long notUsedBudgetGSORT = 0;

// sorts from greater to smaller values
static int
pcmp(
  const void *p1, 
  const void *p2
) {
  GSORT_s gsort = internal_tmp_gsort_;
  int i1 = *(int *)p1;
  int i2 = *(int *)p2;

  if (i1 == -2 && i2 == -2)
    return 0;
  else if (i1 == -2)
    return 1;
  else if (i2 == -2)
    return -1;
  else
    return gsort->G->d[i2] - gsort->G->d[i1];
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

static int policy_func(int inDeg, int outDeg)
{
  // TODO: create other policies
  // DOES NOT WORK FOR RANDOM POLICY! (be sure it is deterministic)
  return inDeg * outDeg;
}


// TODO: BUG --> it gives all the nodes
static int* // returns current fvs position
applyFVSPolicy(
  int *vertList,
  int batchSize, /* set to one to just remove 1 vertex */
  int *fvs
) {
  GSORT_s gsort = internal_gsort_;
  int *l = vertList;
  int *f = fvs;
  int k = 0;
  int *tmp_ptr;
  int vertListSize = 0;
  int visited[gsort->G->v+1];
  int *vis = visited;

  while (*l > -1) l++;
  vertListSize = l - vertList;
  l = vertList;

  int tmp[vertListSize + 1]; // TODO: vertListSize

  if (batchSize >= vertListSize) k = -1;

  tmp_ptr = tmp;
  internal_tmp_gsort_ = gsort; // used by pcmp
  // TODO: does not need to sort the whole array all the time!
  //  as vertexes get trimmed or selected to FVS, they go to end of the list
  qsort(l, vertListSize, sizeof(int), pcmp); 
  assert(1 >= vertListSize || gsort->G->d[vertList[0]] >= gsort->G->d[vertList[1]]);

  // TODO: the list is sorted, so we can ignore *l == -2
  while (*l > -1) {
    // printf("Putting %d (score = %d) in FVS\n", *l, gsort->G->d[*l]);
    *f = *l;
    *vis = *l;
    *tmp_ptr = *l;
    gsort->procVerts++;
    tmp_ptr++;
    f++;
    l++;
    vis++;
    if (k == -1 || ++k == batchSize) break;
  }
  *tmp_ptr = -1;
  *f = -1;
  *vis = -1;

  tmp_ptr = tmp;
  while (*tmp_ptr > -1) {
    // printf("Removing %d from vertList\n", *tmp_ptr);
    removeOneVertexFromList(*tmp_ptr, vertList); // cannot remove in the previous loop
    updateRemFromAdjacencyList(gsort->G, *tmp_ptr, visited);
    gsort->trimmed = trimG(gsort->G, visited, gsort->trimmed, &(gsort->procVerts));
    tmp_ptr++;
  }

  return f;
}

void
GSORT_reset(graph G)
{
  GSORT_s gsort = internal_gsort_;
  int i;

  assert(gsort->G->v == G->v);
  assert(gsort->G->d != NULL);
  gsort->procVerts = 0;

  if (G != gsort->G) {
    graph temp_G = gsort->G;
    gsort->G = G;
    gsort->G->d = temp_G->d;
    gsort->G->d = temp_G->d;
    gsort->G->policy_func = temp_G->policy_func;
    temp_G->d = NULL;
    freeG(temp_G);
  }

  clock_gettime(CLOCK_MONOTONIC_RAW, &gsort->startTime);
  /* Calculate policy_func score */
  for(i = 0; i < gsort->G->v; i++){
    gsort->G->d[i] = policy_func(gsort->G->inDeg[i], gsort->G->outDeg[i]);
  }

  *(gsort->P) = -1; /* Array where the solution is stored. */
}

void
GSORT_init_G(GSORT_parameters_s p, graph G)
{
  GSORT_s gsort = (GSORT_s)malloc(sizeof(struct GSORT_));
  int i;

  gsort->G = G;
  gsort->G->t = NULL;
  gsort->G->d = (int *)malloc(gsort->G->v*sizeof(int));
  gsort->P = malloc((1+gsort->G->v)*sizeof(int));
  gsort->G->policy_func = policy_func;
  gsort->batchSize = p.batchSize;

  internal_gsort_ = gsort;

  GSORT_reset(G);
}

void
GSORT_init_F(GSORT_parameters_s p, const char *fileName)
{
  FILE *stream = fopen(fileName, "r");
  graph G = loadG(stream);
  fclose(stream);

  /*return */GSORT_init_G(p, G);
}

void
GSORT_destroy()
{
  GSORT_s gsort = internal_gsort_;
  // ... free everything
  free(gsort->P);
  freeG(gsort->G);
  free(gsort);
}

void
GSORT_run()
{
  GSORT_s gsort = internal_gsort_;
  int fvs[gsort->G->v+1];
  int vertList[gsort->G->v+1];
  int trimmed[gsort->G->v+1]; // TODO: not used
  int *curr_fvs;
  int v, iter = 0;
  struct timespec curTime, startTime;
  int fvs_size;
  int *kP;

  // printf("# nbVerts\ttime_ms\tsizeFVS\n");
  // clock_gettime(CLOCK_MONOTONIC_RAW, &gsort->startTime);
  startTime = gsort->startTime;

  fvs[0] = -1;
  trimmed[0] = -1;
  curr_fvs = &(fvs[0]); 
  gsort->trimmed = &(trimmed[0]); 
  for (v = 0; v < gsort->G->v; v++) {
    vertList[v] = v;
  }
  vertList[gsort->G->v] = -1;

  // complete graph
  if (gsort->G->e == gsort->G->v*gsort->G->v-gsort->G->v) {
    for (v = 0; v < gsort->G->v-1; v++) {
      fvs[v] = v;
    }
    goto END;
  }

  // clock_gettime(CLOCK_MONOTONIC_RAW, &curTime);
  // printf("%d\t%.3f\t%d\n",
  //   iter++,
  //   CLOCK_DIFF_MS(startTime, curTime),
  //   0
  // );

  gsort->trimmed = trimG(gsort->G, vertList, gsort->trimmed, &(gsort->procVerts));
  while (gsort->G->e > 0) {
    curr_fvs = applyFVSPolicy(vertList, gsort->batchSize, curr_fvs); // also trims
    // clock_gettime(CLOCK_MONOTONIC_RAW, &curTime);
    // printf("%d\t%.3f\t%ld\n",
    //   iter++,
    //   CLOCK_DIFF_MS(startTime, curTime),
    //   (curr_fvs-fvs)
    // );
    assert(curr_fvs-fvs <= gsort->G->v+1);
  }

  // sort FVS (makes easier to gather the solution)
  qsort(fvs, curr_fvs - fvs, sizeof(int), fvs_rem);

END:
  fvs_size = curr_fvs - fvs;
  curr_fvs = &(fvs[0]);

  kP = gsort->P; /* Keep P */
  // printf("FVS: ");
  for (v = gsort->G->v-1; v >= 0; v--) {
    if (v == *curr_fvs) {
      // printf("%d ", *curr_fvs);
      curr_fvs++;
    } else {
      *kP = v;
      kP++;
    }
  }
  // printf("\n");
  *kP = -1;
  // clock_gettime(CLOCK_MONOTONIC_RAW, &curTime);
  // printf("%d\t%.3f\t%d\n",
  //   iter++,
  //   CLOCK_DIFF_MS(startTime, curTime),
  //   fvs_size
  // );
  // printf("# best solution: ");
  // kP = gsort->P;
  // while (*kP != -1) printf("%d ", *(kP++));
  // printf("(size %ld)\n", kP - gsort->P);
}

int*
GSORT_getBestSolution()
{
  GSORT_s gsort = internal_gsort_;
  return gsort->P;
}
