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
#include "state.h"
#include "aux.h"
#include "checker_lib.h"

// #if defined(USE_ROLLBACK)
// #define BATCH_FACTOR 4
// #else 
#define BATCH_FACTOR 0
// #endif

#define MAX_SCC_FILE "scc%i.in.max_scc"

/* Run a simple branch and bound algorithm */

/* Sorted should be in descending order */
__thread CHECKER_s internal_tmp_checker_;
__thread CHECKER_s internal_checker_;
char graph_filename[128];
unsigned long notUsedBudgetCHECKER = 0;

static int
pcmp(
  const void *p1, 
  const void *p2
) {
  int i1 = *(int *)p1;
  int i2 = *(int *)p2;

  return internal_tmp_checker_->d[i2] - internal_tmp_checker_->d[i1];
}

void
CHECKER_init_G(CHECKER_parameters_s p, graph G)
{
  CHECKER_s checker = (CHECKER_s)malloc(sizeof(struct CHECKER_));
  // temp vars, check if we can allocate in the stack
  // int firstOrder[1+G->v];
  // int d[G->v];
  int i;

  checker->G = G;
  checker->G->t = NULL;
  clock_gettime(CLOCK_MONOTONIC_RAW, &checker->startTime);

  checker->firstOrder = malloc((1+G->v)*sizeof(int));
  checker->order = malloc(G->v*sizeof(int));
  checker->SCCsize = malloc((1+G->v)*sizeof(int));
  checker->SCCnbEdges = malloc((1+G->v)*sizeof(int));
  checker->SCCsize[0] = 0;
  checker->rG = sccRestrict(G, checker->firstOrder, checker->SCCsize, checker->SCCnbEdges);
  checker->d = (int *)malloc(checker->rG->v*sizeof(int));
  // checker->SCCsize = realloc(checker->SCCsize, checker->firstOrder[0]*sizeof(int)); // TODO: not needed?

  /* Get sorting criteria */
  for(i = 0; i < checker->rG->v; i++){
    // checker->d[i] = degree(checker->rG, i, out);
    // checker->d[i] *= degree(checker->rG, i, in);
    checker->d[i] = checker->rG->outDeg[i] * checker->rG->inDeg[i];
  }

  /* Create sub-problem order */
  memcpy(checker->order, &(checker->firstOrder[1]), checker->rG->v*sizeof(int));
  checker->vertsPerSCC = malloc(checker->firstOrder[0]*sizeof(int*));
  checker->budgetPerSCC = malloc(checker->firstOrder[0]*sizeof(long));
  checker->accSolPerSCC = malloc(checker->firstOrder[0]*sizeof(int));
  
  for (i = 0; i < checker->firstOrder[0]; ++i) {
    if (checker->SCCsize[i] < 3) {
      notUsedBudgetCHECKER += 128+((checker->n * checker->SCCsize[i]) / checker->rG->v);
    }
  }
  // checker->n += notUsedBudgetCHECKER;

  i = 0;
  while(i < checker->firstOrder[0]) {
    internal_tmp_checker_ = checker;
    qsort(checker->order, checker->SCCsize[i], sizeof(int), pcmp);
    checker->order += checker->SCCsize[i];
    checker->accSolPerSCC[i] = 0;
    checker->vertsPerSCC[i] = malloc((checker->SCCsize[i]+1)*sizeof(int));
    *(checker->vertsPerSCC[i]) = -1;
    if (checker->SCCsize[i] < 3) {
      checker->budgetPerSCC[i] = 2; // 1 should be enough
    } else {
      checker->budgetPerSCC[i] = (((checker->n+notUsedBudgetCHECKER)*checker->SCCsize[i])/checker->rG->v) >> BATCH_FACTOR;
      if (0 >= checker->budgetPerSCC[i]) checker->budgetPerSCC[i] = 128;
    }
    i++;
  }
  // free(checker->d);
  // free(checker->firstOrder);

  checker->P = malloc((1+checker->rG->v)*sizeof(int));
  *(checker->P) = -1; /* Array where the solution is stored. */

  internal_checker_ = checker;
  // return checker;
}

void
CHECKER_init_F(CHECKER_parameters_s p, const char *fileName)
{
  FILE *stream = fopen(fileName, "r");
  graph G = loadG(stream);
  fclose(stream);

  /*return */CHECKER_init_G(p, G);
}

void
CHECKER_destroy()
{
  CHECKER_s checker = internal_checker_;
  // ... free everything
  for (int i = 0; i < checker->firstOrder[0]; i++) {
    free(checker->vertsPerSCC[i]);
  }
  free(checker->budgetPerSCC);
  free(checker->accSolPerSCC);
  free(checker->vertsPerSCC);
  free(checker->firstOrder);
  free(checker->d);
  free(checker->SCCsize);
  free(checker->order);
  free(checker->P);
  freeG(checker->G); // also frees rG
  free(checker);
}

static void
prune_SCC(int scc_id)
{
  CHECKER_s checker = internal_checker_;
  char filename[1024];
  FILE *out_fp;
  int *vertexRemap;
  int i, j;
  int *order = checker->order;
  vertexRemap = (int*)malloc(checker->rG->v*sizeof(int));
  for (i = checker->firstOrder[0] - 1; 0 <= i; i--) {
    order -= checker->SCCsize[i];
    // prints all SCCs
    // if (i == scc_id) {
    sprintf(filename, MAX_SCC_FILE, i);
    out_fp = fopen(filename, "w");
    int *v = order;
    int countE = 0;
    for (j = 0; j < checker->SCCsize[i]; ++j) {
      int *p = checker->rG->E[*v][in]; // I think these are reversed
      vertexRemap[*v] = j;
      while (*p != -1) {
        countE++;
        p++;
      }
      v++;
    }
    v = order;
    fprintf(out_fp, "%i %i\n", checker->SCCsize[i], countE);
    for (j = 0; j < checker->SCCsize[i]; ++j) {
      int *p = checker->rG->E[*v][out];
      while (*p != -1) {
        fprintf(out_fp, "%i %i\n", vertexRemap[*v], vertexRemap[*p]);
        p++;
      }
      v++;
    }
    fclose(out_fp);
    // }
  }
  free(vertexRemap);
}

void
CHECKER_run()
{
  CHECKER_s checker = internal_checker_;
  int *kP;
  unsigned long int tn;
  int i;
  int sboundSave[checker->firstOrder[0]];
  int boundSave[checker->firstOrder[0]];
  int SSCsRemaining = checker->firstOrder[0];
  struct timespec curTime, startTime;

  tn = checker->n; /* Total n value */

  // TODO: check if there are repeated edges

  int nbSCCs = 0;
  int j = checker->firstOrder[0];
  int largest_scc = 0;
  int nbInteresting = 0;
  int nbVertInteresting = 0;
  int *order = checker->order;
  printf("#%s;;;\n", graph_filename);
  printf("#SCCid;nbVert;nbEdge;\n");
  while(0 < j) {
    int nbEdges = 0;
    j--;
    order -= checker->SCCsize[j];
    int *v = order;
    for(i = 0; i < checker->SCCsize[j]; ++i) {
      int *outE = checker->rG->E[*v][in];
      while(*(outE++) != -1) nbEdges++;
      v++;
    }
    printf("%i;%i;%i;", checker->firstOrder[0] - j, checker->SCCsize[j], nbEdges);
    if (nbEdges == checker->SCCsize[j]*checker->SCCsize[j]-checker->SCCsize[j]) {
      printf("\n");
    }
    else {
      nbInteresting++;
      nbVertInteresting += checker->SCCsize[j];
      printf("interesting\n");
    }
    if (checker->SCCsize[j] > checker->SCCsize[largest_scc])
      largest_scc = j;
  }
  printf("# %i interesting SCCs with a total of %i verts \n", nbInteresting, nbVertInteresting);
  // prune_SCC(largest_scc);
  checker->order -= checker->rG->v; // free crashes otherwise
}

int*
CHECKER_getBestSolution()
{
  CHECKER_s checker = internal_checker_;
  return checker->P;
}
