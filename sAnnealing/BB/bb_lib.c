#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <fenv.h>
#include <assert.h>
#include <limits.h>
#include <string.h>
// #include <time.h>

#include "graph.h"
#include "state.h"
#include "aux.h"
#include "bb_lib.h"

#define MAX_SCC_FILE "scc%i.in.max_scc"

static const double NB_BATCHES_IN_EXEC =
#if defined(BATCH_SIZE) && BATCH_SIZE > 0
  (double)BATCH_SIZE
#else
  (double)1.0
#endif
;

/* Run a simple branch and bound algorithm */

/* Sorted should be in descending order */
BB_s internal_tmp_bb_;
BB_s internal_bb_;
static FILE* internal_prof_file_;
char graph_filename[128];
static unsigned long notUsedBudgetBB = 0;

static int
pcmp(
  const void *p1, 
  const void *p2
) {
  int i1 = *(int *)p1;
  int i2 = *(int *)p2;

  return internal_tmp_bb_->d[i1] - internal_tmp_bb_->d[i2];
}

void
BB(
  int l, /* Last useful position in order. */
  int depth,
  int cost
) {
  BB_s bb = internal_bb_;

  if(cost > bb->bound)
    bb->bound = cost;

  // if (sigterm_called)
  //   return;
  // fprintf(stderr, "BB(%d (%d),%d) bb->bound-depth=%i\n", l, bb->SCCgraphs[bb->currSCC]->d[l], depth, bb->bound-depth);
  for(int i = l; /* bb->bound-depth <= i */i>=0 && bb->accWeights[i]+cost > bb->bound; i--)
  {
    // printf("BB_activate(..., %d (%d))\n", bb->orderL[i], bb->SCCgraphs[bb->currSCC]->d[bb->orderL[i]]);
    int v = bb->orderL[i];
    int w = bb->weights[v];
    // TODO: add bb->accWeights
      // accumulates the weights from orderL
    if (BB_activate(bb->s[bb->currSCC], v))
    {
      BB(i-1, depth+1, cost+w);
      if (cost+w > bb->sbound)
      {
        // fprintf(stderr, "BB_activate success improve solution i=%i, depth=%i, bb->sbound=%i\n", i, depth, bb->sbound);
        bb->sbound = cost+w;
        memcpy(bb->vertsPerSCC[bb->currSCC], BB_getSketch(bb->s[bb->currSCC]), (depth+1+1)*sizeof(int));
        // int *r = bb->vertsPerSCC[bb->currSCC];
        // while(*r != -1)
        //   printf(" %d bb->currSCC = %i (depth+1+1) = %i\n", *(r++), bb->currSCC, (depth+1+1));
        // printf("\n");
      }
    }
    BB_deactivate(bb->s[bb->currSCC], v);
  }
}


void
BB_setWeights(long *weights)
{
  BB_s bb = internal_bb_;
  int i;
  for (i = 0; i < bb->G->v; i++)
    bb->weights[i] = weights[i];
}

void
BB_reset(graph G)
{
  BB_s bb = internal_bb_;
  int i = 0;
  
  if (bb->rG)
  {
    G->d = bb->G->d; // TODO: move this elsewhere
    bb->G->d = NULL;
    bb->rG->d = NULL;
    while (i < bb->firstOrder[0])
    {
      BB_freeS(bb->s[i]);
      freeG(bb->SCCgraphs[i]);
      i++;
    }
    // freeG(bb->rG); // freed below
    freeG(bb->G);
  }
  bb->G = G;
  bb->rG = sccRestrict(G, bb->firstOrder, bb->SCCsize, bb->SCCnbEdges);
  bb->rG->d = bb->G->d; // TODO: move this elsewhere
  
  // bb->SCCsize = realloc(bb->SCCsize, bb->firstOrder[0]*sizeof(int)); // TODO: not needed?

  /* Get sorting criteria */
  for (i = 0; i < bb->rG->v; i++)
    // bb->d[i] = bb->rG->outDeg[i] * bb->rG->inDeg[i];
    bb->d[i] = bb->rG->outDeg[i]==0||bb->rG->inDeg[i]==0 ? 0 : bb->rG->outDeg[i]+bb->rG->inDeg[i];

  /* Create sub-problem order */
  int *A = &(bb->firstOrder[1]);
  // memcpy(bb->order, &(bb->firstOrder[1]), bb->rG->v*sizeof(int));

  *bb->vertsPerSCC[0] = 0;
  i = 0;
  bb->order = bb->orderS;
  while (i < bb->firstOrder[0])
  {
    internal_tmp_bb_ = bb;
    qsort(bb->order, bb->SCCsize[i], sizeof(int), pcmp);
    bb->SCCgraphs[i] = extractSCC(bb->rG, A, bb->SCCsize[i]);
    A += bb->SCCsize[i];
    bb->order += bb->SCCsize[i];
    bb->s[i] = BB_allocS(bb->SCCgraphs[i]);
    if (i > 0)
    {
      bb->vertsPerSCC[i] = bb->vertsPerSCC[i-1] + bb->SCCsize[i-1] + 1;
      *bb->vertsPerSCC[i] = 0;
    }
    i++;
  }
  // free(bb->d);
  // free(bb->firstOrder);

  *(bb->P) = -1; /* Array where the solution is stored. */
  *(bb->FVS) = -1; /* Array where the solution is stored. */
  
  bb->order = bb->orderS;
  memset(bb->inMaxDag, 0, bb->G->v);
  while (i < bb->firstOrder[0])
  {
    BB_resetS(bb->s[i]);
    i++;
  }
}

void
BB_init_G(BB_parameters_s p, graph G)
{
  BB_s bb = (BB_s)calloc(1, sizeof(struct BB_));
  internal_bb_ = bb;
  internal_prof_file_ = stdout;
  // temp vars, check if we can allocate in the stack
  // int firstOrder[1+G->v];
  // int d[G->v];
  int i;

  G->d = NULL;
  G->t = NULL;

  bb->firstOrder = (int*)malloc((1+G->v)*sizeof(int));
  bb->order = &(bb->firstOrder[1]); // (int*)malloc(G->v*sizeof(int));
  bb->orderS = bb->order;
  bb->orderE = bb->order + G->v;
  bb->SCCsize = (int*)malloc((1+G->v)*sizeof(int));
  bb->SCCnbEdges = (int*)malloc((1+G->v)*sizeof(int));
  bb->SCCsize[0] = 0;
  bb->inMaxDag = (unsigned char*)malloc(G->v*sizeof(unsigned char));
  bb->d = (int *)malloc(G->v*sizeof(int));
  bb->weights = (long *)malloc(G->v*sizeof(long));
  bb->accWeights = (long *)malloc(G->v*sizeof(long));
  for (int i = 0; i < G->v; ++i)
    bb->weights[i] = 1;
  bb->complexSCCs = (int*)malloc(sizeof(int)*(G->v+1));
  bb->vertsPerSCC = (int**)malloc(/* bb->firstOrder[0] */G->v*sizeof(int*));
  bb->vertsPerSCC[0] = (int*)malloc((G->v*2)*sizeof(int));
  bb->orderL = (int*)malloc((G->v)*sizeof(int));
  bb->P = (int*)malloc((1+G->v)*sizeof(int));
  bb->FVS = (int*)malloc((1+G->v)*sizeof(int));
  G->d = (int*)malloc(G->v*sizeof(int)); // TODO: move this elsewhere

  bb->s = malloc(/* bb->firstOrder[0] */G->v*sizeof(state));
  bb->SCCgraphs = (graph*)malloc(/* bb->firstOrder[0] */G->v*sizeof(graph));

  BB_reset(G);
  // return bb;
}

void
BB_init_F(BB_parameters_s p, const char *fileName)
{
  FILE *stream = NULL;
  if (fileName)
    stream = fopen(fileName, "r");
  if (!stream)
    stream = stdin;
  graph G = loadG(stream);
  if (stream != stdin)
    fclose(stream);

  /*return */BB_init_G(p, G);
}


int
BB_get_nbVerts()
{
  BB_s bb = internal_bb_;
  return bb->G->v;
}

int
BB_get_nbEdges()
{
  BB_s bb = internal_bb_;
  return bb->G->e;
}


void
BB_destroy()
{
  BB_s bb = internal_bb_;
  // ... free everything
  for (int i = 0; i < bb->firstOrder[0]; i++) {
    BB_freeS(bb->s[i]);
    freeG(bb->SCCgraphs[i]);
  }
  free(bb->inMaxDag);
  free(bb->vertsPerSCC[0]);
  free(bb->vertsPerSCC);
  free(bb->s);
  free(bb->firstOrder);
  free(bb->d);
  free(bb->weights);
  free(bb->complexSCCs);
  free(bb->SCCsize);
  // free(bb->order);
  free(bb->orderL);
  free(bb->SCCgraphs);
  free(bb->P);
  freeG(bb->G); // also frees rG
  free(bb);
}

#ifdef CHECK_ONLY
static void
prune_SCC(int scc_id)
{
  BB_s bb = internal_bb_;
  char filename[1024];
  FILE *out_fp;
  int *vertexRemap;
  int i, j;
  int *order = bb->order;
  vertexRemap = (int*)malloc(bb->rG->v*sizeof(int));
  for (i = bb->firstOrder[0] - 1; 0 <= i; i--) {
    order -= bb->SCCsize[i];
    // prints all SCCs
    // if (i == scc_id) {
    sprintf(filename, MAX_SCC_FILE, i);
    out_fp = fopen(filename, "w");
    int *v = order;
    int countE = 0;
    for (j = 0; j < bb->SCCsize[i]; ++j) {
      int *p = bb->rG->E[*v][in]; // I think these are reversed
      vertexRemap[*v] = j;
      while (*p != -1) {
        countE++;
        p++;
      }
      v++;
    }
    v = order;
    fprintf(out_fp, "%i %i\n", bb->SCCsize[i], countE);
    for (j = 0; j < bb->SCCsize[i]; ++j) {
      int *p = bb->rG->E[*v][out];
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
#endif

void
BB_run()
{
  BB_s bb = internal_bb_;
  int *kP;
  unsigned long int tn;
  int i, j;

  // fprintf(stderr, "BB_run 1\n");

  bb->orderL += bb->G->v;
  for (i = bb->firstOrder[0] - 1; 0 <= i; i--)
  {
    int *vertsPerSCC = bb->vertsPerSCC[i];
    for (j = 0; j < bb->SCCsize[i]; j++)
    {
      *vertsPerSCC = j;
      bb->orderL--;
      *bb->orderL = j;
      vertsPerSCC++;
    }
    if (bb->SCCnbEdges[i] == bb->SCCsize[i]*bb->SCCsize[i]-bb->SCCsize[i])
    {
      //! Deal with complete graphs
      long maxW = 0;
      int u = -1;
      for (j = 0; j < bb->SCCsize[i]; j++)
      {
        int v = bb->SCCgraphs[i]->d[j];
        int w = bb->weights[v];
        if (w > maxW)
        {
          maxW = w;
          u = j;
        }
      }
      // printf("SCC %d is trivial\n", i);
      // F++;  /* Ignore 1v or 2v SCCs */
      // with 2v we can just ignore one of them
      // *vertsPerSCC = bb->order[0]; // translated later
      *(bb->vertsPerSCC[i]) = u;
      *(bb->vertsPerSCC[i]+1) = -1;
      bb->complexSCCs[i] = 0;
    }
    else 
    {
      bb->complexSCCs[i] = 1;
      *vertsPerSCC = -1;
    }
  }

  int k = 0;
  for (i = 0; i < bb->firstOrder[0]; ++i)
  {
    for (j = 0; j < bb->SCCsize[i]; j++)
    {
      bb->accWeights[k] = bb->weights[bb->SCCgraphs[i]->d[j]];
      if (k > 0)
        bb->accWeights[k] += bb->accWeights[k-1];
      k++;
    }
  }

  // fprintf(stderr, "BB_run 2\n");
  
  bb->order = bb->orderE;
  bb->orderL += bb->G->v;
  for (i = bb->firstOrder[0] - 1; 0 <= i; i--)
  {
    bb->order -= bb->SCCsize[i];
    bb->orderL -= bb->SCCsize[i];
    
    if (!bb->complexSCCs[i])
      continue;

    int *vertsPerSCC = bb->vertsPerSCC[i];
    bb->currSCCsize = bb->SCCsize[i];
    
    bb->currSCC = i;
    bb->sbound = 0; /* Saved bound */
    bb->bound = -1;

    BB(bb->SCCsize[i]-1, 0, 0);
    BB_reOrder(bb->s[i], vertsPerSCC);
  } /* for each SCC */

  kP = bb->P; /* Keep P */
  // printf("\n");
  for (i = bb->firstOrder[0] - 1; i >= 0; i--)
  {
    // fprintf(stderr, "BB_run b1 SCCsize=%i\n", bb->SCCsize[i]);
    int *vertsPerSCC = bb->vertsPerSCC[i];
    while (*vertsPerSCC != -1)
    {
      *bb->P = bb->SCCgraphs[i]->d[*vertsPerSCC];
      // fprintf(stderr, "%i -> %i\n", *vertsPerSCC, *bb->P);
      bb->P++;
      vertsPerSCC++;
    }
    // fprintf(stderr, "BB_run b2\n");
  }
  bb->maxDagSize = bb->P - kP;
  *bb->P = -1;
  /* *** */

  bb->P = kP; // keep the beginning of the pointer

  // fprintf(stderr, "BB_run 3\n");
  
// #ifndef CHECK_ONLY
//   BB_printBestSolution();
// #endif
}

void
BB_set_prof_file(FILE *fp)
{
  internal_prof_file_ = fp;
}

int*
BB_getBestSolution()
{
  BB_s bb = internal_bb_;
  return bb->P;
}

void BB_printHeader()
{
  fprintf(internal_prof_file_, "# iter\ttime_ms\tscore\tmaxScr\taccScr\tSCCid\tSCCsize\n"); // \tvertPerIt\tcurTemp\tnbDfs
}

// void BB_printLine(int scc_id, int depth)
// {
//   BB_s bb = internal_bb_;
//   struct timespec curTime, startTime = bb->startTime;
//   clock_gettime(CLOCK_MONOTONIC_RAW, &curTime);
//   fprintf(internal_prof_file_, "%ld\t%.3f\t%d\t%d\t%d\t%d\t%d\n",
//     bb->ops,
//     CLOCK_DIFF_MS(startTime, curTime),
//     depth,
//     bb->accSol + bb->bound,
//     bb->accSol,
//     scc_id,
//     bb->SCCsize[scc_id]
//   );
// }

void BB_printBestSolution()
{
  BB_s bb = internal_bb_;
  int sol = 0;
  fprintf(internal_prof_file_, "# best solution: ");
  for(int *L = BB_getBestSolution(/*bb*/); -1 != *L; L++) {
    fprintf(internal_prof_file_, "%d ", *L);
    sol++;
  }
  fprintf(internal_prof_file_, "(size %i)\n", sol);
}

int *BB_getFVS()
{
  BB_s bb = internal_bb_;
  int sol = 0;
  int *L = BB_getBestSolution();
  int *r = bb->FVS;
  while (-1 != *L)
  {
    // printf("  -MAX_DAG- %i \n", *L);
    bb->inMaxDag[*L] = 1;
    L++;
  }
  for (int i = 0; bb->G->v > i; i++)
    if (!bb->inMaxDag[i])
      *(r++) = i;
  *r = -1;
  return bb->FVS;
}
