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
#include "bb_lib.h"

// #if defined(USE_ROLLBACK)
// #define BATCH_FACTOR 4
// #else 
#define BATCH_FACTOR 0
// #endif

#define MAX_SCC_FILE "scc%i.in.max_scc"

/* Run a simple branch and bound algorithm */

/* Sorted should be in descending order */
// TODO: Multi-thread
/*__thread */BB_s internal_tmp_bb_;
/*__thread */BB_s internal_bb_;
static /*__thread */FILE* internal_prof_file_;
char graph_filename[128];
static unsigned long notUsedBudgetBB = 0;

static int
pcmp(
  const void *p1, 
  const void *p2
) {
  int i1 = *(int *)p1;
  int i2 = *(int *)p2;

  return internal_tmp_bb_->d[i2] - internal_tmp_bb_->d[i1];
}

void
BB(
  int l, /* Last useful position in order. */
  int depth
) {
  BB_s bb = internal_bb_;
  /* if(depth < 62){ */
  /*   DEB_PRINTF("depth =  %d\n", depth); */
  /*   DEB_PRINTF("bound =  %d\n\n", bound); */
  /* } */
  // printS(bb->s, depth);

  if(depth > bb->bound){
    bb->bound = depth;
  }

  // if(0 != bb->repOps && 0 == (bb->ops % bb->repOps)) {
  //   BB_printLine(bb->currSCC, depth);
  // }

  bb->ops++;
  (bb->budgetPerSCC[bb->currSCC])--;

  // if (depth > 80000) return; // TODO: stack overflow!

  // printf("-> ");
  for(int i = l; bb->bound-depth <= i; i--){
    // if(bb->ops <= bb->n && activate(bb->s[bb->currSCC], bb->order[i])){
    // TODO: add a score within the state of BB, sbound is now relative to the score
    if (0 <= bb->budgetPerSCC[bb->currSCC] && BB_activate(bb->s[bb->currSCC], bb->order[i])) {
      BB(i-1, depth+1);

      if (depth+1 > bb->sbound) {
        bb->sbound = depth+1;
        // memcpy(bb->P, getSketch(bb->s), (depth+1+1)*sizeof(int));
        memcpy(bb->vertsPerSCC[bb->currSCC], BB_getSketch(bb->s[bb->currSCC]), (depth+1+1)*sizeof(int));
        // BB_printLine(bb->currSCC, depth);
      }
    }
    BB_deactivate(bb->s[bb->currSCC], bb->order[i]);
      // printf("(%d) ", bb->order[i]);
  }
  // printf("\n ------------------------------------ \n");
}

void
BB_reset(graph G)
{
  BB_s bb = internal_bb_;
  int i;
  graph temp_G = NULL;

  assert(NULL != G);
  assert(G->v == bb->G->v); // allow only for number of edges to change

  // orderS is the start of the array order (which is incremented in the algo)
  bb->order = bb->orderS; // TODO: where this should go?

  if (bb->G != G) {
    temp_G = bb->G;
    bb->G = G;
  }

  /* Get sorting criteria */
  bb->rG = sccRestrict(G, bb->firstOrder, bb->SCCsize, bb->SCCnbEdges);
  for(i = 0; i < bb->rG->v; i++){
    // bb->d[i] = degree(bb->rG, i, out);
    // bb->d[i] *= degree(bb->rG, i, in);
    bb->d[i] = bb->rG->outDeg[i] * bb->rG->inDeg[i];
  }

  /* Create sub-problem order */
  memcpy(bb->order, &(bb->firstOrder[1]), bb->G->v*sizeof(int));

  for (i = 0; i < bb->firstOrder[0]; ++i) {
    if (bb->SCCsize[i] < 3) {
      notUsedBudgetBB += 128+((bb->n * bb->SCCsize[i]) / bb->rG->v);
    }
  }
  i=0;
  while(i < bb->firstOrder[0]) {
    internal_tmp_bb_ = bb;
    qsort(bb->order, bb->SCCsize[i], sizeof(int), pcmp);
    bb->order += bb->SCCsize[i];
    bb->accSolPerSCC[i] = 0;
    *(bb->vertsPerSCC[i]) = -1;
    if (bb->SCCsize[i] < 3) {
      bb->budgetPerSCC[i] = 2; // 1 should be enough
    } else {
      bb->budgetPerSCC[i] = (((bb->n+notUsedBudgetBB)*bb->SCCsize[i])/bb->rG->v) >> BATCH_FACTOR;
      if (0 >= bb->budgetPerSCC[i]) bb->budgetPerSCC[i] = 128;
    }
    // printf("%d budget = %ld\n", i, bb->budgetPerSCC[i]);
    BB_resetS(bb->s[i], G);
    i++;
  }
  if (NULL != temp_G) {
    freeG(temp_G);
  }
}

void
BB_init_G(BB_parameters_s p, graph G)
{
  BB_s bb = (BB_s)malloc(sizeof(struct BB_));
  internal_bb_ = bb;
  internal_prof_file_ = stdout;
  // temp vars, check if we can allocate in the stack
  // int firstOrder[1+G->v];
  // int d[G->v];
  int i;

  bb->G = G;
  G->d = NULL;
  bb->G->t = NULL;
  clock_gettime(CLOCK_MONOTONIC_RAW, &bb->startTime);

  bb->accSol = 0;
  bb->n = p.n; /* Limit number of ops */
  bb->ops = 0; /* Count the number of iterations */
  bb->repOps = p.repOps < 1 ? 1 : p.repOps; /* When to report information */

  bb->firstOrder = malloc((1+G->v)*sizeof(int));
  bb->order = malloc(G->v*sizeof(int));
  bb->orderS = bb->order;
  bb->orderE = bb->order + G->v;
  bb->SCCsize = malloc((1+G->v)*sizeof(int));
  bb->SCCnbEdges = malloc((1+G->v)*sizeof(int));
  bb->SCCsize[0] = 0;
  bb->d = (int *)malloc(G->v*sizeof(int));
  // bb->SCCsize = realloc(bb->SCCsize, bb->firstOrder[0]*sizeof(int)); // TODO: not needed?

  bb->vertsPerSCC = malloc(G->v*sizeof(int*));
  bb->budgetPerSCC = malloc(G->v*sizeof(long));
  bb->accSolPerSCC = malloc(G->v*sizeof(int));
  bb->s = malloc(G->v*sizeof(state));

  bb->vertsPerSCC[0] = malloc((G->v*G->v+G->v)*sizeof(int));
  i = 0;
  while(i < G->v) {
    bb->s[i] = BB_allocS(bb->G); // TODO: this should be per SCC
    if (i > 0)
      bb->vertsPerSCC[i] = bb->vertsPerSCC[i-1] + G->v + 1;
    i++;
  }
  // free(bb->d);
  // free(bb->firstOrder);

  bb->P = malloc((1+G->v)*sizeof(int));
  bb->FVS = malloc((1+G->v)*sizeof(int));
  *(bb->P) = -1; /* Array where the solution is stored. */

  BB_reset(G);
  // return bb;
}

void
BB_init_F(BB_parameters_s p, const char *fileName)
{
  FILE *stream = fopen(fileName, "r");
  graph G = loadG(stream);
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
  }
  free(bb->budgetPerSCC);
  free(bb->accSolPerSCC);
  free(bb->vertsPerSCC[0]);
  free(bb->vertsPerSCC);
  free(bb->s);
  free(bb->firstOrder);
  free(bb->d);
  free(bb->SCCsize);
  free(bb->orderS);
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
  int i;
  int sboundSave[bb->firstOrder[0]];
  int boundSave[bb->firstOrder[0]];
  int SSCsRemaining = bb->firstOrder[0];
  struct timespec curTime, startTime;

  tn = bb->n; /* Total n value */


#ifdef CHECK_ONLY
  int nbSCCs = 0;
  int j = bb->firstOrder[0];
  int largest_scc = 0;
  int nbInteresting = 0;
  int nbVertInteresting = 0;
  int *order = bb->order;
  printf("#%s;;;\n", graph_filename);
  printf("#SCCid;nbVert;nbEdge;\n");
  while(0 < j) {
    int nbEdges = 0;
    j--;
    order -= bb->SCCsize[j];
    int *v = order;
    for(i = 0; i < bb->SCCsize[j]; ++i) {
      int *outE = bb->rG->E[*v][in];
      while(*(outE++) != -1) nbEdges++;
      v++;
    }
    printf("%i;%i;%i;", bb->firstOrder[0] - j, bb->SCCsize[j], nbEdges);
    if (nbEdges == bb->SCCsize[j]*bb->SCCsize[j]-bb->SCCsize[j]) {
      printf("\n");
    }
    else {
      nbInteresting++;
      nbVertInteresting += bb->SCCsize[j];
      printf("interesting\n");
    }
    if (bb->SCCsize[j] > bb->SCCsize[largest_scc])
      largest_scc = j;
  }
  printf("# %i interesting SCCs with a total of %i verts \n", nbInteresting, nbVertInteresting);
  // prune_SCC(largest_scc);
  bb->order -= bb->rG->v; // free crashes otherwise
  return;
#endif

  for (i = bb->firstOrder[0] - 1; 0 <= i; i--) {
    sboundSave[i] = 0;
    boundSave[i] = -1;
  }

  bb->ops = 0;
  int noMoreBudget = 0;
  
  clock_gettime(CLOCK_MONOTONIC_RAW, &bb->startTime);
  startTime = bb->startTime;
  bb->order = bb->orderE;
  while (noMoreBudget < bb->firstOrder[0]) {
    noMoreBudget = 0;
    for (i = bb->firstOrder[0] - 1; 0 <= i; i--) {
      int currSCCScore = bb->accSolPerSCC[i];
      int *vertsPerSCC = bb->vertsPerSCC[i];
      bb->currSCCsize = bb->SCCsize[i];
      bb->order -= bb->SCCsize[i];
      
      if (0 >= bb->budgetPerSCC[i]) {
        noMoreBudget++;
        continue;
      }

      bb->accSol -= currSCCScore;

      bb->currSCC = i;
      bb->sbound = sboundSave[i]; /* Saved bound */
      bb->bound = boundSave[i];
      /* Proportional re-distribution of ops */

      // bb->n = (tn*bb->SCCsize[i])/bb->rG->v;

      // BB_printLine(bb->currSCC, bb->sbound);

      // if(1 == bb->SCCsize[i] || 2 == bb->SCCsize[i]) {
      // printf("SCC %d has #V = %i #E = %i\n", i, bb->SCCsize[i], bb->SCCnbEdges[i]);
      if(bb->SCCnbEdges[i] == bb->SCCsize[i]*bb->SCCsize[i]-bb->SCCsize[i]) {
        //! Deal with complete graphs
        // printf("SCC %d is trivial\n", i);
        // F++;  /* Ignore 1v or 2v SCCs */
        // with 2v we can just ignore one of them
        *vertsPerSCC = bb->order[0];
        vertsPerSCC++;
        *vertsPerSCC = -1;
        bb->budgetPerSCC[i] = 0;
        noMoreBudget++;
        bb->accSolPerSCC[i] = 1;
        bb->accSol += 1;
        // BB_printLine(bb->currSCC, bb->sbound);
        continue;
      }

      BB(bb->SCCsize[i]-1, 0);
      if (bb->budgetPerSCC[i] > 0) {
        // still has budget and returned == done with SCC
        bb->budgetPerSCC[i] = 0;
        SSCsRemaining = SSCsRemaining == 0 ? SSCsRemaining : SSCsRemaining-1;
      } else if (bb->ops < tn) {
        // add more budget
        bb->budgetPerSCC[i] += ((tn + notUsedBudgetBB - bb->ops) / SSCsRemaining) >> BATCH_FACTOR;
        if (0 >= bb->budgetPerSCC[i]) bb->budgetPerSCC[i] = 1024;
      }

      // DEB_PRINTF("\n\n"); /* Double newline for gnuplot */

      // reOrder(bb->s, bb->P);
      // while(-1 != *bb->P) {
      //   bb->P++;
      //   bb->accSol++;
      // }
      /* *** */
      BB_reOrder(bb->s[i], vertsPerSCC);
      bb->accSolPerSCC[i] = 0;
      while(-1 != *vertsPerSCC) {
        vertsPerSCC++;
        bb->accSolPerSCC[i]++;
      }
      bb->accSol += bb->accSolPerSCC[i];
      sboundSave[i] = bb->sbound; /* Saved bound */
      boundSave[i] = bb->bound;
    } /* for each SCC */

    bb->order += bb->rG->v;
    
  } /* noMoreBudget */
  // BB_printLine(0, bb->sbound);
  bb->order -= bb->rG->v;

  kP = bb->P; /* Keep P */
  for (i = bb->firstOrder[0] - 1; i >= 0; i--) {
    int *vertsPerSCC = bb->vertsPerSCC[i];
    while (*vertsPerSCC != -1) { *(bb->P++) = *(vertsPerSCC++); }
  }
  *bb->P = -1;
  /* *** */

  bb->P = kP; // keep the beginning of the pointer
  int *fvs = bb->FVS;
  // TODO: find a more efficient way
  for (i = 0; i < bb->G->v; ++i) {
    kP = bb->P; // keep the beginning of the pointer
    while (*kP != -1) {
      if (i == *kP) break;
      kP++;
    }
    if (i != *kP) {
      *fvs = i;
      fvs++;
    }
  }
  *fvs = -1;

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

int*
BB_getFVS()
{
  BB_s bb = internal_bb_;
  return bb->FVS;
}

void BB_printHeader()
{
  fprintf(internal_prof_file_, "# iter\ttime_ms\tscore\tmaxScr\taccScr\tSCCid\tSCCsize\n"); // \tvertPerIt\tcurTemp\tnbDfs
}

void BB_printLine(int scc_id, int depth)
{
  BB_s bb = internal_bb_;
  struct timespec curTime, startTime = bb->startTime;
  clock_gettime(CLOCK_MONOTONIC_RAW, &curTime);
  fprintf(internal_prof_file_, "%ld\t%.3f\t%d\t%d\t%d\t%d\t%d\n",
    bb->ops,
    CLOCK_DIFF_MS(startTime, curTime),
    depth,
    bb->accSol + bb->bound,
    bb->accSol,
    scc_id,
    bb->SCCsize[scc_id]
  );
}

void BB_printBestSolution()
{
  BB_s bb = internal_bb_;
  int sol = 0;
  fprintf(internal_prof_file_, "# best solution: ", sol);
  for(int *L = BB_getBestSolution(/*bb*/); -1 != *L; L++) {
    fprintf(internal_prof_file_, "%d ", *L);
    sol++;
  }
  fprintf(internal_prof_file_, "(size %i)\n", sol);
}
