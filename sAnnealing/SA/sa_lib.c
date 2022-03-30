#include "sa_lib.h"
#include "aux.h"
#include "simulatedAnnealing.h"
#include <math.h>
#include <assert.h>

__thread SA_s internal_lib_sa_;
__thread FILE* internal_prof_file_;

extern unsigned long nbProcVert;
extern unsigned long nbDfsCalls;
unsigned long notUsedBudget = 0;

// call this after the graph is known
static void
allocSAAlgStructs()
{
  SA_s sa = internal_lib_sa_;
  int const minNumberOfBatches = 10; // tentative
  int const minSizeOfBatches = 100; 

  sa->accScore = 0;
  sa->ops = 0;

  // int randval = arc4random();
  // srandom(randval); // for testing
  // printf(" >>> seed is %i <<< \n", randval);
  // srandom(-32777779); // for testing

  // printf("Found %i SSCs in graph\n", sa->order[0]); // TODO: check BB for this

#ifdef POSTSCRIPT
  /* Use this flag only to create animations for the 100x100 torus */
  POST_NB_VERT = ceil(sqrt(sa->G->v));
  if(NULL == LOG)
    LOG = fopen("log", "w");

  fprintf(LOG, "/init { /N exch def 340 N div dup scale -1 -1 translate \
             /Palatino-Roman 0.8 selectfont 0 setlinewidth \
             1 1 N { 1 1 N { 1 index c } for pop } for } bind def \
     /r { translate 0 0 1 1 4 copy rectfill 0 setgray rectstroke } bind def \
     /i { gsave 0.5 setgray r grestore } bind def \
     /d { gsave 0.0 setgray r grestore } bind def \
     /q { gsave r 0.5 0.28 translate (Q) dup stringwidth pop -2 div 0 moveto \
          1 setgray show grestore } bind def \
     /c { gsave 1 setgray r grestore } bind def\n");
  fprintf(LOG, "%d init\n", POST_NB_VERT);
  for (int d = sa->G->v; d < POST_NB_VERT*POST_NB_VERT; ++d) {
    fprintf(LOG, "%d %d d\n",
      1+(d%POST_NB_VERT),
      1+(d/POST_NB_VERT)
    );
  }
#endif /* POSTSCRIPT */

  /* printG(rG); */
  int numberOfBatches = minNumberOfBatches;
  int sizeForTheBatch = 1 + (sa->n / numberOfBatches);

  while (sizeForTheBatch > 10 * minSizeOfBatches) {
    numberOfBatches++;
    sizeForTheBatch = 1 + (sa->n / numberOfBatches);
  }

  for (int i = 0; i < sa->order[0]; ++i) {
    if (sa->SCCsize[i] < 3) {
      notUsedBudget += 1+((sa->n * sa->SCCsize[i]) / sa->rG->v);
    }
  }
  // sa->n += notUsedBudget;
  int *A = sa->order+1;
  int k = 0;
  for (int i = 0; i < sa->order[0]; ++i) {
    prepareS(sa->s[i], A, sa->SCCsize[i]);
#if BATCH_SIZE > 0
    sa->bestState[i] = cpyS(sa->bestState[i], sa->s[i]);
#endif
    A += sa->SCCsize[i];
#if BATCH_SIZE > 0
    if (sa->bestState[i])
      freeS(sa->bestState[i]);
    sa->bestState[i] = allocS(sa->rG);
#endif
    for (int j = 0; j < sa->SCCsize[i]; ++j)
      *(sa->vertsPerSCC[i]+j) = k++;
    *(sa->vertsPerSCC[i] + sa->SCCsize[i]) = -1;
    // sa->budgetPerSCC[i] = ((sa->n * sa->SCCsize[i]) / sa->rG->v);
    sa->budgetPerSCC[i] = (((sa->n+notUsedBudget) * sa->SCCsize[i]) / sa->rG->v) >> BATCH_FACTOR; // allows for a first run over all SCCs
    sa->budgetPerSCC[i] /= numberOfBatches;
    if (0 >= sa->budgetPerSCC[i]) sa->budgetPerSCC[i] = 2;
    // sa->budgetPerSCC[i] += sa->rG->v; // add a little extra to compensate small SCCs
    sa->initBudgetPerSCC[i] = sa->budgetPerSCC[i];
    sa->accSolPerSCC[i] = 0;
  }

  sa->numberOfBatch = numberOfBatches;
  // sa->hotArray = (double*)malloc(sizeof(double)*sa->numberOfBatch);
  // sa->coldArray = (double*)malloc(sizeof(double)*sa->numberOfBatch);
}


int
SA_get_nbVerts()
{
  SA_s sa = internal_lib_sa_;
  return sa->G->v;
}

int
SA_get_nbEdges()
{
  SA_s sa = internal_lib_sa_;
  return sa->G->e;
}

int
SA_reset()
{
  SA_s sa = internal_lib_sa_;
  clock_gettime(CLOCK_MONOTONIC_RAW, &sa->startTime);
  allocSAAlgStructs();
  return 0;
}

int
SA_init_G(SA_parameters_s params, graph G)
{
  SA_s sa = (SA_s)malloc(sizeof(struct SA_));
  internal_lib_sa_ = sa;
  internal_prof_file_ = stdout;

  sa->repOps = params.repOps;
  sa->n = params.n;
  sa->hot = params.hot;
  sa->hotD = params.hotD;
  sa->cold = params.cold;
  sa->coldD = params.coldD;

  G->t = NULL;
  G->d = NULL;
  sa->G = G;
  sa->order = malloc((1+sa->G->v)*sizeof(int));
  sa->SCCsize = malloc((1+sa->G->v)*sizeof(int));
  sa->SCCnbEdges = malloc((1+sa->G->v)*sizeof(int));
  sa->rG = sccRestrict(sa->G, sa->order, sa->SCCsize, sa->SCCnbEdges);

  sa->maxE = calloc(sa->order[0], sizeof(int));
  sa->s = calloc(sa->order[0], sizeof(sa_state));
#if BATCH_SIZE > 0
  sa->bestState = calloc(sa->G->v, sizeof(sa_state));
#endif

  sa->budgetPerSCC = malloc(sa->order[0]*sizeof(long));
  sa->initBudgetPerSCC = malloc(sa->order[0]*sizeof(long));
  sa->accSolPerSCC = malloc((sa->order[0]+1)*sizeof(int));
  sa->vertsPerSCC = calloc(sa->order[0]+1, sizeof(int*));
  sa->vertsPerSCC[0] = malloc((sa->G->v*2)*sizeof(int));

  for (int i = 0; i < sa->order[0]; ++i) {
    sa->s[i] = allocS(sa->rG);
#if BATCH_SIZE > 0
    sa->bestState[i] = allocS(sa->rG);
#endif
    if (i > 0)
      sa->vertsPerSCC[i] = sa->vertsPerSCC[i-1] + sa->SCCsize[i-1] + 1;
  }

  SA_reset();
  return 0;
}

int
SA_init_F(SA_parameters_s params, const char *filename)
{

  FILE *stream = fopen(filename, "r");
  graph G = loadG(stream);
  fclose(stream);

  return SA_init_G(params, G);
}

struct timespec
SA_getStartTime()
{
  SA_s sa = internal_lib_sa_;
  return sa->startTime;
}

void
SA_destroy()
{
  SA_s sa = internal_lib_sa_;
  for (int i = 0; i < sa->order[0]; ++i) {
    freeS(sa->s[i]);
  }
  free(sa->maxE);
  free(sa->vertsPerSCC[0]);
  free(sa->vertsPerSCC);
  free(sa->s);
  freeG(sa->G);
  free(sa->order);
  free(sa->SCCsize);
  free(sa->budgetPerSCC);
  free(sa->accSolPerSCC);
  free(sa);
}

int*  // loop with while(-1 != *bestSolution)
SA_getBestSolution()
{
  SA_s sa = internal_lib_sa_;
  return getBestSolutionBuffer(sa->G);
}

void
SA_run()
{
  SA_printHeader();
  
  SA_s sa = internal_lib_sa_;
  /* With 10% prob. a -5 change */
  double hotStart = getTemperature(sa->hot, sa->hotD);
  /* With 10% prob. a -1 change */
  double coldStart = getTemperature(sa->cold, sa->coldD);

  // add small epsilon
  double dT = (hotStart - coldStart - 0.00001)/((double)sa->numberOfBatch * (double)(1<<BATCH_FACTOR)); /* Update per batch */

  double hot = hotStart;
  double cold = hotStart - dT;

  int maxE = 0;

  // printf("hotStart=%f, coldStart=%f, dT=%f, hot=%f, cold=%f\n", hotStart, coldStart, dT, hot, cold);

  int *L = getBestSolutionBuffer(sa->G); /* Current best config */
  int *P;

  // clock_gettime(CLOCK_MONOTONIC_RAW, &sa->startTime);
  /* Running simulated Annealing */
  int *A = &(sa->order[1+sa->rG->v]);
  volatile struct timespec curTime, startTime = SA_getStartTime(sa);

  sa->ops = 0;
  int noMoreBudget = 0;
  int isFirstRun = 1;

  // SA_printLine(0);

  while (noMoreBudget < sa->order[0]) {
    noMoreBudget = 0;

#ifdef LIMIT_TIME
		TIMER_T curTime;
		TIMER_READ(curTime);
		if (CLOCK_DIFF_MS(sa->startTime, curTime) > MAXTIME)
			goto EXITPOS_SA;
#endif

    for (int i = sa->order[0] - 1; 0 <= i; i--) {
      int *vertsPerSCC = sa->vertsPerSCC[i];
      long curOps = sa->ops;

      sa->currSCC = i;
      sa->currSCCsize = sa->SCCsize[i];

      if (0 >= sa->budgetPerSCC[i]) {
        noMoreBudget++;
        continue;
      }

      sa->accScore -= sa->accSolPerSCC[i];

      A -= sa->SCCsize[i];
      if (isFirstRun) {
        *vertsPerSCC = *A;
      }
      
      // always print initial solution (approx. alg)
      // SA_printLine(i);

      // if(1 == sa->SCCsize[i] || 2 == sa->SCCsize[i]) {
      // printf("SCC %d has #V = %i #E = %i\n", i, sa->SCCsize[i], sa->SCCnbEdges[i]);
      if(sa->SCCnbEdges[i] == sa->SCCsize[i]*sa->SCCsize[i]-sa->SCCsize[i]) {
        //! Deal with complete graphs
        // printf("SCC %d is trivial (ptr=%p)\n", i, vertsPerSCC);
        // F++;  /* Ignore 1v or 2v SCCs */
        // with 2v we can just ignore one of them
        vertsPerSCC++;
        *vertsPerSCC = -1;
        sa->budgetPerSCC[i] = 0;
        noMoreBudget++;
        sa->maxE[i] = 1;
        
        // always print initial solution (approx. alg)
        // SA_printLine(i);
        
        continue;
      }

      // printf("Calling execute on SCC %d\n", i);
      // done on reset
//       if (isFirstRun) {
//         prepareS(sa->s[i], A, sa->SCCsize[i]);
// #if BATCH_SIZE > 0
//         sa->bestState[i] = cpyS(sa->bestState[i], sa->s[i]);
// #endif
//       }

      vertsPerSCC = executeSA(sa->s[i],
        sa->budgetPerSCC[i],
        hot, cold,
        vertsPerSCC, sa->repOps);
      assert(-1 == *vertsPerSCC && "Wrong list ending");
      assert(sa->currSCCsize > vertsPerSCC - sa->vertsPerSCC[i] && "Too many vertexes in list");
      // printf("SCC %d is complex (Sptr=%p,Eptr=%p,prev=%i,%i,next=%i,%i)\n",
      //   i, sa->vertsPerSCC[i], vertsPerSCC,
      //   sa->vertsPerSCC[i-1][0], sa->vertsPerSCC[i-1][1],
      //   sa->vertsPerSCC[i+1][0], sa->vertsPerSCC[i+1][1]);

      // TODO: simple SCCs with ~3 vertices could be cut earlier
      if (sa->ops > sa->n || sa->ops - curOps < 2 /* TODO */) {
        noMoreBudget++;
        sa->budgetPerSCC[i] = 0;
      }

      sa->accSolPerSCC[i] = 0;
      vertsPerSCC = sa->vertsPerSCC[i];
      while(-1 != *vertsPerSCC && vertsPerSCC - sa->vertsPerSCC[i] < sa->currSCCsize) {
        vertsPerSCC++;
        sa->accSolPerSCC[i]++;
      }
      sa->accScore += sa->accSolPerSCC[i];

      // if(0 != sa->repOps && 0 == ((++sa->ops) % sa->repOps)) {
        // SA_printLine(i);
      // }
    } // end loop per SCC
    // SA_printLine(0);
    A += sa->rG->v;
    if (cold > coldStart) {
      hot = cold;
      cold -= dT;
    }
    isFirstRun = 0;
    if (noMoreBudget == sa->order[0]) break;
  } // end loop budget

EXITPOS_SA:

#ifndef NDEBUG
  for (int i = 0; i < sa->order[0]; i++) {
    int *vertsPerSCC = sa->vertsPerSCC[i];
    assert(-1 != *vertsPerSCC); // at least 1 per SCC
    while (*(++vertsPerSCC) != -1);
    assert(sa->SCCsize[i] >= vertsPerSCC - sa->vertsPerSCC[i]);
  }
#endif

  P = L;
  for (int i = 0; i < sa->order[0]; i++) {
    int *vertsPerSCC = sa->vertsPerSCC[i];
    while (*vertsPerSCC != -1) {
      assert(sa->G->v > L - P);
      *L = *vertsPerSCC;
      assert(-1 != *L);
#ifndef NDEBUG
      for (int *kL = P; kL != P; ++kL) assert(*kL != *L);
#endif
      L++;
      vertsPerSCC++;
    }
    // printf("SCC%i <- %i verts\n", i, vertsPerSCC - sa->vertsPerSCC[i]);
  }
  // printf("TOTAL <- %i verts\n", L-P);
  *L = -1;
  L = P;

  /* printG(rG); */

#ifdef POSTSCRIPT
  if(NULL != LOG)
    fclose(LOG);
#endif /* POSTSCRIPT */
}

void
SA_set_prof_file(FILE *fp)
{
  internal_prof_file_ = fp;
}

void SA_printHeader()
{
  fprintf(internal_prof_file_, "# iter\ttime_ms\tscore\tmaxScr\taccScr\tSCCid\tSCCsize\n"); // \tvertPerIt\tcurTemp\tnbDfs
}

void SA_printLine(int scc_id)
{
  
  SA_s sa = internal_lib_sa_;

  struct timespec curTime;
  clock_gettime(CLOCK_MONOTONIC_RAW, (struct timespec*)&curTime);
  int maxE = 0;
  for (int j = 0; j < sa->order[0]; ++j) maxE += sa->maxE[j];
  fprintf(internal_prof_file_, "%ld\t%.3f\t%d\t%d\t%d\t%d\t%d\n",
    sa->ops,
    //  clock()/(CLOCKS_PER_SEC/1000),
    CLOCK_DIFF_MS(sa->startTime, curTime),
    getE(sa->s[scc_id]),
    maxE,
    sa->accScore,
    scc_id,
    sa->SCCsize[scc_id]/*,
    nbProcVert,
    cold,
    nbDfsCalls*/
  );
}
