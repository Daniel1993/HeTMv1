#include "hy_lib.h"
#include "aux.h"
#include "simulatedAnnealing.h"
#include <math.h>

#include <stdlib.h>

extern unsigned long nbProcVert;
extern unsigned long nbDfsCalls;
extern unsigned long notUsedBudget;

extern __thread BB_s internal_bb_;
extern __thread BB_s internal_tmp_bb_;
extern __thread SA_s internal_lib_sa_;

struct Hy_ {
  BB_s bb;
  SA_s sa;
};

void // also stores internally
Hy_init_G(Hy_parameters_s p, graph g)
{
  SA_init_G(p.sa_p, g);
  BB_init_G(p.bb_p, g);
}

void
Hy_init_F(Hy_parameters_s p, const char *filename)
{
  SA_init_F(p.sa_p, filename);
  BB_init_F(p.bb_p, filename);
}

int
Hy_get_nbVerts()
{
  SA_s sa = internal_lib_sa_;
  return sa->G->v;
}

int
Hy_get_nbEdges()
{
  SA_s sa = internal_lib_sa_;
  return sa->G->e;
}

int
Hy_reset()
{
  SA_reset();
  BB_reset();
  return 0;
}

void
BB(
  int l, /* Last useful position in order. */
  int depth
);

void
Hy_run()
{
  SA_s sa = internal_lib_sa_;
  BB_s bb = internal_bb_;
  BB_s tmp_bb = internal_tmp_bb_;

  SA_printHeader();

  /* With 10% prob. a -5 change */
  double hotStart = getTemperature(sa->hot, sa->hotD);
  /* With 10% prob. a -1 change */
  double coldStart = getTemperature(sa->cold, sa->coldD);

  double dT = (hotStart - coldStart - 0.00001)/((double)sa->numberOfBatch * (double)(1<<BATCH_FACTOR)); /* Update per batch */

  double hot = hotStart;
  double cold = hotStart - dT;

  // printf("hotStart=%f, coldStart=%f, dT=%f, hot=%f, cold=%f\n", hotStart, coldStart, dT, hot, cold);

  int *L = getBestSolutionBuffer(sa->G); /* Current best config */

  /* Running simulated Annealing */
  int *A = &(sa->order[1+sa->rG->v]);
  volatile struct timespec curTime, startTime;

  int *kP;
  unsigned long int tn;
  int i;
  int sboundSave[bb->firstOrder[0]];
  int boundSave[bb->firstOrder[0]];
  int SSCsRemaining = bb->firstOrder[0];
  int noMoreBudget = 0;
  int isFirstRun = 1;

  kP = bb->P; /* Keep P */
  tn = bb->n; /* Total n value */

  sa->ops = 0;

  for (i = bb->firstOrder[0] - 1; 0 <= i; i--) {
    sboundSave[i] = 0;
    boundSave[i] = -1;
#ifdef USE_BB_WHEN_16V // TODO: give name to second heuristic 
    bb->budgetPerSCC[i] = 1 << 30; // very large
#else /* !USE_BB_WHEN_16V */
    bb->budgetPerSCC[i] = 128; // just enough to compute simple and small SCCs
    // bb->budgetPerSCC[i] = sa->rG->v * 2; // just enough to compute simple and small SCCs
#endif
  }
  
  clock_gettime(CLOCK_MONOTONIC_RAW, &sa->startTime);
  bb->startTime = sa->startTime;
  startTime = SA_getStartTime(sa);

  int maxE = 0;
  // for (int j = 0; j < sa->order[0]; ++j) maxE += sa->maxE[j];
  // clock_gettime(CLOCK_MONOTONIC_RAW, (struct timespec*)&curTime);
  // printf("%ld\t%.3f\t%d\t%d\t%d\t%d\t%d\n",
  //   sa->ops,
  //   //  clock()/(CLOCKS_PER_SEC/1000),
  //   CLOCK_DIFF_MS(startTime, curTime),
  //   sa->accScore,
  //   maxE,
  //   sa->accScore,
  //   -1,
  //   0
  // );

  while (noMoreBudget < sa->order[0]) {
    noMoreBudget = 0;

#ifdef LIMIT_TIME
		TIMER_T curTime;
		TIMER_READ(curTime);
		if (CLOCK_DIFF_MS(sa->startTime, curTime) > MAXTIME)
			goto EXITPOS_SA;
#endif

    for (int i = sa->order[0] - 1; 0 <= i; i--) {

      int currSCCScore_bb = bb->accSolPerSCC[i];
      int *vertsPerSCC = sa->vertsPerSCC[i]; // CPY BB-->SA
      long curOps = sa->ops;
      // int *vertsPerSCC_sa = sa->vertsPerSCC[i];

      sa->currSCC = i;
      bb->currSCC = i;
      sa->currSCCsize = sa->SCCsize[i];
      bb->currSCCsize = sa->SCCsize[i];
      bb->order -= bb->SCCsize[i];
      bb->sbound = sboundSave[i]; /* Saved bound */
      bb->bound = boundSave[i];

      if (0 >= sa->budgetPerSCC[i]) {
        noMoreBudget++;
        continue;
      }

      sa->accScore -= sa->accSolPerSCC[i];
      bb->accSol = sa->accScore;

      A -= sa->SCCsize[i];
      if (isFirstRun) {
        *vertsPerSCC = *A;
      }

      SA_printLine(i);

      // if(1 == sa->SCCsize[i] || 2 == sa->SCCsize[i]) {
      // printf("SCC %d has #V = %i #E = %i\n", i, sa->SCCsize[i], sa->SCCnbEdges[i]);
      if(sa->SCCnbEdges[i] == sa->SCCsize[i]*sa->SCCsize[i]-sa->SCCsize[i]) {
        //! Deal with complete graphs
        // printf("SCC %d is trivial\n", i);
        // F++;  /* Ignore 1v or 2v SCCs */
        // with 2v we can just ignore one of them
        *(vertsPerSCC+1) = -1;
        sa->accSolPerSCC[i] = 1;
        bb->accSolPerSCC[i] = 1;
        sa->accScore += 1;
        bb->accSol = sa->accScore;
        sa->budgetPerSCC[i] = 0;
        noMoreBudget++;
        sa->maxE[i] = 1;
        
        goto endSCCForLoop;
      }
#ifdef USE_BB_WHEN_16V
      if (sa->SCCsize[i] < 16) {
        // just run BB
        int *vertsPerSCC_bb = bb->vertsPerSCC[i];
        BB(bb->SCCsize[i]-1, 0);
        BB_reOrder(bb->s[i], vertsPerSCC_bb);
        sa->ops = bb->ops;
        sa->budgetPerSCC[i] = 0;
        noMoreBudget++;
        vertsPerSCC_bb = bb->vertsPerSCC[i];
        int *vertsPerSCC_sa = sa->vertsPerSCC[i];
        while (*vertsPerSCC_bb != -1) *(vertsPerSCC_sa++) = *(vertsPerSCC_bb++);
        *vertsPerSCC_sa = -1;
        vertsPerSCC = bb->vertsPerSCC[i];
        sa->maxE[i] = vertsPerSCC_bb - bb->vertsPerSCC[i];
#endif /* !USE_BB_WHEN_16V */
      if (isFirstRun) {
#ifndef USE_BB_WHEN_16V // TODO: give name to second heuristic 
        int *vertsPerSCC_bb = bb->vertsPerSCC[i];
        int allCandidates[bb->G->v+1];
        for (int i = 0; i < sa->SCCsize[i]; ++i) allCandidates[i] = A[i];
        allCandidates[sa->SCCsize[i]] = -1;
        BB(bb->SCCsize[i]-1, 0);
        BB_reOrder(bb->s[i], vertsPerSCC_bb);
        sa->ops = bb->ops;
        // printf("   <<<< SCC %i/%i with size = %i >>> BB budget left = %i!\n",
        //   i, sa->order[0]-1, sa->SCCsize[i], bb->budgetPerSCC[i]);
        if (1 < bb->budgetPerSCC[i]) { // found solution
          sa->budgetPerSCC[i] = 0;
          noMoreBudget++;
          // printf("   <<<< SCC %i !done! with BB!\n", i);
        // } else {
        //   prepareS(sa->s[i], A, sa->SCCsize[i]);
        //   sa->bestState[i] = cpyS(sa->bestState[i], sa->s[i]);
        }

        vertsPerSCC_bb = bb->vertsPerSCC[i];
        int *vertsPerSCC_sa = sa->vertsPerSCC[i];
        while (*vertsPerSCC_bb != -1) *(vertsPerSCC_sa++) = *(vertsPerSCC_bb++);
        *vertsPerSCC_sa = -1;
        vertsPerSCC = bb->vertsPerSCC[i];
        sa->maxE[i] = vertsPerSCC_bb - bb->vertsPerSCC[i];

        prepareSwithKnowSolution(sa->s[i], vertsPerSCC, sa->maxE[i], bb->order, bb->SCCsize[i]);
#else /* USE_BB_WHEN_16V */
        prepareS(sa->s[i], A, sa->SCCsize[i]);
        sa->bestState[i] = cpyS(sa->bestState[i], sa->s[i]);
#endif /* !USE_BB_WHEN_16V */
      } else {
        vertsPerSCC = executeSA(sa->s[i],
          sa->budgetPerSCC[i],
          hot, cold,
          sa->vertsPerSCC[i], sa->repOps);
        bb->ops = sa->ops;

        // TODO: simple SCCs with ~3 vertices could be cut earlier
        if (sa->ops > sa->n || sa->ops - curOps < 2 /* TODO */) {
          noMoreBudget++;
          sa->budgetPerSCC[i] = 0;
        }
        // vertsPerSCC = sa->vertsPerSCC[i];
      }

      bb->accSolPerSCC[i] = 0;
      sa->accSolPerSCC[i] = 0;
      while(-1 != *vertsPerSCC) {
        vertsPerSCC++;
        bb->accSolPerSCC[i]++;
        sa->accSolPerSCC[i]++;
      }
      bb->accSol += bb->accSolPerSCC[i];
      sa->accScore += sa->accSolPerSCC[i];

      sboundSave[i] = bb->sbound; /* Saved bound */
      boundSave[i] = bb->bound;

endSCCForLoop:
      SA_printLine(i);
    } // end loop per SCC
    A += sa->rG->v;
    bb->order += bb->rG->v;
    if (cold > coldStart) {
      hot = cold;
      cold -= dT;
    }
    isFirstRun = 0;
    SA_printLine(0);
    if (noMoreBudget == sa->order[0]) break;
  } // end loop budget

EXITPOS_SA:
  bb->order -= bb->rG->v;

  int *P = L;
  for (int i = sa->order[0] - 1; i >= 0; i--) {
    if (sa->vertsPerSCC[i] == NULL) {
      sa->vertsPerSCC[i] = bb->vertsPerSCC[i];
    } else if (bb->vertsPerSCC[i] == NULL) {
      bb->vertsPerSCC[i] = sa->vertsPerSCC[i];
    }
    int *vertsPerSCC = sa->vertsPerSCC[i];
    while (*vertsPerSCC != -1) *(L++) = *(vertsPerSCC++);
  }
  *L = -1;
  L = P;

  // *F = -1;

  // free(order);
  // free(SCCsize);

// #ifndef NDEBUG
//   printf("# ");
//   for(P = L; -1 != *P; P++){
//     printf("%d ", *P);
//   }
//   printf("\n");
// #endif /* NDEBUG */

  /* printG(rG); */

#ifdef POSTSCRIPT
  if(NULL != LOG)
    fclose(LOG);
#endif /* POSTSCRIPT */
}

int*  // loop with while(-1 != *bestSolution)
Hy_getBestSolution()
{
  return SA_getBestSolution();
}

void
Hy_set_prof_file(FILE *fp)
{
  SA_set_prof_file(fp);
  BB_set_prof_file(fp);
}

void
Hy_destroy()
{
  SA_destroy();
  BB_destroy();
}
