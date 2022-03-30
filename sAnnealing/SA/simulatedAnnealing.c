#include <stdlib.h>
#include <bsd/stdlib.h>
#include <stdio.h>
#include <math.h>
#include <fenv.h>
#include <assert.h>
#include <limits.h>
#include <time.h>

#include "sa_state.h"
#include "sa_lib.h"
#include "aux.h"

extern unsigned long nbProcVert;
extern unsigned long nbDfsCalls;

extern __thread SA_s internal_lib_sa_;

/* Used for defining calibration values */
double
getTemperature(double p, /* probability */
	       int dE /* Energy delta */
	       )
{
  assert(0.5 > p && "Invalid probability to define temp.");
  double res = dE / log2((1/p)-1);
  // printf("temp: %f\n", res);
  return res;
}

// /* Used for defining calibration values */
// double // TODO: dE == 1
// getProb(double T, int d)
// { // TODO: check the formula
//   double res = 1 /(2^{d/T} - 1);
//   // printf("temp: %f\n", res);
//   return res;
// }

/* Return pointer to the end of the buffer */
int *
executeSA(
    sa_state s, /* State struct to use */
	  unsigned long int n, /* Number of iterations to run */
	  double hot, /* Hot temperature */
	  double cold, /* Cold temperature */
	  int *L, /* Buffer for storing solution */
	  unsigned long int repOps /* Report batch size */
	  )
{
  SA_s sa = internal_lib_sa_;
  int *R = L; /* Return value */
  double dT = (hot - cold)/(float)n; /* Update per batch */
  double T = hot; /* Temperature */
  int pdE;
  fesetround(FE_TONEAREST); /* For dE computation */

  // function re-enters multiple times (compute temperature) 
  for (unsigned long int i = 0; i < n; i++, sa->ops++) {
    int dE;

#ifdef LIMIT_TIME
		TIMER_T curTime;
		TIMER_READ(curTime);
		if (CLOCK_DIFF_MS(sa->startTime, curTime) > MAXTIME)
			goto EXITPOS_SA;
#endif

    if(0 != repOps && 0 == (sa->ops % repOps)) {
      SA_printLine(sa->currSCC);
    }

    dE  = INT_MAX; /* Energy delta */
    uint32_t B = arc4random();  /* Random bits buffer */
    // uint32_t B = random(); // for testing

    if(0 != B) {
      float t; /* temporary variable */
      t = log2f(-B);
      t -= log2f(B);
      t *= T;
      dE = lrintf(t);
    }

    if(dE > 1) dE = 1; /* Impossible to improve more than 1 in this problem */
    /* printf("dE %d\n", dE); */

    /* Generate next sa_state */
    if (choose(s)) {
      // printf("Invalid sa_state\n");
      break; // invalid sa_state
    }

    pdE = check(s, dE); /* Returns proposed dE */
    /* printf("pdE %d\n", pdE); */
    /* fprintf(stderr, "dE = %d\t", dE); */
    /* fprintf(stderr, "skip = %d\n", skipGen); */

    // if pdE == 1 accept else always reject --> Greedy
#ifndef USE_GREEDY
    if (dE <= pdE) {
#else /* USE_GREEDY */
    if (pdE == 1) {
#endif
      if (getE(s) > sa->maxE[sa->currSCC]) {
        sa->maxE[sa->currSCC] = getE(s);
        if (0 > pdE) {
          /* Store if you are about to lose best config. */
          /* printf(">>>>>>> REGISTER\n"); */
          if(0 != repOps && 0 == (sa->ops % repOps)) {
            SA_printLine(sa->currSCC);
          }
          // printf("lost best config: ");
          // for (int *P = L; *L != -1; ++L) printf("%i ", *L);
          // printf("\n");
          R = getSketch(s, L);
          // printf("got this one: ");
          // for (int *P = R; *R != -1; ++R) printf("%i ", *R);
          // printf("\n");
#if BATCH_SIZE > 0
          sa->bestState[sa->currSCC] = cpyS(sa->bestState[sa->currSCC], s);
#endif
        }
      }
      execute(s);
    } else {
      /* printf(">>>>>>> REJECTED\n"); */
    }

    /* printf("E= %d\n", getE(s)); */

    /* End update */
    T -= dT; /* Try linear reduction */
  }

EXITPOS_SA:
  if (getE(s) >= sa->maxE[sa->currSCC]) {
    sa->maxE[sa->currSCC] = getE(s);
    R = getSketch(s, L);
  }

  if (R == L) {
    R = getSketch(s, L);
  }
  assert(R != L && "No solution returned");
  assert(sa->currSCCsize > R - L && "Wrong sized solution");

  // if returns to this function, compute from best solution so far
#if BATCH_SIZE > 0
  sa->s[sa->currSCC] = cpyS(sa->s[sa->currSCC], sa->bestState[sa->currSCC]);
#endif

  return R;
}
