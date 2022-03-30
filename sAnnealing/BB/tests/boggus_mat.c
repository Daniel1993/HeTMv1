#define _GNU_SOURCE
#include <sched.h>
#include <unistd.h>

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

extern char graph_filename[128];

int
main(int argc,
     char** argv
     )
{
  /* Usage: ./project fileName n repOps */
  /*  */
  /* fileName: is the name of the file that contains the graph. */
  /* n: is the maximum allowed number of operations. */
  /* repOps: is the amount of operations until a report line is printed. */
  /* A rep value of 0 means no report. This is the default */

  int exp = 1;
  BB_parameters_s params = {
    .n = 1<<21,
    .repOps = 1<<10
  };
  int affinityCore = -1;


  /* Default values are given in variable definition */

  if (affinityCore != -1) {
    cpu_set_t set;
    CPU_SET(affinityCore, &set);
    sched_setaffinity(getpid(), sizeof(set), &set);
  }

  unsigned char mat[] = {
    0,0,0,0,0,
    0,0,0,1,0,
    0,1,0,0,0,
    0,0,1,0,0,
    0,0,0,0,0,
  } ;

  graph G = fromSquareMat(5, mat);
  BB_init_G(params, G);
  // BB_reset(G);
  BB_run();
  BB_printBestSolution();
  G = fromSquareMat(5, mat);
  BB_reset(G);
  BB_run();
  BB_printBestSolution();
  BB_destroy();

  return 0;
}
