#define _GNU_SOURCE
#include <sched.h>
#include <stdlib.h>

#include <sys/types.h>
#include <unistd.h>

#include <stdio.h>
#include <limits.h>

#include "hy_lib.h"

int
main(int argc,
     char** argv
     )
{
  /* Usage: ./project fileName n hotP hotD coldP coldD repOps */
  /*  */
  /* fileName: is the name of the file that contains the graph. */
  /* hotP: is the percentage for hot. */
  /* hotD: is the change for hot < 0. */
  /* coldP: is the percentage for cold. */
  /* coldD: is the change for cold < 0. */
  /* n: is the maximum allowed number of operations. */
  /* repOps: is the amount of operations until a report line is printed. */
  /* A rep value of 0 means no report. This is the default */

  char fileName[1<<10] = "input";
  int exp;
  SA_parameters_s params_sa = {
    .n = 40,
    // SALibState.repOps = 1<<4; /* When to report information */
    .repOps = 1,

    .hot = 0.10,
    .hotD = -1,
    .cold = 0.08,
    .coldD = -1
  };
  BB_parameters_s params_bb = {
    .n = 40,
    // SALibState.repOps = 1<<4; /* When to report information */
    .repOps = 1
  };
  Hy_parameters_s params_hy = {
    .bb_p = params_bb,
    .sa_p = params_sa
  };
  int affinityCore = -1;

  /* Default values are given in variable definition */
  switch(argc){
  case 9:
    sscanf(argv[8], "%d", &affinityCore);
  case 8:
    sscanf(argv[7], "%d", &exp);
    params_hy.bb_p.repOps = 1<<exp;
    params_hy.sa_p.repOps = 1<<exp;
  case 7:
    sscanf(argv[6], "%d", &exp);
    params_hy.bb_p.n = 1<<exp;
    params_hy.sa_p.n = 1<<exp;
  case 6:
    sscanf(argv[5], "%d", &params_hy.sa_p.coldD);
  case 5:
    sscanf(argv[4], "%lf", &params_hy.sa_p.cold);
  case 4:
    sscanf(argv[3], "%d", &params_hy.sa_p.hotD);
  case 3:
    sscanf(argv[2], "%lf", &params_hy.sa_p.hot);
  case 2:
    strcpy(fileName, argv[1]);
  case 1: default:
    break;
  }

  if (affinityCore != -1) {
    cpu_set_t set;
    CPU_SET(affinityCore, &set);
    sched_setaffinity(getpid(), sizeof(set), &set);
  }

  Hy_init_F(params_hy, fileName);
  // FILE *stream = fopen(fileName, "r");
  // SALibState.G = loadG(stream);
  // fclose(stream);

  Hy_run();

  // printf("# ");
  int res = 0;
  for(int *P = Hy_getBestSolution(); -1 != *P; P++){
    res++;
    // printf("%d ", *P);
  }
  // printf("\n");
  printf("# count nb vertices: %i\n", res);

  Hy_destroy();

  return 0;
}
