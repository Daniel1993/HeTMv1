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

  char fileName[1<<10] = "input";
  int exp = 1;
  BB_parameters_s params = {
    .n = 21,
    .repOps = 10
  };
  int affinityCore = -1;


  /* Default values are given in variable definition */
  switch(argc){
  case 5:
    sscanf(argv[4], "%d", &affinityCore);
  case 4:
    sscanf(argv[3], "%d", &exp);
    params.repOps = 1L<<exp;
  case 3:
    sscanf(argv[2], "%d", &exp);
    params.n = 1L<<exp;
  case 2:
    strcpy(fileName, argv[1]);
    strcpy(graph_filename, argv[1]);
  case 1: default:
    break;
  }

  if (affinityCore != -1) {
    cpu_set_t set;
    CPU_SET(affinityCore, &set);
    sched_setaffinity(getpid(), sizeof(set), &set);
  }

  /*BB_s bb = */BB_init_F(params, fileName);
  BB_run(/*bb*/);

  // FILE *stream = fopen(fileName, "r");
  // BBLibState.G = loadG(stream);
  // fclose(stream);

  BB_printBestSolution();



  printf(" TEST WITH MATRIX \n");
  unsigned char mat[] = {
    0, 1, 1, 0,
    1, 0, 1, 1,
    1, 1, 0, 0,
    1, 0, 0, 0
    };
  long weights[] = {
    1, 1, 2, 10
  };
  graph G = fromSquareMat(4, mat);
  BB_setWeights(weights);
  BB_reset(G);
  BB_run();
  BB_printBestSolution();

  long weights2[] = {
    1, 2, 1, 1
  };
  G = fromSquareMat(4, mat);
  BB_setWeights(weights2);
  BB_reset(G);
  BB_run();
  BB_printBestSolution();



  BB_destroy(/*bb*/);

  return 0;
}
