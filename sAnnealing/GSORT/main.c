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
#include "aux.h"
#include "gsort_lib.h"

extern char graph_filename[128];

int
main(int argc,
     char** argv
     )
{
  /* Usage: ./project fileName n repOps */
  /*  */
  /* fileName: is the name of the file that contains the graph. */
  /* batchSize: how many nodes to put into FVS at each iteration. */
  /* affinityCore: coreID to pin the thread to (multi-instantiation) */
  char fileName[1<<10] = "input";
  int batchSize = 1;
  GSORT_parameters_s params = {
    .batchSize = 1
  };
  int affinityCore = -1;


  /* Default values are given in variable definition */
  switch(argc){
  case 4:
    sscanf(argv[3], "%d", &affinityCore);
  case 3:
    sscanf(argv[2], "%d", &batchSize);
    params.batchSize = batchSize;
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

  /*GSORT_s gsort = */GSORT_init_F(params, fileName);
  GSORT_run(/*gsort*/);

  // FILE *stream = fopen(fileName, "r");
  // GSORTLibState.G = loadG(stream);
  // fclose(stream);

  GSORT_destroy(/*gsort*/);

  return 0;
}
