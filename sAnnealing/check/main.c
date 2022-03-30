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
#include "checker_lib.h"

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
  CHECKER_parameters_s params = {
    .none = 0,
  };
  int affinityCore = -1;


  /* Default values are given in variable definition */
  switch(argc){
  case 3:
    sscanf(argv[2], "%d", &affinityCore);
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

  /*checker_s checker = */CHECKER_init_F(params, fileName);
  CHECKER_run(/*checker*/);

  // FILE *stream = fopen(fileName, "r");
  // checkerLibState.G = loadG(stream);
  // fclose(stream);

  CHECKER_destroy(/*checker*/);

  return 0;
}
