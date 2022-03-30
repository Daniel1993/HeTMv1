#ifndef HY_LIB_H_GUARD_
#define HY_LIB_H_GUARD_

#ifdef __cplusplus
extern "C" {
#endif

#include "bb_lib.h"
#include "sa_lib.h"

typedef struct Hy_ *Hy_s;

typedef struct Hy_parameters_ {
  BB_parameters_s bb_p;
  SA_parameters_s sa_p;
} Hy_parameters_s;

void // also stores internally
Hy_init_G(Hy_parameters_s, graph);

void
Hy_init_F(Hy_parameters_s, const char *filename);

int
Hy_get_nbVerts();

int
Hy_get_nbEdges();

int
Hy_reset();

void
Hy_run();

int*  // loop with while(-1 != *bestSolution)
Hy_getBestSolution();

void
Hy_destroy();

void
Hy_set_prof_file(FILE *fp);

#ifdef __cplusplus
}
#endif

#endif /* HY_LIB_H_GUARD_ */