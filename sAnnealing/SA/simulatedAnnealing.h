#ifndef _SIMULATED_ANNEALING_H
#define _SIMULATED_ANNEALING_H

#include "sa_state.h"
#include "sa_lib.h"

/* Used for defining calibration values */
double
getTemperature(double p, /* probability */
	       int dE /* Energy delta */
	       );


/* Return pointer to the end of the buffer */
/* The array of vertexes A, does not need */
/* to be a DAG. A discarding heuristic is used. */
int *
executeSA(
	sa_state s, /* State struct to use */
	unsigned long int n, /* Number of iterations to run */
	double hot, /* Hot temperature */
	double cold, /* Cold temperature */
	int *L, /* Buffer for storing solution */
	unsigned long int repOps /* Report batch size */
	  );

#endif /* _SIMULATED_ANNEALING_H */
