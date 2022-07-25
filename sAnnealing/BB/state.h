#ifndef _BB_STATE_H
#define _BB_STATE_H

#include "graph.h"

typedef struct state_ *state;

void
BB_resetS(state s
       );

state
BB_allocS(graph G
       );

void
BB_freeS(state s
      );

/* A list of vertexes ending in -1 */
int *
BB_getSketch(state s
          );

/* For torus print */
void
BB_printS(state s,
       int depth
       );

void
BB_reOrder(state s, /* Current state */
	int *C	 /* Current list */
	);

/* Return true if it is possible to activate state */
int
BB_activate(state s,
	 int v
	 );

void
BB_deactivate(state s,
	   int v
	   );

#endif /* _BB_STATE_H */
