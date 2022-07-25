#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <strings.h>

#include "graph.h"
#include "state.h"

/* Keep a register of active vertexes */

typedef struct Qitem *Queue; /* Queues implemented
as linked lists. */

struct Qitem {
  Queue next; /* Next element in the queue */
};

struct state_ {
  graph G;
  int time; /* Used to check color */
  int *A; /* Active states */
  int *C; /* State color */
  enum adjacencies *D; /* Search direction */
  Queue I; /* Array of items */
  Queue Qf[2]; /* Queue fronts */
  Queue Qe[2]; /* Queue ends */
  int *N[2];  /* Current neighbour pointers */
  int *L; /* List of active nodes */
  int *Li; /* Current list element */
};

void
BB_resetS(
  state s
) {
  s->time = -1;
  s->Li = s->L;
  *(s->Li) = -1;
}

state
BB_allocS(graph G
       )
{
  state s;

  s = (state)malloc(sizeof(struct state_));
  s->G = G;
  s->A = (int *)calloc(G->v, sizeof(int));
  s->C = (int *)malloc(G->v*sizeof(int));
  s->D = (enum adjacencies *)malloc(G->v*sizeof(enum adjacencies));
  s->I = (Queue)malloc(G->v*sizeof(struct Qitem));
  s->L = (int *)malloc((1+G->v)*sizeof(int));
  BB_resetS(s);

  return s;
}

void
BB_freeS(state s
      )
{
  free(s->L);
  free(s->I);
  free(s->D);
  free(s->C);
  free(s->A);
  free(s);
}

int *
BB_getSketch(state s
          )
{
  return s->L;
}

FILE *LOG = NULL;

void
BB_printS(state s,
       int depth
       )
{
  if(NULL == LOG)
    LOG = fopen("log", "w");

  int side = 29;

  for(int i = 0; i < side*side; i++){
    if(1 == s->A[i]) {
      fprintf(LOG, "%.3d ", i);
    } else {
      fprintf(LOG, "    ");
    }
    if(0 == ((1+i)%side)){
      fprintf(LOG, "\n");
    }
  }

  fprintf(LOG, "depth = %d\n", depth);
}

static void
DFSvisit(
  state s,
	int u
) {
  if (1 == s->A[u] && s->C[u] != s->time)
  {
    s->C[u] = s->time; /* Mark node */

    int *pv = s->G->E[u][out];
    while (-1 != *pv)
    {
      DFSvisit(s, *pv);
      pv++;
    }

    *(s->Li) = u;
    s->Li++;
    *(s->Li) = -1;
  }
}

void
BB_reOrder(
  state s, /* Current state */
	int *C	 /* Current list */
) { /* Assume the state is clean */

  int *pC = C;
  while (-1 != *pC)
  {
    s->A[*pC] = 1;
    pC++;
  }
  assert(pC-C <= s->G->v && "List overflow");

  /* The only stuff you need is colors */
  s->time++;
  if (0 == s->time) /* Reset the BFS related structures */
  {
    bzero(s->C, s->G->v*sizeof(int));
    s->time++;
  }
  assert(0 < s->time && "Time overflow");

  pC = C;
  while (-1 != *pC)
  {
    DFSvisit(s, *pC);
    pC++;
  }
  assert(pC-C <= s->G->v && "List overflow");

  /* Clean-up */
  pC = C;
  while (-1 != *pC)
  {
    s->A[*pC] = 0;

    s->Li--;
    *pC = *(s->Li);
    *(s->Li) = -1;

    pC++;
  }
  assert(pC-C <= s->G->v && "List overflow");
  assert(s->Li-s->L <= s->G->v && "List overflow");
}

/* Computed by performing a bidirectional interleaved BFS */
int
BB_activate(
  state s,
	int v /* Start vertex */
) { /* Check that there is no loop */
  
  int cycle = 0;
  s->time++;

  if (0 == s->time) /* Reset the BFS related structures */
  {
    bzero(s->C, s->G->v*sizeof(int));
    s->time++;
  }
  assert(0 < s->time && "Time overflow");

  /* Initialize Queues */
  s->Qf[out] = NULL;
  s->Qe[out] = NULL;
  s->Qf[in] = NULL;
  s->Qe[in] = NULL;

  s->C[v] = s->time;
  s->N[out] = s->G->E[v][out];
  s->N[in] = s->G->E[v][in];

  while (!cycle &&
         !(-1 == *(s->N[out]) && NULL == s->Qf[out]) &&
         !(-1 == *(s->N[in]) && NULL == s->Qf[in])
	) {
    for (enum adjacencies d = out; !cycle && d <= in; d++)
    {
      if (-1 == *(s->N[d])) /* End of neighbours */
      {
        Queue t = s->Qf[d];
        int u = t - s->I;
        s->N[d] = s->G->E[u][d];
        s->Qf[d] = t->next;
        t->next = NULL; /* Clean up */
      }
      else /* Pick next neighbour */
      {
        int u = *(s->N[d]);
        if (1 == s->A[u]) /* Process only active nodes. */
        {
          if (s->C[u] == s->time) /* neighbour already found */
          {
            cycle = d != s->D[u];
          }
          else /* Not found yet */
          {
            /* Queue management */
            s->I[u].next = NULL;
            if (NULL != s->Qe[d])
              s->Qe[d]->next = &(s->I[u]);
            s->Qe[d] = &(s->I[u]);
            if (NULL == s->Qf[d])
              s->Qf[d] = &(s->I[u]);

            s->C[u] = s->time;
            s->D[u] = d;
          }
        }
        s->N[d]++;
      }
    }
  }

  if (!cycle && 0 == s->A[v])
  {
#ifndef NDEBUG
    int *l = s->Li;
    while (l-- != s->L)
    {
      assert(v != *l && "Repeated vertex in solution");
      // printf("%d ", *l);
    }
    // printf("  (v=%d, s->A[v]=%d)\n", v, s->A[v]);
#endif
    *(s->Li) = v;
    s->Li++;
    *(s->Li) = -1;
    s->A[v] = 1;
    // printf("Added %i to List\n", v);
    assert(s->Li-s->L <= s->G->v && "List overflow");
  }

  return !cycle;
}

void
BB_deactivate(
  state s,
  int v
) {
  // printf("BB_deactivate v=%d s->A[v]=%i distL=%d\n", v, s->A[v], s->Li-s->L);
  if (1 == s->A[v])
  {
    s->Li--;
    *(s->Li) = -1;
    s->A[v] = 0;

    // int *l = s->Li;
    // while (l-- != s->L)
    //   printf("%d ", *l);
    // printf("  (v=%d)\n", v);
  }
  assert(s->Li >= s->L && "List error, cannot deactivate vertex");
}
