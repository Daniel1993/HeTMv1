#include <bsd/stdlib.h>
#include <string.h>

#include "queue.h"
#include "splayTree.h"
#include "reorganize.h"
#include "sa_state.h"

struct organizer{
  graph G;
  int time; /* Used to check color */
  char *A; /* Break node booleans */
  int *C[2]; /* node color */
  int *cS[2]; /* re-start color */
  queue rS[2]; /* Re-starts */

  sTree t; /* Resulting order */
  int cnt; /* Heuristic counter */
};

unsigned long nbDfsCalls = 0;

/* Creates a new organizing structure */
organizer
allocO(graph G,
       sTree t
       )
{
  organizer o = malloc(sizeof(struct organizer));

  o->G = G;
  o->time = -1;
  o->A = calloc(G->v, sizeof(char));

  o->C[in] = malloc(G->v*sizeof(int));
  o->C[out] = malloc(G->v*sizeof(int));

  o->cS[in] = malloc(G->v*sizeof(int));
  o->cS[out] = malloc(G->v*sizeof(int));
  o->rS[in] = allocQ(G->v);
  o->rS[out] = allocQ(G->v);

  o->t = t;
  o->cnt = 0;

  return o;
}

organizer
dupO(organizer o
      )
{
  organizer dup = calloc(1, sizeof(struct organizer));
  return cpyO(dup, o);
}

organizer
cpyO(organizer dst,
     organizer src
    )
{
  dst->G = src->G;
  dst->time = src->time;
  dst->A = (char*)realloc(dst->A, src->G->v*sizeof(char));
  memcpy(dst->A, src->A, src->G->v*sizeof(char));

  dst->C[in] = (int*)realloc(dst->C[in], src->G->v*sizeof(int));
  memcpy(dst->C[in], src->C[in], src->G->v*sizeof(int));
  dst->C[out] = (int*)realloc(dst->C[out], src->G->v*sizeof(int));
  memcpy(dst->C[out], src->C[out], src->G->v*sizeof(int));

  dst->cS[in] = (int*)realloc(dst->cS[in], src->G->v*sizeof(int));
  memcpy(dst->cS[in], src->cS[in], src->G->v*sizeof(int));
  dst->cS[out] = (int*)realloc(dst->cS[out], src->G->v*sizeof(int));
  memcpy(dst->cS[out], src->cS[out], src->G->v*sizeof(int));

  dst->rS[in] = cpyQ(dst->rS[in], src->rS[in]);
  dst->rS[out] = cpyQ(dst->rS[out], src->rS[out]);

  dst->t = cpySTree(dst->t, src->t);
  dst->cnt = src->cnt;

  return dst;
}

void
freeO(organizer o
      )
{
  free(o->A);
  free(o->C[in]);
  free(o->C[out]);
  free(o->cS[in]);
  free(o->cS[out]);
  freeQ(o->rS[in]);
  freeQ(o->rS[out]);
  o->t = NULL;

  free(o);
}

int
checkO(organizer o,
       int dE
       )
{
  // TODO: set this factor to balance calls to dfs
  // with algorithm precision
  const float factor = 2.0f;
  int r = o->cnt > factor*(o->G->v + o->G->e);

  if(!r){
    o->cnt += factor * dE;
  }

  return r;
}

/* Using global vars to optimize stack space */
static organizer Go;
static enum adjacencies GVout;
static node tu; /* Insert after this node */

static void
dfs(int v /* Vertex to start from */
    )
{
  // count here 
  organizer o = Go;
  enum adjacencies Vout = GVout;

  enum adjacencies Vin = in;
  enum childI Vright = right;

  nbDfsCalls++;

  if(Vout == in){
    Vin = out;
    Vright = left;
  }

  o->C[Vout][v] = o->time; /* Paint node */

  int *Adj = o->G->E[v][Vout];
  while (-1 != *Adj) {
    int u = *Adj;
    Adj++;

    if(!o->A[u] && /* Not a break node */
       o->time != o->C[Vout][u] /* Unvisited */
       ){

      o->C[Vout][u] = o->time; /* Paint node */
      dfs(u);

      /* Check on the other search */
      if (o->time != o->C[Vin][u] &&
          o->time != o->cS[Vin][u] ) {
        o->cS[Vin][u] = o->time; /* Paint node */
        pushQ(o->rS[Vin], u);
      }
    }
  }

  if(NULL != tu && !o->A[v])
    insertN(o->t, tu, v, Vright);
}

/* Inserts topological order into t */
void
reorganize(organizer o,
	   int *L /* List of break nodes */
	   )
{
  o->cnt = 0;

  o->time++;
  if(0 == o->time){
    o->time=1;
    bzero(o->C[in], o->G->v*sizeof(int));
    bzero(o->C[out], o->G->v*sizeof(int));
    bzero(o->cS[in], o->G->v*sizeof(int));
    bzero(o->cS[out], o->G->v*sizeof(int));
  }

  enum adjacencies Vout = out;
  enum adjacencies Vin = in;
  enum childI Vleft = left;
  if(0 == arc4random_uniform(2)){
    Vout = in;
    Vin = out;
    Vleft = right;
  }

  clearTree(o->t);
  reRoot(o->t, o->G->v, Vleft);
  node min = getNode(o->t, o->G->v);

  for(int i = 0; -1 != L[i]; i++){
    o->A[L[i]] = 1;
  }

  for(int i = 0; -1 != L[i]; i++){
    if(o->cS[Vout][L[i]] != o->time){

      pushQ(o->rS[Vout], L[i]); /* Cold re-start */
      o->cS[Vout][L[i]] = o->time;

      while(!(emptyQ(o->rS[Vout]) &&
	            emptyQ(o->rS[Vin]))){

        while(!emptyQ(o->rS[Vout])){ /* Simple re-start */
          int v = frontQ(o->rS[Vout]);
          if(o->time != o->C[Vout][v]){
            Go = o;
            GVout = Vout;
            tu = min;
            dfs(v);
          }
          popQ(o->rS[Vout]);
        }

        if(!emptyQ(o->rS[Vin])){
          int v = frontQ(o->rS[Vin]);
          if(o->time != o->C[Vin][v]){
            Go = o;
            GVout = Vin;
            tu = NULL;
            dfs(v);
          }
          popQ(o->rS[Vin]);
        }
      }
    }
  }

  for(int i = 0; -1 != L[i]; i++){
    o->A[L[i]] = 0;
  }

  /* Remove sentinel from the tree */
  removeN(o->t, getNode(o->t, o->G->v));
}
