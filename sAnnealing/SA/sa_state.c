#include <stdlib.h>
#include <bsd/stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <limits.h>
#include <math.h>

#include "graph.h"
#include "splayTree.h"
#include "darray.h"
#include "reorganize.h"
#include "sa_state.h"

#ifdef POSTSCRIPT
FILE *LOG = NULL;
int POST_NB_VERT = 0;
#endif /* POSTSCRIPT */

#ifndef NDEBUG
static void
assertState(sa_state s);
#endif /* NDEBUG */

#define ARC4RAND_UNIF(_num) arc4random_uniform(_num)
// #define ARC4RAND_UNIF(_num) random() % _num

unsigned long nbProcVert = 0;

struct sa_state_ {
  graph G; /* The underlying graph */
  sTree t; /* K = vertex index. V = Topological Order index. */
  sTree O[2]; /* K = Topological index. V = sub counter. */
  int *Top2V; /* Maps topological order to vertex. */
  int *condemned; /* List of nodes to remove. Normaly small */

  /* Candidate List stuff */
  // char *cBool; /* Candidate booleans */
  int cLs; /* Size of candidate list */
  int *cList; /* List of candidates */
  int reorderCount;
  /* The current candidate is cList[0] */

  /* Excluded adjacency in check */
  darray DegEx[2]; /* Excluded vertexes */

  /* Position lives in the topological order */
  int p; /* The position selected by check */

  enum adjacencies dir; /* Direction selected by check */
  /* Delta E for last check. Used to trim execute. */
  int checkDE;

  /* Supporting the reorganizing heuristic */
  organizer o;
};

sa_state
allocS(graph G)
{
  sa_state s = (sa_state) calloc(1, sizeof(struct sa_state_));

  s->G = G;
  s->t = allocSTree(1+G->v); /* Add a sentinel */
  s->O[in] = allocSTree(G->v);
  s->O[out] = allocSTree(G->v);
  s->Top2V = (int *)calloc(G->v, sizeof(int));
  s->condemned = (int *)calloc(G->v, sizeof(int));
  // s->cBool = (char *)calloc(G->v, sizeof(char));
  s->reorderCount = 0;
  s->cList = (int *)malloc(G->v*sizeof(int));

  s->DegEx[in] = allocDA();
  s->DegEx[out] = allocDA();

  s->o = allocO(G, s->t);

  return s;
}

sa_state // duplicates the sa_state
dupS(sa_state s)
{
  sa_state dup = (sa_state) calloc(1, sizeof(struct sa_state_));
  return cpyS(dup, s);
}

sa_state
cpyS(sa_state dst,
     sa_state src
    )
{
  // printf("DST %p sa_state\n", dst);
  // printS(dst);
  // printf("SRC %p sa_state\n", src);
  // printS(src);
  if (!dst) return dupS(src);

#ifndef NDEBUG
  assertState(src);
#endif /* NDEBUG */

  dst->G       = src->G;
  dst->cLs     = src->cLs;
  dst->checkDE = src->checkDE;
  dst->p       = src->p;
  dst->dir     = src->dir;

  dst->t = cpySTree(dst->t, src->t); /* Add a sentinel */
  dst->O[in] = cpySTree(dst->O[in], src->O[in]);
  dst->O[out] = cpySTree(dst->O[out], src->O[out]);

  dst->Top2V = (int *)realloc(dst->Top2V, src->G->v*sizeof(int));
  memcpy(dst->Top2V, src->Top2V, src->G->v*sizeof(int));

  dst->condemned = (int *)realloc(dst->condemned, src->G->v*sizeof(int));
  memcpy(dst->condemned, src->condemned, src->G->v*sizeof(int));

  // dst->cBool = (char *)realloc(dst->cBool, src->G->v*sizeof(char));
  // memcpy(dst->cBool, src->cBool, src->G->v*sizeof(char));

  dst->cList = (int *)realloc(dst->cList, src->G->v*sizeof(int));
  memcpy(dst->cList, src->cList, src->G->v*sizeof(int));

  dst->DegEx[in] = cpyDA(dst->DegEx[in], src->DegEx[in]);
  dst->DegEx[out] = cpyDA(dst->DegEx[out], src->DegEx[out]);

  dst->o = cpyO(dst->o, src->o);

#ifndef NDEBUG
  assertState(dst);
#endif /* NDEBUG */

  // printf("dst: %p\n", (void*)dst);
  // printS(dst);
  // printf("src: %p\n", (void*)src);
  // printS(src);

  return dst;
}

void
prepareS(sa_state s,
         int*  A, /* Array with vertexes */
         int   l	/* length of the array */
) {
  s->cLs = 0;
  clearTree(s->t);

#ifndef NUSE_INIT_APPROX
  /* Dump array into the tree */
  for (int i = 0; i < l; i++) {
    reRoot(s->t, A[i], left);
    // s->cBool[A[i]] = 1;

#ifdef POSTSCRIPT
    fprintf(LOG, "%d %d i\n",
	    1+(A[i]%POST_NB_VERT),
	    1+(A[i]/POST_NB_VERT));
    fflush(LOG);
#endif /* POSTSCRIPT */
  }

  /* Run approximation discarding */
  for(int i = 0; i < l; i++){
    int v = A[i];
    /* Identify position in topological order */
    node nv = getNode(s->t, v);
    if(NULL != nv){
      int top = value(s->t, nv, NULL);
      int *outA = s->G->E[v][out];

      node u;
      int valid = 1;
      while (-1 != *outA && valid) {
        // s->cBool[*outA] = 1;
        u = getNode(s->t, *outA);
        if (NULL != u &&
            value(s->t, u, NULL) <= top
        ) valid = 0;
        outA++;
        nbProcVert++;
      }

      if (!valid) { /* Means early temination */
        removeN(s->t, u);
        s->cList[s->cLs] = outA[-1];
        s->cLs++;
#ifdef POSTSCRIPT
	fprintf(LOG, "%d %d c\n",
		1+(outA[-1]%POST_NB_VERT),
		1+(outA[-1]/POST_NB_VERT)); // TODO
	fflush(LOG);
#endif /* POSTSCRIPT */
        removeN(s->t, nv);
        s->cList[s->cLs] = v;
        s->cLs++;
#ifdef POSTSCRIPT
	fprintf(LOG, "%d %d c\n",
		1+(v%POST_NB_VERT),
		1+(v/POST_NB_VERT));
	fflush(LOG);
#endif /* POSTSCRIPT */
      }
    }
  }
#else
  for (int i = 0; i < l; i++) {
    s->cList[s->cLs] = A[i];
    s->cLs++;
  }
#endif /* NUSE_INIT_APPROX */
}

void
prepareSwithKnowSolution(sa_state s,
         int*  A, /* Array with vertexes */
         int   l,	/* length of the array */
         int*  C, /* Array all vertexes in SCC */
         int  sC /* size of the SCC */
) {
  s->cLs = 0;
  clearTree(s->t);
  /* Dump array into the tree */
  for (int i = 0; i < l; i++) {
    reRoot(s->t, A[i], left);
  }
  for (int i = 0; i < sC; i++) {
    if (NULL == getNode(s->t, C[i])) {
      s->cList[s->cLs] = C[i];
      s->cLs++;
    }
  }
}
void
freeS(sa_state s)
{
  freeSTree(s->t);
  freeSTree(s->O[in]);
  freeSTree(s->O[out]);
  free(s->Top2V);
  free(s->condemned);
  // free(s->cBool);
  free(s->cList);

  freeDA(s->DegEx[in]);
  freeDA(s->DegEx[out]);
  freeO(s->o);

  free(s);
}

int
getE(sa_state s)
{
  return size(s->t);
}

#ifndef NDEBUG
static void
gdbBreak(void)
{}

static void
assertState(sa_state s)
{
  for(int i = 0; i < s->G->v; i++){
    node u = getNode(s->t, i);
    if(NULL != u){
      int posU = value(s->t, u, NULL);
      int *outA = s->G->E[i][out];
      while(-1 != *outA){
        node v = getNode(s->t, *outA);
        if(NULL != v){
          int posV = value(s->t, v, NULL);
          // printf("test edge %i(pos=%i)->%i(pos=%i) \n", i, posU, *outA, posV);
          if(posU >= posV)
            gdbBreak();
          assert(posU < posV && "Invalid sa_state configuration");
        }
        outA++;
      }
    }
  }
}
#endif /* NDEBUG */

/* Modified sa_state print for torus */

void
printS(sa_state s)
{
  int side = ceil(sqrt(s->G->v));
  int A[side*side];

  for(int i = 0; i < side*side; i++)
    A[i] = -1;

  if(NULL != s->t){
    int L[1+size(s->t)];
    *getInorder(s->t, &L[0], NULL, left) = -1;

    int j = 0;
    int *T = L;
    while(-1 != *T){
      A[*T] = j++;
      T++;
    }
  }

  int v = s->cList[0];
  for(int i = 0; i < side*side; i++){
    if(v == i) {
      fprintf(stderr, " * ");
    } else if(-1 != A[i]) {
      fprintf(stderr, "%.2d ", A[i]);
    } else {
      fprintf(stderr, "   ");
    }
    if(0 == ((1+i)%side)){
      fprintf(stderr, "\n");
    }
  }

  /* for(int j = 0; j<20; j++) */
    /* fprintf(stderr, "\n"); */
}

/* static void */
/* printS(sa_state s) */
/* { */
/*   printf("Printing sa_state\n"); */
/*   if(NULL != s->t){ */
/*     int *L = (int *)malloc((1+size(s->t))*sizeof(int)); */
/*     *getInorder(s->t, L, NULL, left) = -1; */

/*     int *T = L; */
/*     printf("Order: "); */
/*     while(-1 != *T){ */
/*       printf("%d ", *T); */
/*       T++; */
/*     } */
/*     free(L); */
/*     printf("\n"); */
/*   } */

/*   printf("Candidates: "); */
/*   for(int i = 0; i < s->cLs; i++){ */
/*     printf("%d ", s->cList[i]); */
/*   } */
/*   printf("\n"); */
/* } */

static void
swapI(int *A,
      int *B
      )
{
  static int T;

  T = *A;
  *A = *B;
  *B = T;
}

int
choose(sa_state s
       )
{
  assert(0 < s->cLs && "Empty candidate list");
  // if (0 >= s->cLs) return -1;
  int swapIdx = ARC4RAND_UNIF(s->cLs);
  // printf("choose(s) cList[0](%i)<-->cList[%i](%i)\n",
  //   s->cList[0], swapIdx, s->cList[swapIdx]);
  swapI(&(s->cList[0]), &(s->cList[swapIdx]));
  // printf("-cList (size t = %i): ", size(s->t));
  // for (int i = 0; i < s->cLs; ++i) printf("%i ", s->cList[i]);
  // printf("\n");
  return 0;
}

/* Returns how many elements will be removed if position
   p is used. */
static int
incompatibleSize(sa_state s,
		 int p
		 )
{
  int r = 0;
  node floor;
  node ceil;

  roundSt(s->O[out], p, &floor, &ceil);
  if(NULL != ceil){
    r += value(s->O[out], ceil, NULL);
  } else
    r += size(s->O[out]);
  /* Counting strictly less than p */

  roundSt(s->O[in], p, &floor, &ceil);
  if(NULL != floor){
    int c;
    value(s->O[in], floor, &c);
    r += c;

    if(floor == ceil){ /* Also equal to p */
      r++;
    }
  } else
    r += size(s->O[in]);
  /* Counting less than or equal to p */

  return r;
}

static int
next(sa_state s,
     int pos,
     int bound
     )
{
  int p = INT_MAX;
  /* int pout = INT_MAX; */

  node floor;
  node ceil;

  nbProcVert++;

  roundSt(s->O[in], pos, &floor, &ceil);
  if(NULL != ceil)
    p = 1 + key(s->O[in], ceil);

  /* roundSt(s->O[out], pos, &floor, &ceil); */
  /* if(NULL != ceil) */
  /*   pout = 1 + key(s->O[out], ceil); */

  /* if(pout < p) */
  /*   p = pout; */

  if(pos < bound && p > bound)
    p = bound;

  return p;
}

static int
prev(sa_state s,
     int pos,
     int bound
     )
{
  int p = -1;
  int pout = -1;

  node floor;
  node ceil;

  nbProcVert++;

  /* roundSt(s->O[in], pos-1, &floor, &ceil); */
  /* if(NULL != floor) */
  /*   p = key(s->O[in], floor); */

  roundSt(s->O[out], pos-1, &floor, &ceil);
  if(NULL != floor)
    pout = key(s->O[out], floor);

  if(pout > p)
    p = pout;

  if(pos > bound && p < bound)
    p = bound;

  return p;
}

/* Returns the proposed energy variation */
int
check(sa_state s,
      int dE
      )
{
  // printf("cList (size t = %i): ", size(s->t));
  // for (int i = 0; i < s->cLs; ++i) printf("%i ", s->cList[i]);
  // printf("\n");
  assert(2 > dE && "Check with impossible dE");

  /* Apply re-organization heuristic */
  if(checkO(s->o, 0)){
    /*s->reorderCount = 1;*/ // this fix does not work
    s->cList[s->cLs] = -1;
    reorganize(s->o, s->cList);
#ifndef NDEBUG
    assertState(s);
#endif /* NDEBUG */
  }

  /* Candidate */
  int v = s->cList[0];

  /* printf("State before check\n"); */
  /* printS(s); */
  /* printf(">>>> Candidate is : %d\n", v); */

  /* Load trees O[in] and O[out] */
  clearTree(s->O[in]);
  clearTree(s->O[out]);
  resetDA(s->DegEx[in]);
  resetDA(s->DegEx[out]);

  int *inA = s->G->E[v][in];
  int *outA = s->G->E[v][out];
  /* Positions over Topological order */
  int pa = 0;
  int pb = size(s->t);

  while(pa <= pb &&
        !(-1 == *inA && -1 == *outA)
	){
    /* Increment the heuristic */
    checkO(s->o, 2);

    /* Add one incoming */
    if(pa <= pb && -1 != *inA){
      if(NULL == getNode(s->t, *inA))
	      pushDA(s->DegEx[in], *inA);
      else { /* Vertex exists in DAG */
        int top = value(s->t, getNode(s->t, *inA), NULL);
        s->Top2V[top] = *inA;
        insertKey(s->O[in], top);

        /* Move pa forward */
        while(
          pa <= pb &&
	        incompatibleSize(s, pa) > 1 - dE
        ) {
          pa = next(s, pa, 1+pb);
        }
      }
      inA++; /* Cycle step */
    }

    /* Add one outgoing */
    if(pa <= pb && -1 != *outA){
      if(NULL == getNode(s->t, *outA))
	      pushDA(s->DegEx[out], *outA);
      else { /* Vertex exists in DAG */
        int top = value(s->t, getNode(s->t, *outA), NULL);
        s->Top2V[top] = *outA;
        insertKey(s->O[out], top);

        /* Move pb backward */
        while(
          pa <= pb &&
          incompatibleSize(s, pb) > 1 - dE
        ) {
          pb = prev(s, pb, pa); // count this as processed node
        }
      }
      outA++; /* Cycle step */
    }
  }

  /* Selecting the actual split point */
  s->p = pa;
  if(pa <= pb){ /* Otherwise check failed */
    int count = 0;
    int pmin = INT_MAX;
    int p = pa;
    while(p <= pb){
      if(incompatibleSize(s, p) < pmin){
        pmin = incompatibleSize(s, p);
        count = 0;
      }
      if(incompatibleSize(s, p) == pmin) count++;
      p = next(s, p, 1+pb);
    }

    if(1 < count){
      count = ARC4RAND_UNIF(count);
      count++;
    }

    p = pa;
    while(0 < count){
      if(incompatibleSize(s, p) == pmin)
	      count--;
      if(0 < count){
	      p = next(s, p, 1+pb);
      }
    }

    /* Choose uniformly in interval */
    if (p < pb && p+1 < next(s, p, 1+pb)){
      p += ARC4RAND_UNIF(next(s, p, 1+pb) - p);
    }

    s->p = p;
  }

  /* Reload graph Adj */
  int *T = getInorder(s->O[in], s->G->E[v][in], s->Top2V, right);
  dumpDA(s->DegEx[in], T);
  T = getInorder(s->O[out], s->G->E[v][out], s->Top2V, left);
  dumpDA(s->DegEx[out], T);

  /* printf("State after check\n"); */
  /* printS(s); */

#ifndef NDEBUG
  assertState(s);
#endif /* NDEBUG */

  s->checkDE = 1 - incompatibleSize(s, s->p);

  /* fprintf(stderr, "pa = %d ", pa); */
  /* fprintf(stderr, "p = %d ", s->p); */
  /* fprintf(stderr, "pb = %d ", pb); */
  /* fprintf(stderr, "dE = %d ", dE); */
  /* fprintf(stderr, "\n"); */

  // printf("--- checkDE = %i, p = %i\n", s->checkDE, s->p);
  return s->checkDE;
}

/* Executes validated transition */
/* Returns number of new candidates */
void
execute(sa_state s
        )
{
  /* printf("State before execute\n"); */
  /* printS(s); */
  /* fprintf(stderr, "Before candidate insertion"); */
  /* fprintf(stderr, "\n"); */

  int v = s->cList[0]; /* The candidate */
  s->cLs--;
  swapI(&(s->cList[0]), &(s->cList[s->cLs]));
  /* printf("Adopting candidate %d\n", v); */
#ifdef POSTSCRIPT
  fprintf(LOG, "%d %d i\n", 1+(v%POST_NB_VERT), 1+(v/POST_NB_VERT));
#endif /* POSTSCRIPT */

  // printf("size_tree = %i, ins in pos = %i\n",
  //   size(s->t), s->p);

  /* 1 - add v */
  insertInorderKey(s->t, s->p, v);

  assert(size(s->t) > 0);

  /* printS(s); */
  /* fprintf(stderr, "After candidate insertion"); */
  /* fprintf(stderr, "\n"); */

  // printf("checkDE = %i, incompatible size = %i, p = %i, size_tree = %i\n",
  //   s->checkDE, incompatibleSize(s, s->p), s->p, size(s->t)/*, s->t->root*/);

  /* 2 - remove incompatible nodes */
  int *condemned = s->condemned; /* Relabel and re-use */
  splitSt(s->O[in], s->p, right);
  condemned = getInorder(s->O[in], condemned, s->Top2V, right);

  splitSt(s->O[out], s->p-1, left);
  condemned = getInorder(s->O[out], condemned, s->Top2V, left);
  *condemned = -1;

  condemned = s->condemned;
  while(-1 != *condemned && 0 >= s->checkDE){
    removeN(s->t, getNode(s->t, *condemned));
    /* Return back to candidate list, as current candidate. */
    /* printf("Returning candidate %d\n", s->Top2V[*condemned]); */
#ifdef POSTSCRIPT
    fprintf(LOG, "%d %d c\n",
	    1+(*condemned%POST_NB_VERT),
	    1+(*condemned/POST_NB_VERT));
#endif /* POSTSCRIPT */

    /* printS(s); */
    /* fprintf(stderr, "Removed node %d", *condemned); */
    /* fprintf(stderr, "\n"); */

    s->cList[s->cLs] = *condemned;
    s->cLs++;
    condemned++;
    s->checkDE++;
  }

  /* printf("State after execute\n"); */
  /* printS(s); */
}

int *
getSketch(sa_state s,
	        int *L
          )
{
  int *R = getInorder(s->t, L, NULL, left);

  *R = -1;
  return R;
}
