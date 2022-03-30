#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#include "graph.h"

static int alreadyPrinted = 0;

static void
tarjanAllocG(graph G, int inPlace);

static void
tarjanFreeG(graph G);

graph
allocG(long v, long e)
{
  graph G = (graph) malloc(sizeof(struct graph_));
  G->v = v;
  G->e = e;
  G->d = NULL;
  G->inDeg  = malloc(v * sizeof(int));
  G->outDeg = malloc(v * sizeof(int));
  G->E = malloc(v*2*sizeof(int*));
  G->t = NULL;
  return G;
}

void
freeG(graph G)
{
  tarjanFreeG(G);
  free(G->E[0][out]);
  free(G->E);
  if (G->d) free(G->d);
  free(G);
}

// matrix accessed via idx_row*nbVertices+idx_col
// TODO: add weights
graph
fromSquareMat(long nbVertices, unsigned char *mat)
{
  int edgeEstimate = nbVertices*nbVertices-nbVertices;
  graph G;

  int (*E)[2] = (int (*)[2])malloc(edgeEstimate*2*sizeof(int*));
  int countE = 0;
  for(int i = 0; i < nbVertices; i++){
    for(int j = 0; j < nbVertices; j++){
      int idx = i * nbVertices + j;
      if (mat[idx] != 0) {
        E[countE][out] = i;
        E[countE][in]  = j;
        assert(E[countE][out] != E[countE][in] && "Loop node in input");
        countE++;
        // now the estimate is the maximum possible number of edges
        // if (countE > edgeEstimate) {
        //   edgeEstimate <<= 1;
        //   E = realloc(E, edgeEstimate*2*sizeof(int*));
        // }
      }
    }
  }
  G = allocG(nbVertices, countE);

  G->t = NULL;
  G->d = NULL;

  /* Count Sort for packing */
  int *c = calloc(2*G->v, sizeof(int)); /* Clean-up */
  for(int i=0; i<G->e; i++){ /* Count */
    int outE = 2*E[i][out];
    int inE = 2*E[i][in];
    c[out+outE]++;
    c[in+inE]++;
  }

  /* Pointer */
  int *p = malloc(2*(G->v+G->e)*sizeof(int));
  for(int i=0; i<2*G->v; i++){ /* Acc */
    p += c[i];
    if(out == i%2)
      G->E[i/2][out] = p;
    else
      G->E[i/2][in] = p;
    *p = -1; /* Add terminator */
    p++;
  }
  free(c);

  G->outDeg = (int*)calloc(G->v, sizeof(int));
  G->inDeg = (int*)calloc(G->v, sizeof(int));
  for(int i=0; i<G->e; i++){ /* Transfer */
    G->E[E[i][out]][out]--;
    *(G->E[E[i][out]][out])=E[i][in];
    G->outDeg[E[i][out]]++;
    G->E[E[i][in]][in]--;
    *(G->E[E[i][in]][in])=E[i][out];
    G->inDeg[E[i][in]]++;
  }
#ifndef NDEBUG
  for (int i = 0; i < G->v; i++) {
    assert(G->inDeg[i] == degree(G, i, in));
    assert(G->outDeg[i] == degree(G, i, out));
  }
#endif
  free(E);

  return G;
}

graph
loadG(FILE *stream)
{
  graph G = (graph) malloc(sizeof(struct graph_));
  fscanf(stream, "%ld%ld", &(G->v), &(G->e));
  G->t = NULL;

// #ifndef NDEBUG
  if (!alreadyPrinted) {
    printf("# V=%li,E=%li,d(v)=%f,logV=%f\n", G->v, G->e, (float)G->e/(float)G->v, log2f(G->v));
    alreadyPrinted = 1;
  }
// #endif

  G->E = malloc(G->v*2*sizeof(int *));

  int (*E)[2] = (int (*)[2])malloc(G->e*2*sizeof(int));
  for(int i = 0; i < G->e; i++){
    // TODO: check input issues like repeated edges (self-loops fails the assert)
    fscanf(stream, "%d%d", &(E[i][out]), &(E[i][in]));

    // TODO: add these 2 lines for TangEtAl2017 graphs (vertex ID starts at 1)
    // (E[i][out])--;
    // (E[i][in])--;

    assert(E[i][out] != E[i][in] && "Loop node in input");
    assert(E[i][out] >= 0 && E[i][out] < G->v && "Invalid vertex ID");
    assert(E[i][ in] >= 0 && E[i][ in] < G->v && "Invalid vertex ID");
  }

  /* Count Sort for packing */
  int *c = calloc(2*G->v, sizeof(int)); /* Clean-up */
  for(int i=0; i<G->e; i++){ /* Count */
    c[out+2*E[i][out]]++;
    c[in+2*E[i][in]]++;
  }

  /* Pointer */
  int *p = malloc(2*(G->v+G->e)*sizeof(int)); // TODO: shouldn't be (2*G->v+G->e) ?
  for(int i=0; i<2*G->v; i++){ /* Acc */
    p += c[i];
    if(out == i%2)
      G->E[i/2][out] = p;
    else
      G->E[i/2][in] = p;
    *p = -1; /* Add terminator */
    p++;
  }
  free(c);

  G->outDeg = (int*)calloc(G->v, sizeof(int));
  G->inDeg = (int*)calloc(G->v, sizeof(int));
  for(int i=0; i<G->e; i++){ /* Transfer */
    G->E[E[i][out]][out]--;
    *(G->E[E[i][out]][out])=E[i][in];
    G->outDeg[E[i][out]]++;
    G->E[E[i][in]][in]--;
    *(G->E[E[i][in]][in])=E[i][out];
    G->inDeg[E[i][in]]++;
  }
#ifndef NDEBUG
  for (int i = 0; i < G->v; i++) {
    assert(G->inDeg[i] == degree(G, i, in));
    assert(G->outDeg[i] == degree(G, i, out));
  }
#endif
  free(E);

  return G;
}

void
printG(graph G)
{
  for(int i = 0; i < G->v; i++){
    printf("vertex %d\n", i);
    printf("out: ");
    int *p = G->E[i][out];
    while(-1 != *p){
      printf("%d ", *p);
      p++;
    }
    printf("\n");

    printf("in: ");
    p = G->E[i][in];
    while(-1 != *p){
      printf("%d ", *p);
      p++;
    }
    printf("\n");
  }
}

static int countEdgesPerSCC(graph G, int count_rG)
{
  int *v = &(G->t->order[1]);
  int *scc = G->t->SCCsizeS;
  int *edges = G->t->SCCnbEdges;
#ifndef NDEBUG
  int count_scc_size = 0;
  while (*scc != -1) {
    count_scc_size += *scc;
    scc++;
  }
  assert(count_scc_size == G->v);
  scc = G->t->SCCsizeS;
#endif
  if (edges) { // optional: count edges per SCC
    while (*scc != -1) {
      *edges = 0;
      for(int i = 0; i < *scc; ++i, ++v) {
        int *outE;
        if (count_rG)
          outE = G->t->rG->E[*v][in];
        else
          outE = G->E[*v][in];
        while(*outE != -1) {
          (*edges)++;
          outE++;
        }
      }
      edges++;
      scc++;
    }
    *edges = -1;
  } else {
    return -1;
  }
  return 0;
}

static void
tarjanAllocG(graph G, int inPlace)
{
  if (!G->t) {
    G->t = (TarjanMetadata) malloc(sizeof(struct TarjanMetadata_));
    if (!inPlace) G->t->rG = allocG(G->v, G->e);
    else          G->t->rG = NULL;
    G->t->TSbool = (char *)calloc(G->v, sizeof(char));
    G->t->TS = (int *)malloc(2*G->v*sizeof(int));
    G->t->low = (int *)malloc(G->v*sizeof(int));
    G->t->d = (int *)calloc(G->v, sizeof(int));
    G->t->L = (int *)malloc((1+G->v)*sizeof(int));
    *G->t->L = -1;
    // for (int i = 0; i < G->v; ++i) G->t->L[i+1] = i;
    G->t->tSize = 0;
    G->t->visited = 0;
    G->t->top = 0;
    G->t->t = NULL;
  }
}

static void
tarjanFreeG(graph G)
{
  if (G->t) {
    if (G->t->rG) free(G->t->rG);
    free(G->t->TSbool);
    free(G->t->TS);
    free(G->t->low);
    free(G->t->d);
    free(G->t->L);
    if (G->t->t) {
      free(G->t->t);
    }
    free(G->t);
  }
}

static void
TarjanVisit(graph G, int u
	    )
{
  G->t->visited++;
  G->t->d[u] = G->t->visited;
  G->t->low[u] = G->t->visited;

  /* Add discovery items */
  G->t->TS[G->t->top] = u;
  G->t->top++; /* Push u to TS */
  G->t->TSbool[u] = 1;

  int *pv = G->E[u][out];
  while(-1 != *pv){
    if(0 == G->t->d[*pv])
      TarjanVisit(G, *pv);
    if(1 == G->t->TSbool[*pv] && G->t->low[*pv] < G->t->low[u])
      G->t->low[u] = G->t->low[*pv];

    pv++;
  }

  /* Add finishing items */
  G->t->TS[G->t->top] = u;
  G->t->top++; /* Push u to TS */

  if(G->t->d[u] == G->t->low[u]){
    (*G->t->sccC)++; /* Count another SCC */

    /* Rewrite low as identifier */
    *G->t->SCCsize = 0;
    G->t->top--;
    do {
      if(1 == G->t->TSbool[G->t->TS[G->t->top]]){
        /* Found a finishing item */
        G->t->low[G->t->TS[G->t->top]] = 1+u;  /* Avoid 0x */
        (*G->t->SCCsize)++;
        *G->t->orderP = G->t->TS[G->t->top];
        G->t->orderP++;
        G->t->TSbool[G->t->TS[G->t->top]] = 0; /* Really pop */
      }
      G->t->top--;
    } while(G->t->TS[G->t->top] != u);

    G->t->SCCsize++; /* Move to the next SCC */
  }
  *G->t->SCCsize = -1; /* for debug loop */
}

// local, just for the sorting
static __thread graph tmp_G;

static int
score(graph G, int v)
{
  return G->inDeg[v]*G->outDeg[v];
}
static int
pcmp(const void *p1, const void *p2)
{
  int i1 = *(int *)p1;
  int i2 = *(int *)p2;
  return score(tmp_G, i1) - score(tmp_G, i2);
}

int
sccRestrictInPlace(graph Gin,
      int scc_id,
	    int *order,
	    int *SCCSA,
      int *SCCED
	    )
{
  graph G = Gin;
  int tarjan_reorder = 0;
  int order_tmp[G->v+2];
  int scc_size_tmp[G->v+1];

  tarjanAllocG(G, 1); // first time: alloc, also checks if Graph changes
  G->t->orderP = &order[1];
  G->t->order = &order[0];
  G->t->sccC = &order[0];
  G->t->SCCsize = SCCSA;
  G->t->SCCsizeS = SCCSA;
  G->t->SCCnbEdges = SCCED;

  int *p;

  if (scc_id != -1) {
    int nb_sccs = G->t->order[0];
    int scc_size = SCCSA[scc_id];
    assert(scc_size <= G->v);
    int *scc = &(G->t->order[1]);
    int i = 0;
    // will add more elements to list
    order_tmp[0] = 0;
    G->t->SCCsize = scc_size_tmp;
    G->t->sccC = &(order_tmp[0]);
    G->t->orderP = &(order_tmp[1]);
    while (i < scc_id) {
      scc += SCCSA[i];
      i++;
    }
    i = 0;
    while (i < scc_size) {
      int v = scc[i];
      if(0 == G->t->d[v]) {
        TarjanVisit(G, v);
      }
      i++;
    }
    int size_before = scc_id;
    int size_in = 0;
    int size_after = nb_sccs - size_before;
    int *tmp_p = scc_size_tmp;
    int *p_after = SCCSA + size_before; // points after ssc_id
    int *p_before = p_after;
    // SCCSA points at the begining
    int count_scc_size = 0;
    int count_sccs = 0;

    // some vertexes may be reordered after this SCC
    while (*tmp_p != -1)
    {
      count_sccs++;
      count_scc_size += *tmp_p;
      
      if (count_scc_size > scc_size) {
        // this means that Tarjan needs to reorder the SCCs
        int proc_verts = G->t->orderP - &(order_tmp[1]);
        tarjan_reorder = 1;
        G->t->SCCsize = SCCSA + order_tmp[0];
        G->t->sccC = &(order[0]);
        G->t->orderP = &(order[1]) + proc_verts;
        memcpy(G->t->SCCsizeS, scc_size_tmp, (order_tmp[0]+1) * sizeof(int));
        memcpy(G->t->sccC, order_tmp, (proc_verts+1) * sizeof(int));
        goto TARJAN_REORDER; // TODO: avoid redo the computation
      }
      tmp_p++;
    }
        
    assert(count_scc_size == scc_size);
    order[0]--; // 1 SCC may be split in a few more
    order[0] += order_tmp[0];
    assert(order[0] <= G->v);
    size_in = tmp_p - scc_size_tmp; // size of list to inject in SCCSA
    assert(size_in < scc_size+1);
    i = 0;

#ifndef NDEBUG
    int dbg_count_scc_size = 0;
    int *scc_ptr_dbg = SCCSA;
    while (*scc_ptr_dbg != -1) {
      dbg_count_scc_size += *scc_ptr_dbg;
      scc_ptr_dbg++;
    }
    assert(dbg_count_scc_size == G->v);
    dbg_count_scc_size = 0;
    scc_ptr_dbg = scc_size_tmp;
    while (*scc_ptr_dbg != -1) {
      dbg_count_scc_size += *scc_ptr_dbg;
      scc_ptr_dbg++;
    }
    assert(dbg_count_scc_size == scc_size);
#endif

    memcpy(tmp_p, p_after + 1, size_after * sizeof(int));
    memcpy(p_before, scc_size_tmp, (size_after + size_in) * sizeof(int));

    G->t->SCCsize = SCCSA;
    G->t->sccC = &(order[0]);
    G->t->orderP = &(order[1]);
  } else {
    order[0] = 0;
TARJAN_REORDER:
    for(int i = 0; i < G->v; i++) {
      if(0 == G->t->d[i]) {
        TarjanVisit(G, i);
      }
    }
  }


#ifndef NDEBUG
    int dbg_count_scc_size = 0;
    assert(G->t->SCCsizeS == SCCSA);
    int *scc_ptr_dbg = G->t->SCCsizeS;
    while (*scc_ptr_dbg != -1) {
      dbg_count_scc_size += *scc_ptr_dbg;
      scc_ptr_dbg++;
    }
    assert(dbg_count_scc_size == G->v);

    // checks if there is no duplicated vertexes
    dbg_count_scc_size = 0;
    int repeated = 0;
    for (int j = 0; j < G->v; ++j) {
      scc_ptr_dbg = &(order[1]);
      int found = 0;
      int i = 0;
      while (i < G->v) {
        if (j == *scc_ptr_dbg && found) {
          repeated = 1;
          break;
        }
        if (j == *scc_ptr_dbg) {
          dbg_count_scc_size++;
          found = 1;
        }
        scc_ptr_dbg++;
        i++;
      }
      assert(!repeated && found);
    }
    assert(dbg_count_scc_size == G->v);
#endif

  /* Restricted graph */
  for(int u = 0; u < G->v; u++){
    p = G->E[u][out];
    while(-1 != *p){
      if(G->t->low[u] != G->t->low[*p]) {
        G->e--; /* Remove SCC crossing edges */
        G->outDeg[u]--;
        G->inDeg[*p]--;
        assert(G->inDeg[*p] >= 0);
        assert(G->outDeg[u] >= 0);
      }
      p++;
    }
  }

  if (0 == G->t->tSize) { // allocs only once at the beginning
    long unsigned int sizeE = 2*(G->v+G->e)*sizeof(int);
    G->t->tSize = sizeE;
    G->t->t = malloc(G->t->tSize);
  }
  int *t = G->t->t;
  int *oldP = G->E[0][out];
  tmp_G = G;
  for(int u = 0; u < G->v; u++) {
    for(enum adjacencies dir = out; dir <= in; dir++) {
      p = G->E[u][dir];
      G->E[u][dir] = t; // replaces the pointer in G->E
      while(-1 != *p) {
        if(G->t->low[u] == G->t->low[*p]) {
          *t = *p;
          t++;
        }
        p++;
      }
      *t = -1;
      qsort(G->E[u][dir], t - G->E[u][dir], sizeof(int), pcmp);
      t++;
    }
  }
  G->t->tSize = 0;
  G->t->t = NULL;
  free(oldP);

#ifndef NDEBUG
  countEdgesPerSCC(G, 0);
  for (int i = 0; i < G->v; i++) {
    assert(G->inDeg[i] == degree(G, i, in));
    assert(G->outDeg[i] == degree(G, i, out));
  }
#endif

  memset(G->t->d, 0, G->v * sizeof(int)); // allow to run again after G is modified
  memset(G->t->TSbool, 0, G->v * sizeof(char));
  G->t->top = 0;
  G->t->visited = 0;

  return tarjan_reorder;
}

graph
sccRestrict(graph Gin,
	    int *order,
	    int *SCCSA,
      int *SCCED
	    )
{
  graph G = Gin;
  tarjanAllocG(G, 0); // first time: alloc, also checks if Graph changes
  G->t->top = 0;
  G->t->visited = 0;
  G->t->orderP = &order[1];
  G->t->order = &order[0];
  order[0] = 0;
  G->t->sccC = &order[0];
  G->t->SCCsize = SCCSA;
  G->t->SCCsizeS = SCCSA;
  G->t->SCCnbEdges = SCCED;

  int *p;
  graph rG = G->t->rG;

  memcpy(rG->outDeg, G->outDeg, G->v*sizeof(int));
  memcpy(rG->inDeg, G->inDeg, G->v*sizeof(int));

  for(int i = 0; i < G->v; i++)
    if(0 == G->t->d[i])
      TarjanVisit(G, i);

  /* Restricted graph */
  for(int u = 0; u < G->v; u++){
    p = G->E[u][out];
    while(-1 != *p){
      if(G->t->low[u] != G->t->low[*p]) {
        rG->e--; /* Remove SCC crossing edges */
        rG->outDeg[u]--;
        rG->inDeg[*p]--;
        assert(rG->inDeg[*p] >= 0);
        assert(rG->outDeg[u] >= 0);
      }
      p++;
    }
  }

  /* Load info into the new graph */
  // int *t = malloc(2*(rG->v+rG->e)*sizeof(int));
  long unsigned int sizeE = 2*(G->v+G->e)*sizeof(int);
  if (0 == G->t->tSize) {
    G->t->tSize = sizeE;
    G->t->t = malloc(G->t->tSize);
  } else if (G->t->tSize < sizeE) {
    G->t->tSize = sizeE;
    // G->t->t = realloc(G->t->t, G->t->tSize);
  }
  int *t = G->t->t;
  tmp_G = G;
  for(int u = 0; u < G->v; u++) {
    for(enum adjacencies dir = out; dir <= in; dir++) {
      rG->E[u][dir] = t;
      p = G->E[u][dir];
      while(-1 != *p) {
        if(G->t->low[u] == G->t->low[*p]) {
          *t = *p;
          t++;
        }
        p++;
      }
      // sort by score
      *t = -1;
      qsort(rG->E[u][dir], t - rG->E[u][dir], sizeof(int), pcmp);
      t++;
    }
  }

  countEdgesPerSCC(G, 1); // TODO: used to check if SCC is complete

#ifndef NDEBUG
  // memset(G->t->d, 0, G->v * sizeof(int)); // allow to run again after G is modified
  for (int i = 0; i < rG->v; i++) {
    assert(rG->inDeg[i] == degree(rG, i, in));
    assert(rG->outDeg[i] == degree(rG, i, out));
  }
#endif

  return rG;
}

int
updateRemFromAdjacencyList(graph G, int v, int *visited)
{
  int *p = G->E[v][out];
  int *q = G->E[v][in];
  int *t = visited;

  while (*p != -1) {
    removeOneVertexFromList(v, G->E[*p][in]);
    // printf("remove %d from in of %d\n", v, *p);
    G->inDeg[*p]--;
    assert(0 <= G->inDeg[*p]);
    G->d[*p] = G->policy_func(G->inDeg[*p], G->outDeg[*p]);
    G->e--;
    if (t) {
      *t = *p;
      t++;
    }
    p++;
  }
  while (*q != -1) {
    removeOneVertexFromList(v, G->E[*q][out]);
    // printf("remove %d from out of %d\n", v, *q);
    G->outDeg[*q]--;
    assert(0 <= G->outDeg[*q]);
    G->d[*q] = G->policy_func(G->inDeg[*q], G->outDeg[*q]);
    G->e--;
    if (t) {
      *t = *q;
      t++;
    }
    q++;
  }
  if (t) *t = -1;
  G->outDeg[v] = 0;
  G->inDeg[v] = 0;
  G->E[v][in][0] = -1;
  G->E[v][out][0] = -1;
  G->d[v] = G->policy_func(0, 0);
  return t - visited;
}

// // merges v into u, puts u in fvs and removes it if self-loop detected
// int // THIS OPERATION IS EXPENSIVE!!!
// updateMerge2Verts(graph G, int v, int u, int *fvs)
// {
//   /* TODO */
//   return 0;
// }

void
removeOneVertexFromList(int toRem, int *list)
{
  int *l = list;
  int *p;
  while (*l != -1 && *l != toRem) l++;
  // we reached either toRem or -1
  p = l;
  assert(*l == toRem || *l == -1);
  while (*l != -1) l++;
  if (l-1 != p) *p = *(l-1);
  *(l-1) = -1;
}

int *
trimG(graph G, int *vertList, int *trimmed, int *procVerts)
{
  int *l = vertList;
  int *t = trimmed;
  int *visited = malloc((G->v+1)*sizeof(int));
  *visited = -1;

  while (*l != -1) {
    int v = *l;
    int is_trimmed = 0;
    // IN0;OUT0 rule
    if (0 == G->inDeg[v] || 0 == G->outDeg[v]) {
      *t = v;
      is_trimmed = 1;
      t++;
      assert(t-trimmed < G->v+1);
      // printf("Trimming %d\n", *l);
      int sizeVis = updateRemFromAdjacencyList(G, v, visited);
      assert(sizeVis < G->v);
      if (sizeVis > 0) {
        t = trimG(G, visited, t, procVerts);
      }
      if (procVerts) (*procVerts)++;
    }
    // IN1;OUT1 rule --> may create self-loops (solved elsewhere)
    // TODO: seems expensive
    if (is_trimmed) {
      removeOneVertexFromList(*l, vertList);
    } else l++;
    assert(l-vertList < G->v+1); // allow for the ending -1
  }
  free(visited);
  *t = -1;
#ifndef NDEBUG
  for (int i = 0; i < G->v; i++) {
    assert(G->inDeg[i] == degree(G, i, in));
    assert(G->outDeg[i] == degree(G, i, out));
    assert(G->d[i] == G->policy_func(G->inDeg[i], G->outDeg[i])); // DOES NOT WORK FOR RANDOM POLICY!
  }
#endif
  return t;
}

int
degree(graph G,
       int v, /* The node */
       enum adjacencies dir
       )
{
  int r = 0;

  int *pv = G->E[v][dir];
  while(-1 != *pv){
    r++;
    pv++;
  }

  return r;
}

int *
getBestSolutionBuffer(graph G)
{
  return G->t->L;
}
