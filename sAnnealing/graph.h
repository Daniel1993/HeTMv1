#ifndef _GRAPH_H
#define _GRAPH_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// NOTE: in C++ struct graph also defines a keyword "graph"
typedef struct graph_ *graph;
typedef struct TarjanMetadata_ *TarjanMetadata;

enum adjacencies {out, in};

// TODO: hide these 2 (now they are used somewhere else)
struct graph_ {
  long v; /* Number of vertexes */
  long e; /* Number of edges */
  int *inDeg; // keep track of the in-/out-degree count for each vertex
  int *outDeg;
  int *d; // array to keep the score of the vertexes (not used by all solutions)
   // outputs the score for that vertex (e.g., product of inDeg and outDeg)
  int (*policy_func)(int inDeg, int outDeg);
  int *(*E)[2]; /* Array of pairs of pointers */
  TarjanMetadata t;
};

struct TarjanMetadata_ {
  graph rG;
  char *TSbool;
  int *TS;
  int top;
  int *low;
  int *d;
  int *t;
  int *L;
  size_t tSize;
  int visited;
  int *order; // ref to begining of orderP
  int *orderP;
  int *sccC; /* SCC counter */
  int *SCCsize; // TODO: this pointer goes up and down... not reliable
  int *SCCsizeS; // start of SCCsize
  int *SCCnbEdges;
};


graph
allocG(long v, long e);

void
freeG(graph G);

// matrix accessed via idx_row*nbVertices+idx_col
graph
fromSquareMat(long nbVertices, unsigned char *mat);

graph
loadG(FILE *stream);

void
freeG(graph G);

void
printG(graph G);

int
sccRestrictInPlace(graph Gin,
      int scc_id,
	    int *order,
	    int *SCCSA,
      int *SCCED
	    );

/* Return a new graph without edges that cross SCCs. Also returns the
   component graph in reverse topological order and each SCC in topological
   order (approximately). */
graph
sccRestrict(graph G,
	    int *order,
	    int *SCCsize,
	    int *SCCnbEdges
	    );

void // TODO: move to some utils
removeOneVertexFromList(int toRem, int *list);

int
updateRemFromAdjacencyList(graph G, int v, int *visited);

int *
trimG(graph G, int *vertList, int *trimmed, int *procVerts);

/* Return in or out degree */
int
degree(graph G,
       int v, /* The node */
       enum adjacencies dir
       );

int *
getBestSolutionBuffer(graph G);

#ifdef __cplusplus
}
#endif

#endif /* _GRAPH_H */
