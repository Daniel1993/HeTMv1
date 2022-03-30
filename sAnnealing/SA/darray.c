#include <stdlib.h>
#include <string.h>

#include "darray.h"

struct darray{
  int *A;
  int a; /* Space alloced for array */
  int n; /* Number or elements in array */
  int i; /* Current iterator position */
};

darray
allocDA(void
	)
{
  darray d = (darray) malloc(sizeof(struct darray));

  d->i = 0;
  d->n = 0;
  d->a = 4;
  d->A = (int *)malloc(d->a*sizeof(int));

  return d;
}

darray
dupDA(darray d
	)
{
  darray dup = (darray) malloc(sizeof(struct darray));
  d->A = NULL;
  return cpyDA(dup, d);
}

darray
cpyDA(darray dst,
      darray src
	   )
{
  if (!dst) return dupDA(src);
  
  dst->i = src->i;
  dst->n = src->n;
  dst->a = src->a;
  dst->A = (int*)realloc(dst->A, dst->a*sizeof(int));
  memcpy(dst->A, src->A, src->a*sizeof(int));

  return dst;
}

void
freeDA(darray d
       )
{
  free(d->A);
  free(d);
}

void
pushDA(darray d,
       int i
       )
{
  if(d->a == d->n){
    d->a *= 2;
    d->A = realloc(d->A, d->a*sizeof(int));
  }

  d->A[d->n] = i;
  d->n++;
}

/* Dump the array into a buffer */
void
dumpDA(darray d,
       int *B
       )
{
  memcpy(B, d->A, d->n*sizeof(int));
}

void
expandDA(darray d,
	 int m
	 )
{
  if(m > d->a){
    d->a *= 2;
    if(m > d->a)
      d->a = m;

    d->A = realloc(d->A, d->a*sizeof(int));
  }
}

void
resetDA(darray d
	)
{
  d->n = 0;
}

void
resetIterator(darray d
	      )
{
  d->i = 0;
}

int
hasNextDA(darray d
	  )
{
  return d->i < d->n;
}

int
getNextDA(darray d
	  )
{
  int r = -1;

  if(d->i < d->n){
    r = d->A[d->i];
    d->i++;
  }

  return r;
}
