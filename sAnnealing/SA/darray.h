#ifndef _DARRAY_H
#define _DARRAY_H

typedef struct darray *darray;

darray
allocDA(void
	);

darray
dupDA(darray d
	);

darray
cpyDA(darray dst,
      darray src
	   );

void
freeDA(darray d
       );

void
pushDA(darray d,
       int i
       );

/* Dump the array into a buffer */
void
dumpDA(darray d,
       int *B
       );

void
expandDA(darray d,
	 int m
	 );

void
resetDA(darray d
	);

void
resetIterator(darray d
	      );

int
hasNextDA(darray d
	  );

int
getNextDA(darray d
	  );

#endif /* _DARRAY_H */
