/* #include <stdlib.h> */
#include <stdio.h>
#include <string.h>
#include <bsd/stdlib.h>

int l = 10; /* Number of nodes in the minimal local optimum */
int f = 2; /* Multiplication factor global optimum is f*local in size */
int d = 4; /* Necessary penalty. Need to allow for d-1 nodes to decrease to
 have a chance of reaching the global optimum. */
int k = 4; /* Repetitions of local optimums */

static int *
permutation(int n)
{
  int *A = (int *)malloc(n*sizeof(int));

  for(int i = 0; i < n; i++)
    A[i] = i;

  int t;
  for(int i = n; 1 < i; ){
    int j = arc4random_uniform(i);
    i--;

    t = A[i];
    A[i] = A[j];
    A[j] = t;
  }

  return A;
}

static void
shuffle(int e,
	int *E
	)
{
  for(int i = e; 1 < i; ){
    int j = arc4random_uniform(i);
    i--;

    int t[2];
    memcpy(t, &E[2*i], 2*sizeof(int));
    memcpy(&E[2*i], &E[2*j], 2*sizeof(int));
    memcpy(&E[2*j], t, 2*sizeof(int));
  }
}

int
main(int argc, char **argv)
{
  if(1 < argc)
    sscanf(argv[1], "%d", &l);
  if(2 < argc)
    sscanf(argv[2], "%d", &f);
  if(3 < argc)
    sscanf(argv[3], "%d", &d);
  if(4 < argc)
    sscanf(argv[4], "%d", &k);

  int v = l*(k+f);
  /* int e = 2*l*k*f*d+l*k*(k-1); */
  int e = l*k*(2*f*d+k-1);
  int ec = 0;

  int (*E)[2] = malloc(e*2*sizeof(int));

  printf("%d %d\n", v, e);

  for(int i = 0; i < l; i++){
    for(int j = 0; j < k; j++){
      for(int jj = j+1; jj < k; jj++){
        E[ec][0] = i+j*l;
        E[ec][1] = i+jj*l;
        ec++;

        E[ec][1] = i+j*l;
        E[ec][0] = i+jj*l;
        ec++;
      }
    } /* Do complete graph on local choices */

    for(int jj = 0; jj < k; jj++){
      for(int j = 0; j < d*f; j++){
        E[ec][0] = i+jj*l;
        E[ec][1] = l*k + (i*f + j)%(f*l);
        ec++;

        E[ec][1] = i+jj*l;
        E[ec][0] = l*k + (i*f + j)%(f*l);
        ec++;
      }
    }
  }

  int *Pv = permutation(v);
  for(int ec = 0; ec < e; ec++){
    E[ec][0] = Pv[E[ec][0]];
    E[ec][1] = Pv[E[ec][1]];
  }
  shuffle(e, &E[0][0]);
  free(Pv);

  for(int ec = 0; ec < e; ec++){
    printf("%d ", E[ec][0]);
    printf("%d\n", E[ec][1]);
  }

  free(E);

  return 0;
}
