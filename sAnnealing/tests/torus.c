#include <stdio.h>
#include <string.h>
#include <bsd/stdlib.h>

int side = 10;

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

static int
s2i(int i, int j)
{
  return i*side+j;
}

int
main(int argc, char **argv)
{
  sscanf(argv[1], "%d", &side);

  int v = side*side;
  int e = 2*v;
  int ec = 0; /* Counter */

  int (*E)[2] = malloc(e*2*sizeof(int));

  printf("%d %d\n", v, e);

  for(int i = 0; i < side; i++){
    for(int j = 0; j < side; j++){
      E[ec][0] = s2i(i,j);
      E[ec][1] = s2i(i, (j+1)%side);
      ec++;

      E[ec][0] = s2i(i,j);
      E[ec][1] = s2i((i+1)%side, j);
      ec++;
    }
  }

  int *Pv = permutation(v);

  for(int ec = 0; ec < e; ec++){
    E[ec][0] = Pv[E[ec][0]];
    E[ec][1] = Pv[E[ec][1]];
  }
  shuffle(e, &E[0][0]);

  for(int ec = 0; ec < e; ec++){
    printf("%d ", E[ec][0]);
    printf("%d\n", E[ec][1]);
  }

  free(Pv);
  free(E);

  return 0;
}
