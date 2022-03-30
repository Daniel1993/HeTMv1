#include "zipf_dist.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
  if (argc != 4) {
    printf("Usage: \n\tjava Main NB_ITEMS PARAM SIZE");
    exit(-1);
  }

  long nb_items = atol(argv[1]);
  double  param = atof(argv[2]);
  long     size = atol(argv[3]);

  zipf_setup(nb_items, param);

  for (int i = 0; i < size; ++i) {
    printf("%lu\n", zipf_gen());
  }
}
