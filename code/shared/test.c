#include "input_handler.h"

#include <stdio.h>
#include <stdlib.h>

static int print_input(const char *arg)
{
  char buffer[128];
  input_getString((char*)arg, buffer);
  printf("%12s :  %s\n", arg, buffer);
  return 0;
}

int main(int argc, char **argv)
{
  input_parse(argc, argv);
  input_foreach(print_input);
  return EXIT_SUCCESS;
}
