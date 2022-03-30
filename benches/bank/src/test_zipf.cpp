#include "zipf_dist.hpp"

using namespace std;

int main ()
{
  mt19937 gen;
  zipf_distribution<int, double> dist(1000);

  for (int i = 0; i < 10000; ++i) {
    printf ("%i\n", dist(gen));
  }

  return 0;
}
