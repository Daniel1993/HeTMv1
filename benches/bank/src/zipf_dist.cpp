#include "zipf_dist.h"
#include "zipf_dist.hpp"

using namespace std;

static mt19937 gen;
// 1st item is generated 10% of the times
static zipf_distribution<int, double> *dist = nullptr;

void zipf_setup(unsigned long nbItems, double param)
{
  if (dist != nullptr) delete dist;
  dist = new zipf_distribution<int, double>(nbItems, param);
}

unsigned long zipf_gen()
{
  return (*dist)(gen);
}
