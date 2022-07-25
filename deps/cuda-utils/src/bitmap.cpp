#include "bitmap.hpp"

#define BREAK_AT 20

int
bitmap_print(
  unsigned short *bitmap,
  size_t size,
  FILE *fp
) {
  int i, bitsToOne = 0, bitsToZero = 0, j = 0;
  fprintf(fp, "bitmap (size=%zu): \n", size);
  for (i = size / sizeof(short) - 1; i >= 0 ; --i) {
    unsigned short bitmapVal = bitmap[i];
    int popcount = __builtin_popcount(bitmapVal);
    bitsToOne  += popcount;
    bitsToZero += sizeof(short) * 8 - popcount;
    fprintf(fp, "%04x", bitmapVal);
    if (++j % BREAK_AT == 0) fprintf(fp, "\n");
  }
  fprintf(fp, "\n (%i 1's, %i 0's)\n", bitsToOne, bitsToZero);
  return 0;
}
