#ifndef MURMURHASH2_H_GUARD
#define MURMURHASH2_H_GUARD

#include "murmurhash2-internal.cuh"

__host__ __device__ unsigned int
murmurhash2(const void * key, int len, const unsigned int seed)
{ return murmurhash2_i(key, len, seed); }

#endif /* MURMURHASH2_H_GUARD */
