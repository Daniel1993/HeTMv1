#ifndef BITMAP_H_GUARD_
#define BITMAP_H_GUARD_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdint.h>

// TODO: use this or rand_r(&seed) instead of erand48
#define RAND_R_FNC(seed) ({ \
	uint64_t next = seed; \
	uint64_t result; \
	next *= 1103515245; \
	next += 12345; \
	result = (uint64_t) (next >> 16) & (2048-1); \
	next *= 1103515245; \
	next += 12345; \
	result <<= 10; \
	result ^= (uint64_t) (next >> 16) & (1024-1); \
	next *= 1103515245; \
	next += 12345; \
	result <<= 10; \
	result ^= (uint64_t) (next >> 16) & (1024-1); \
	seed = next; \
	result; \
})

// TODO: delete ........
#define LOG2_16BITS      4
#define LOG2_16BITS_MASK 0b1111
// .....................
#define LOG2_32BITS      5
#define LOG2_32BITS_MASK 0b11111

int bitmap_print(unsigned short *bitmap, size_t size, FILE *fp);

#ifdef BMAP_ENC_1BIT

#define BM_SET_POS_GPU(_pos, _bm) ({\
  unsigned long evalPos = (_pos); \
	unsigned *bmLoc = ((unsigned*)_bm)+(evalPos>>LOG2_32BITS); \
	unsigned bit = (1L<<(LOG2_32BITS_MASK&evalPos)); \
	atomicOr(bmLoc, bit); \
})\
//
#define BM_SET_POS_CPU(_pos, _bm) ({\
  unsigned long evalPos = (_pos); \
	unsigned *bmLoc = ((unsigned*)_bm)+(evalPos>>LOG2_32BITS); \
	unsigned bit = (1L<<(LOG2_32BITS_MASK&evalPos)); \
	__atomic_or_fetch(bmLoc, bit, __ATOMIC_RELEASE); \
})\
//
#define BM_GET_POS(_pos, _bm) ({ \
	unsigned *BM_bitmap = (unsigned*)(_bm); \
	unsigned long bmIdx = _pos >> LOG2_32BITS; \
	unsigned long withinShort = LOG2_32BITS_MASK & _pos; \
	unsigned long withinShortIdx = 1L << withinShort; \
	BM_bitmap[bmIdx] & withinShortIdx; \
}) \
//

// #define ByteM_SET_POS(_pos, _bm, _valToSet) // TODO
// #define ByteM_GET_POS(_pos, _bm) // TODO
// (((((uint32_t*)(_bm))+((_pos)>>LOG2_32BITS))&(1L<<(LOG2_32BITS_MASK&(_pos)))) == (1L<<(LOG2_32BITS_MASK&(_pos)))) 
#define BMAP_CHECK_POS(_bm, _pos, _val) \
	(BM_GET_POS(_pos, _bm) == (1L << (LOG2_32BITS_MASK & _pos)) )\
//

#else
#define ByteM_SET_POS(_pos, _bm, _valToSet) \
	((unsigned char*)_bm)[_pos] = (unsigned char)(_valToSet)
//
#define ByteM_GET_POS(_pos, _bm) \
	((unsigned char*)_bm[_pos])
//
#define BMAP_CHECK_POS(_bmap, _pos, _val) \
	(((unsigned char*)(_bmap))[_pos] == (unsigned char)(_val)) \
//
#endif

#define BMAP_CONVERT_ADDR(_base, _addr, _enc_bits) \
	((uintptr_t)(_addr)-(uintptr_t)(_base) >> (_enc_bits))

// TODO: create a API for the bitmap
// uses the given buffer as bitmap
// int bitmap_init(void *buffer, size_t bufSize, size_t gran);

// does nothing
// int bitmap_destroy(void*);


#ifdef __cplusplus
}
#endif

#endif /* BITMAP_H_GUARD_ */
