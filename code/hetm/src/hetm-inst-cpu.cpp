#include <stdio.h>
#include <stdlib.h>

#include "hetm-log.h"
#include "stm-wrapper.h"
#include "hetm.cuh"

__thread HETM_LOG_T *HeTM_log;
__thread void* volatile HeTM_bufAddrs[HETM_BUFFER_MAXSIZE];
__thread uintptr_t HeTM_bufVal[HETM_BUFFER_MAXSIZE];
__thread uintptr_t HeTM_bufVers[HETM_BUFFER_MAXSIZE];
__thread void* volatile HeTM_bufAddrs_reads[HETM_BUFFER_MAXSIZE];
__thread size_t HeTM_ptr_reads;
__thread size_t HeTM_ptr;
__thread uintptr_t HeTM_version;

__thread HETM_LOG_T *stm_thread_local_log = NULL;

// void *stm_devMemPoolBackupBmap[HETM_NB_DEVICES];
// void *stm_devMemPoolBmap[HETM_NB_DEVICES];
void *stm_baseMemPool[HETM_NB_DEVICES];

void *stm_rsetCPU; // only for HETM_BMAP_LOG
void *stm_rsetCPUCache; // only for HETM_BMAP_LOG
size_t stm_rsetCPUCacheBits;

void *stm_wsetCPU; // only for HETM_BMAP_LOG
void *stm_wsetCPUCache; // only for HETM_BMAP_LOG
void *stm_wsetCPUCacheConfl; // only for HETM_BMAP_LOG
void *stm_wsetCPUCacheConfl3; // only for HETM_BMAP_LOG
void *stm_wsetCPUCache_x64[HETM_NB_DEVICES]; // bytes are in different cache lines
size_t stm_wsetCPUCacheBits; // only for HETM_BMAP_LOG

__thread chunked_log_node_s *chunked_log_node_recycled[SIZE_OF_FREE_NODES];
__thread unsigned long chunked_log_free_ptr = 0;
__thread unsigned long chunked_log_alloc_ptr = 0;
__thread unsigned long chunked_log_free_r_ptr = 0;
__thread unsigned long chunked_log_alloc_r_ptr = 0;
long *hetm_batchCount;

HETM_LOG_T* stm_log_init()
{
  HETM_LOG_T *res = NULL;
  __sync_synchronize();
  return res;
};

#ifdef BMAP_ENC_1BIT
#define SET_POS_CPU(_addr, _bm, _val) BM_SET_POS_CPU(_addr, _bm)
#else
#define SET_POS_CPU(_addr, _bm, _val) ByteM_SET_POS(_addr, _bm, _val)
#endif /* BMAP_ENC_1BIT */

void
stm_log_newentry(HETM_LOG_T *log, long* pos, int val, long vers)
{
#if !defined(HETM_CMP_TYPE) || HETM_CMP_TYPE != HETM_CMP_DISABLED
  // TODO: only writing in dev0 need to copy from dev0 CPU data to other GPUs
  long posBmap = BMAP_CONVERT_ADDR(stm_baseMemPool[0], pos, 2);
  long posBmap_cache = BMAP_CONVERT_ADDR(stm_baseMemPool[0], pos, CACHE_GRANULE_BITS+2);
  // printf("WRITE_LOG pos %li cache pos %li (pos=%p)\n", posBmap, posBmap_cache, pos);
  SET_POS_CPU(posBmap, stm_wsetCPU, *hetm_batchCount);
  SET_POS_CPU(posBmap_cache, stm_wsetCPUCache, *hetm_batchCount);
  // memman_access_addr_gran(stm_wsetCPU, stm_baseMemPool[0], pos, 1, 2/*4B*/, *hetm_batchCount);
  // memman_access_addr_gran(stm_wsetCPUCache, stm_baseMemPool[0], pos, 1, CACHE_GRANULE_BITS+2/*4B*/, *hetm_batchCount);
#endif
}

void
stm_log_read_entry(long* pos)
{
#if !defined(HETM_CMP_TYPE) || HETM_CMP_TYPE != HETM_CMP_DISABLED
  long posBmap = BMAP_CONVERT_ADDR(stm_baseMemPool[0], pos, 2);
  long posBmap_cache = BMAP_CONVERT_ADDR(stm_baseMemPool[0], pos, CACHE_GRANULE_BITS+2);
  // printf("READ_LOG pos %li cache pos %li\n", posBmap, posBmap_cache);
  SET_POS_CPU(posBmap, stm_rsetCPU, *hetm_batchCount);
  SET_POS_CPU(posBmap_cache, stm_rsetCPUCache, *hetm_batchCount);
  // memman_access_addr_gran(stm_rsetCPU, stm_baseMemPool[0], pos, 1, 2/*4B*/, *hetm_batchCount);
  // memman_access_addr_gran(stm_rsetCPUCache, stm_baseMemPool[0], pos, 1,
  //   CACHE_GRANULE_BITS+2/*cache_gran+4B*/, *hetm_batchCount);
#endif
}

