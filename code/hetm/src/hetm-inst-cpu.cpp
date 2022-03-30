#include <stdio.h>
#include <stdlib.h>

#include "hetm-log.h"
#include "stm-wrapper.h"
#include "hetm.cuh"

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

void
stm_log_newentry(HETM_LOG_T *log, long* pos, int val, long vers)
{
  // TODO: only writing in dev0 need to copy from dev0 CPU data to other GPUs

  ByteM_SET_POS(BMAP_CONVERT_ADDR(stm_baseMemPool[0], pos, 2), stm_wsetCPU, *hetm_batchCount);
  ByteM_SET_POS(BMAP_CONVERT_ADDR(stm_baseMemPool[0], pos, CACHE_GRANULE_BITS+2), stm_wsetCPUCache, *hetm_batchCount);

  // memman_access_addr_gran(stm_wsetCPU, stm_baseMemPool[0], pos, 1, 2/*4B*/, *hetm_batchCount);
  // memman_access_addr_gran(stm_wsetCPUCache, stm_baseMemPool[0], pos, 1, CACHE_GRANULE_BITS+2/*4B*/, *hetm_batchCount);
}

void
stm_log_read_entry(long* pos)
{
  ByteM_SET_POS(BMAP_CONVERT_ADDR(stm_baseMemPool[0], pos, 2), stm_rsetCPU, *hetm_batchCount);
  ByteM_SET_POS(BMAP_CONVERT_ADDR(stm_baseMemPool[0], pos, CACHE_GRANULE_BITS+2), stm_rsetCPUCache, *hetm_batchCount);

  // memman_access_addr_gran(stm_rsetCPU, stm_baseMemPool[0], pos, 1, 2/*4B*/, *hetm_batchCount);
  // memman_access_addr_gran(stm_rsetCPUCache, stm_baseMemPool[0], pos, 1,
  //   CACHE_GRANULE_BITS+2/*cache_gran+4B*/, *hetm_batchCount);
}

