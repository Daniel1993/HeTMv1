#include "hetm-log.h"
#include "hetm-cmp-kernels.cuh"
#include "hetm.cuh"

// Accessible globally
static HeTM_knl_global_s HeTM_knl_global[HETM_NB_DEVICES];

// ---------------------- EXPLICIT
static size_t explicitLogBlock = 0;
// -------------------------------

void HeTM_set_explicit_log_block_size(size_t size)
{
  explicitLogBlock = size;
}

size_t HeTM_get_explicit_log_block_size()
{
  return explicitLogBlock;
}

// HETM_BMAP_LOG requires a specific kernel

__global__ void interGPUConflDetect(HeTM_knl_cmp_args_s args, size_t offset)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  int sizeWSet = args.sizeWSet; // TODO: size of the chunk

  // offset is needed to chop comparison in chunks (cache granularity)
  id += offset;
  if (id >= sizeWSet) return; // this thread has nothing to do

  int devId = args.devId;
  int otherDevId = args.otherDevId;
  int nbGPUs = args.nbOfGPUs;
  int batchCount = args.batchCount;
  
  // TODO: run the BitmapCache first
  unsigned char *my_rset = (unsigned char*)args.knlGlobal.devRSet;
  unsigned char *remote_wset = (unsigned char*)args.knlGlobal.devWSet[otherDevId];
  unsigned char *confl_mat = (unsigned char*)args.knlGlobal.localConflMatrix[devId];

  if (my_rset[id] == batchCount && remote_wset[id] == batchCount) {
    // int CPUid = nbGPUs; // CPU has the last ID
    int coord_l = (nbGPUs+1)*devId + otherDevId;
    // printf("[GPU%i] conflict with GPU%i at pos %i (my_rset=%p, remote_wset=%p)\n", devId, otherDevId, id, my_rset, remote_wset);
    confl_mat[coord_l] = 1;
  }
}

__global__ void CPUGPUConflDetect(HeTM_knl_cmp_args_s args, size_t offset)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  int sizeWSet = args.sizeWSet; // TODO: size of the chunk

  // offset is needed to chop comparison in chunks (cache granularity)
  id += offset;
  if (id >= sizeWSet) return; // this thread has nothing to do

  int devId = args.devId;
  int otherDevId = args.otherDevId; // CPU id == args.nbOfGPUs
  int nbGPUs = args.nbOfGPUs;
  int batchCount = args.batchCount;
  
  // TODO: run the BitmapCache first
  unsigned char *my_rset = (unsigned char*)args.knlGlobal.devRSet;
  unsigned char *cpu_wset = (unsigned char*)args.knlGlobal.hostWSet;
  unsigned char *confl_mat = (unsigned char*)args.knlGlobal.localConflMatrix[devId];

  if (my_rset[id] == batchCount && cpu_wset[id] == batchCount) {
    // int CPUid = nbGPUs; // CPU has the last ID
    int coord_l = (nbGPUs+1)*devId + otherDevId;
    // printf("[GPU%i] conflict with CPU at pos %i (my_rset=%p, cpu_wset=%p)\n", devId, id, my_rset, cpu_wset);
    confl_mat[coord_l] = 1;
  }
}

__global__ void CPUrsGPUwsConflDetect(HeTM_knl_cmp_args_s args, size_t offset)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  int sizeWSet = args.sizeWSet; // TODO: size of the chunk

  // offset is needed to chop comparison in chunks (cache granularity)
  id += offset;
  if (id >= sizeWSet) return; // this thread has nothing to do

  int devId = args.devId;
  int otherDevId = args.otherDevId; // CPU id == args.nbOfGPUs
  int nbGPUs = args.nbOfGPUs;
  int batchCount = args.batchCount;
  
  // TODO: run the BitmapCache first
  unsigned char *host_rset = (unsigned char*)args.knlGlobal.hostRSet;
  unsigned char *my_wset = (unsigned char*)args.knlGlobal.devWSet[devId];
  unsigned char *confl_mat = (unsigned char*)args.knlGlobal.localConflMatrix[devId];

  if (host_rset[id] == batchCount && my_wset[id] == batchCount)
  {
    // int CPUid = nbGPUs; // CPU has the last ID
    int coord_l = (nbGPUs+1)*otherDevId + devId;
    // printf("[GPU%i] conflict with CPU at pos %i (my_rset=%p, cpu_wset=%p)\n", devId, id, my_rset, cpu_wset);
    confl_mat[coord_l] = 1;
  }
}

__global__ void HeTM_knl_checkTxBitmapCache(HeTM_knl_cmp_args_s args)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;

  // int devId = args.devId;
  size_t nbChunks = args.knlGlobal.hostWSetChunks;

  if (id >= nbChunks) {
    return; // the thread has nothing to do
  }

  // unsigned char *rset = (unsigned char*)args.knlGlobal.devRSet;
  unsigned char *rset = (unsigned char*)args.knlGlobal.hostWSetCacheConfl;
  unsigned char *rwsetConfl = (unsigned char*)args.knlGlobal.hostWSetCacheConfl2;
  unsigned char *unionWS = (unsigned char*)args.knlGlobal.hostWSetCacheConfl3;
  unsigned char *GPUwset = (unsigned char*)args.knlGlobal.GPUwsBmap;
  unsigned char *wset = (unsigned char*)args.knlGlobal.hostWSetCache;

  // int cacheId = id >> wsetBits;
  // unsigned char isNewWrite = wset[cacheId] == args.batchCount;
  unsigned char isNewWrite = wset[id] == args.batchCount;

  // checks for a chunck of CPU dataset if it was read in GPU (4096 items)
  // memman_bmap_s *key_bmap = (memman_bmap_s*)HeTM_knl_global.devMemPoolBackupBmap;
  // char *bytes = key_bmap->dev;
  char *bytes = (char*)args.knlGlobal.devMemPoolBackupBmap;

  if (isNewWrite) { // TODO: divergency
    bytes[id] = 1; /*args.batchCount*/
  }
  int isConfl = (rset[id] == args.batchCount) && isNewWrite;
  int isConflW = (GPUwset[id] && isNewWrite) ? 1 : 0;
  unionWS[id] = (GPUwset[id] || isNewWrite) ? 1 : 0;

  // the rset cache is also used as conflict detection
  rset[id] = isConfl ? args.batchCount : 0;
  rwsetConfl[id] = isConflW ? 1 : 0;

  // printf("id=%i rset[id]=%i rwsetConfl[id]=%i\n", id, (int)rset[id], (int)rwsetConfl[id]);

  if (isConfl) {
    *args.knlGlobal.isInterConfl = 1;
    // printf("GPU conflict found\n");
    // ((unsigned char*)(args.knlGlobal.hostWSetCacheConfl))[cacheId] = args.batchCount;
  }
}

__global__ void HeTM_knl_checkTxBitmap(HeTM_knl_cmp_args_s args, size_t offset)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  int idPlusOffset = id + offset;
  // int devId = args.devId;

  // TODO: these must be the same
  int sizeWSet = args.sizeWSet; /* Size of the host log */
  // int sizeRSet = args.sizeRSet; /* Size of the device log */

  if (idPlusOffset >= sizeWSet) return; // the thread has nothing to do

  unsigned char *rset = (unsigned char*)args.knlGlobal.devRSet;
  unsigned char *wset = (unsigned char*)args.knlGlobal.hostWSet;

  // TODO: use shared memory
  unsigned char isNewWrite = wset[idPlusOffset] == args.batchCount;
  unsigned char isConfl = (rset[idPlusOffset] == args.batchCount) && isNewWrite;

  if (isConfl) {
    // printf("[%i] found conflict wset[%i]=%i\n", id, idPlusOffset, (int)wset[idPlusOffset]);
    *args.knlGlobal.isInterConfl = 1;
  }
  // if (isNewWrite) {
  //   a[id] = b[id];
  // }
}

__global__ void HeTM_knl_checkTxBitmap_Explicit(HeTM_knl_cmp_args_s args)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  int sizeRSet = args.sizeRSet / sizeof(PR_GRANULE_T);
  // int devId = args.devId;
  unsigned char *wset = (unsigned char*)args.knlGlobal.hostWSet; /* Host log */
  int *devLogR = (int*)args.knlGlobal.devRSet;  /* GPU read log */
  int jobSize = sizeRSet / (blockDim.x*gridDim.x);
  size_t GPUIndexMaxVal = args.knlGlobal.nbGranules;
  int i;
  jobSize++; // --> case mod is not zero
  for (i = 0; i < jobSize; ++i) {
    // TODO: review formula
    int idRSet = id*jobSize + i;
    if (idRSet >= sizeRSet || *args.knlGlobal.isInterConfl) {
      return;
    }
    int GPUIndex = devLogR[idRSet] - 1;
    // TODO
    if (GPUIndex > -1 && GPUIndex < GPUIndexMaxVal && wset[GPUIndex] == 1) {
      *args.knlGlobal.isInterConfl = 1;
    }
  }
}

__global__ void HeTM_knl_writeTxBitmap(HeTM_knl_cmp_args_s args, size_t offset)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x + offset;
  // int devId = args.devId;

  // TODO: these must be the same
  int sizeWSet = args.sizeWSet; /* Size of the host log */
  // int sizeRSet = args.sizeRSet; /* Size of the device log */

  if (sizeWSet < id) return; // the thread has nothing to do

  // unsigned char *rset = (unsigned char*)args.knlGlobal.devRSet;
  unsigned char *wset = (unsigned char*)args.knlGlobal.hostWSet;
  PR_GRANULE_T *mempool = (PR_GRANULE_T*)args.knlGlobal.devMemPoolBasePtr;
  PR_GRANULE_T *backup  = (PR_GRANULE_T*)args.knlGlobal.devMemPoolBackupBasePtr;

  // TODO: use shared memory
  unsigned char isNewWrite = wset[id];

  long condToIgnore = !isNewWrite;
  condToIgnore = ((condToIgnore | (-condToIgnore)) >> 63);
  // long maskIgnore = -condToIgnore;
  long maskIgnore = condToIgnore; // TODO: for some obscure reason this is -1

  // applies -1 if address is invalid OR the index if valid
  mempool[id] = (maskIgnore & mempool[id]) | ((~maskIgnore) & backup[id]);
  // if (isNewWrite) {
  //   a[id] = b[id];
  // }
}


HeTM_knl_global_s *HeTM_get_global_arg(int devId)
{
  return &HeTM_knl_global[devId];
}

void HeTM_set_global_arg(int devId, HeTM_knl_global_s arg)
{
  memcpy(&HeTM_knl_global[devId], &arg, sizeof(HeTM_knl_global_s));
  // CUDA_CHECK_ERROR(
  //   cudaMemcpyToSymbol(args.knlGlobal, &arg, sizeof(HeTM_knl_global_s)),
  //   "");
  // printf("HeTM_knl_global.hostWSet = %p\n", HeTM_knl_global.hostWSet);
}
