#include "hetm-log.h"
#include "hetm-cmp-kernels.cuh"
#include "hetm.cuh"

// Accessible globally
static HeTM_knl_global_s HeTM_knl_global[HETM_NB_DEVICES];

// HETM_BMAP_LOG requires a specific kernel

__global__ void interGPUConflDetect(HeTM_knl_cmp_args_s args, char *remote_wset, char *my_rset, size_t offset)
{
  unsigned long id = blockIdx.x*blockDim.x+threadIdx.x;
  int sizeWSet = args.sizeWSet; // TODO: size of the chunk

  int devId = args.devId;
  int otherDevId = args.otherDevId;
  int nbGPUs = args.nbOfGPUs;
  int batchCount = args.batchCount;

  // if (threadIdx.x == 0)
  //   printf("devId=%d otherDevId=%i id=%lu\n", devId, otherDevId, id);

  // offset is needed to chop comparison in chunks (cache granularity)
  id += offset;
  if (id >= sizeWSet) return; // this thread has nothing to do
  
  // TODO: run the BitmapCache first
  // unsigned char *my_rset = (unsigned char*)args.knlGlobal.devRSet;
  // unsigned char *remote_wset = (unsigned char*)args.knlGlobal.devWSet[otherDevId];
  unsigned char *confl_mat = (unsigned char*)args.knlGlobal.localConflMatrix[devId];

  int inMyRSet = 0;
  int inRemoteWSet = 0;
  //
#ifdef BMAP_ENC_1BIT
  inMyRSet = BMAP_CHECK_POS(my_rset, id, batchCount);
  inRemoteWSet = BMAP_CHECK_POS(remote_wset, id, batchCount);
#else /* BMAP_ENC_1BIT */
  inMyRSet = (my_rset[id] == batchCount);
  inRemoteWSet = (remote_wset[id] == batchCount);
#endif /* BMAP_ENC_1BIT */

  if (inMyRSet && inRemoteWSet)
  {
    // int CPUid = nbGPUs; // CPU has the last ID
    int coord_l = (nbGPUs+1)*devId + otherDevId;
    // printf("[GPU%i] conflict with GPU%i at pos %i (my_rset=%p, remote_wset=%p)\n", devId, otherDevId, id, my_rset, remote_wset);
    confl_mat[coord_l] = 1;
  }
}

__global__ void CPUGPUConflDetect(HeTM_knl_cmp_args_s args, size_t offset)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  int sizeWSet = args.sizeWSet;

  // offset is needed to chop comparison in chunks (cache granularity)
  id += offset;
  if (id >= sizeWSet) return; // this thread has nothing to do

  int devId = args.devId;
  int otherDevId = args.otherDevId; // CPU id == args.nbOfGPUs
  int nbGPUs = args.nbOfGPUs;
  int batchCount = args.batchCount;
  
  unsigned char *my_rset = (unsigned char*)args.knlGlobal.devRSet;
  unsigned char *cpu_wset = (unsigned char*)args.knlGlobal.hostWSet;
  unsigned char *confl_mat = (unsigned char*)args.knlGlobal.localConflMatrix[devId];

  int inMyRSet = 0;
  int inCPUWSet = 0;

#ifdef BMAP_ENC_1BIT
  inMyRSet = BMAP_CHECK_POS(my_rset, id, batchCount);
  inCPUWSet = BMAP_CHECK_POS(cpu_wset, id, batchCount);
#else /* BMAP_ENC_1BIT */
  inMyRSet = (my_rset[id] == batchCount);
  inCPUWSet = (cpu_wset[id] == batchCount);
#endif /* BMAP_ENC_1BIT */

  // if (id == 0)
  //   printf("inMyRSet == %i inCPUWSet == %i\n", inMyRSet, inCPUWSet);

  if (inMyRSet && inCPUWSet) {
    // int CPUid = nbGPUs; // CPU has the last ID
    int coord_l = (nbGPUs+1)*devId + otherDevId;
    /* printf("[GPU%i] CPU_GPU conflict with CPU at pos %i (my_rset=%p, cpu_wset=%p)\n", devId, id, my_rset, cpu_wset); */
    confl_mat[coord_l] = 1;
  }
}

__global__ void CPUrsGPUwsConflDetect(HeTM_knl_cmp_args_s args, size_t offset)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  int sizeWSet = args.sizeWSet; 

  // offset is needed to chop comparison in chunks (cache granularity)
  id += offset;
  if (id >= sizeWSet) return; // this thread has nothing to do

  int devId = args.devId;
  int otherDevId = args.otherDevId; // CPU id == args.nbOfGPUs
  int nbGPUs = args.nbOfGPUs;
  int batchCount = args.batchCount;
  
  unsigned char *host_rset = (unsigned char*)args.knlGlobal.hostRSet;
  unsigned char *my_wset = (unsigned char*)args.knlGlobal.devWSet[devId];
  unsigned char *confl_mat = (unsigned char*)args.knlGlobal.localConflMatrix[devId];

  int inCPURSet = 0;
  int inMyWSet = 0;

#ifdef BMAP_ENC_1BIT
  inCPURSet = BMAP_CHECK_POS(host_rset, id, batchCount);
  inMyWSet = BMAP_CHECK_POS(my_wset, id, batchCount);
#else /* BMAP_ENC_1BIT */
  inCPURSet = (host_rset[id] == batchCount);
  inMyWSet = (my_wset[id] == batchCount);
#endif /* BMAP_ENC_1BIT */

  // if (id == 0)
  //   printf("host_rset = %p my_wset = %p inCPURSet == %i inMyWSet == %i\n",
  //     host_rset, my_wset, inCPURSet, inMyWSet);

  if (inCPURSet && inMyWSet)
  {
    // int CPUid = nbGPUs; // CPU has the last ID
    int coord_l = (nbGPUs+1)*otherDevId + devId;
    /* printf("[GPU%i] CPUrsGPUws conflict with CPU at pos %i (my_wset=%p, cpu_rset=%p)\n", devId, id, my_wset, host_rset); */
    confl_mat[coord_l] = 1;
  }
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
