#ifndef SETUP_KERNELS_H_
#define SETUP_KERNELS_H_

#include "knlman.hpp"

#define ARCH             FERMI
#define DEVICE_ID        0

// TODO: find a way of nuking this:
// --> previous philosophy: retry until the error goes aways --> should simply abort
#define CHECK_ERROR_CONTINUE(cuda_call) \
  cudaStatus = cuda_call; \
  if (cudaStatus != cudaSuccess) { \
    printf("Error " #cuda_call " \n" __FILE__ ":%i\n   > %s\n", \
      __LINE__, cudaGetErrorString(cudaStatus)); \
    /*goto Error;*/ \
    continue; \
  } \
//

int HeTM_bankTx_cpy_IO();
int HeTM_teardown_bankTx();

int HeTM_setup_memcdReadTx(int nbBlocks, int nbThreads);
int HeTM_setup_memcdWriteTx(int nbBlocks, int nbThreads);
int HeTM_teardown_memcdReadTx();
int HeTM_teardown_memcdWriteTx();

int HeTM_setup_checkTxCompressed();
int HeTM_teardown_checkTxCompressed();

int HeTM_setup_checkTxExplicit();
int HeTM_teardown_checkTxExplicit();

int HeTM_setup_finalTxLog2();
int HeTM_teardown_finalTxLog2();

#endif /* SETUP_KERNELS_H_ */
