#include "TestBank.cuh"

// ##########################################

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "helper_timer.h"

#define PR_rand(n) \
	PR_i_rand(args, n) \
//

typedef struct {
  curandState *state;
} pr_args_ext_t;

typedef struct {
  curandState *state;
} pr_dev_buff_ext_t;

#ifdef PR_BEFORE_RUN_EXT
#undef PR_BEFORE_RUN_EXT
#endif

#define PR_BEFORE_RUN_EXT(args) ({ \
	pr_dev_buff_ext_t *cuRandBuf; \
	args->host.pr_args_ext = malloc(sizeof(pr_dev_buff_ext_t)); \
	cuRandBuf = (pr_dev_buff_ext_t*)args->host.pr_args_ext; \
	PR_ALLOC(args->dev.pr_args_ext, sizeof(pr_dev_buff_ext_t)); \
	PR_ALLOC(cuRandBuf->state, PR_blockNum * PR_threadNum * sizeof(curandState)); \
	cudaFuncSetCacheConfig(setupKernel, cudaFuncCachePreferL1); \
	setupKernel <<< PR_blockNum, PR_threadNum >>>(cuRandBuf->state); \
	cudaThreadSynchronize(); \
	PR_CPY_TO_DEV(args->dev.pr_args_ext, args->host.pr_args_ext, sizeof(pr_dev_buff_ext_t)); \
}) \
//

#ifdef PR_AFTER_RUN_EXT
#undef PR_AFTER_RUN_EXT
#endif

#define PR_AFTER_RUN_EXT(args) ({ \
	cudaFree(((pr_dev_buff_ext_t*)args->host.pr_args_ext)->state); \
	cudaFree(args->dev.pr_args_ext); \
	free(args->host.pr_args_ext); \
}) \
//

#include "pr-stm.cuh"

__global__ void setupKernel(curandState *state);
__device__ unsigned PR_i_rand(pr_tx_args_dev_host_s args, unsigned n);

__constant__ PR_DEVICE unsigned PR_seed = 1234;

#include "pr-stm-internal.cuh"

// ##########################################

using namespace std;

static int isInit = 0;

CPPUNIT_TEST_SUITE_REGISTRATION(TestBank);

static PR_ENTRY_POINT void testEmptyTXKernel(PR_globalKernelArgs);
static PR_ENTRY_POINT void testReadKernel(PR_globalKernelArgs);
static PR_ENTRY_POINT void testWriteKernel(PR_globalKernelArgs);
static PR_ENTRY_POINT void testReadWriteKernel(PR_globalKernelArgs);
static PR_ENTRY_POINT void testRandomKernel(PR_globalKernelArgs);

TestBank::TestBank()  { }
TestBank::~TestBank() { }

void TestBank::setUp()
{
  // TODO: need a PR_ call for this
  PR_CHECK_CUDA_ERROR(cudaSetDevice(0), "CUDA GPU not found!");
	if (!isInit) {
		PR_init(1);
		isInit = 1;
	}
}

void TestBank::tearDown()
{
}

void TestBank::TestMacros()
{
	pr_tx_args_s prArgs;

  PR_blockNum = 1;
  PR_threadNum = 1;
	PR_createStatistics(&prArgs);
  PR_run(testEmptyTXKernel, &prArgs, NULL);
	PR_waitKernel();

  CPPUNIT_ASSERT(true);
}

void TestBank::TestRead()
{
  int i;
	const int dataSize = 1024;
	const int resSize = 5*20;
  PR_GRANULE_T data[dataSize], *dataDev;
  PR_GRANULE_T res[resSize], *resDev;
  char msg[1024];
	pr_tx_args_s prArgs;

	cudaMalloc(&dataDev, sizeof(PR_GRANULE_T)*dataSize); // TODO: test errors
	cudaMalloc(&resDev, sizeof(PR_GRANULE_T)*resSize); // TODO: test errors
  memset(data, 0, sizeof(data));
  pr_buffer_s inBuf, outBuf;

  inBuf.buf = (void*)dataDev;
  inBuf.size = sizeof(data);
  outBuf.buf = (void*)resDev;
  outBuf.size = sizeof(res);

  data[5] = 5;
	PR_CPY_TO_DEV(dataDev, data, inBuf.size);
	PR_CPY_TO_DEV(resDev, res, outBuf.size);

  PR_blockNum = 1;
  PR_threadNum = 1;
	PR_createStatistics(&prArgs);
	PR_prepareIO(&prArgs, inBuf, outBuf);
  PR_run(testReadKernel, &prArgs, NULL);
	PR_waitKernel();
	PR_retrieveIO(&prArgs, NULL);
	PR_disposeIO(&prArgs);
	PR_CPY_TO_HOST(res, resDev, outBuf.size);

  for (i = 0; i < 1; ++i) {
    sprintf(msg, "Expected 5 but got %i", res[i]);
    CPPUNIT_ASSERT_MESSAGE(msg, 5 == res[i]);
  }

  PR_blockNum = 1;
  PR_threadNum = 2;
	PR_createStatistics(&prArgs);
	PR_prepareIO(&prArgs, inBuf, outBuf);
  PR_run(testReadKernel, &prArgs, NULL);
	PR_waitKernel();
	PR_retrieveIO(&prArgs, NULL); // statistics
	PR_disposeIO(&prArgs);
	PR_CPY_TO_HOST(res, resDev, outBuf.size);

  for (i = 0; i < 1*2; ++i) {
    sprintf(msg, "Expected 5 but got %i", res[i]);
    CPPUNIT_ASSERT_MESSAGE(msg, 5 == res[i]);
  }

  PR_blockNum = 1;
  PR_threadNum = 20;
	PR_createStatistics(&prArgs);
  PR_run(testReadKernel, &prArgs, NULL);
	PR_waitKernel();
	PR_retrieveIO(&prArgs, NULL); // statistics
	PR_disposeIO(&prArgs);
	PR_CPY_TO_HOST(res, resDev, outBuf.size);

  for (i = 0; i < 1*20; ++i) {
    sprintf(msg, "Expected 5 but got %i", res[i]);
    CPPUNIT_ASSERT_MESSAGE(msg, 5 == res[i]);
  }

  PR_blockNum = 5;
  PR_threadNum = 20;
	PR_createStatistics(&prArgs);
  PR_run(testReadKernel, &prArgs, NULL);
	PR_waitKernel();
	PR_retrieveIO(&prArgs, NULL); // statistics
	PR_disposeIO(&prArgs);
	PR_CPY_TO_HOST(res, resDev, outBuf.size);

  for (i = 0; i < 5*20; ++i) {
    sprintf(msg, "Expected 5 but got %i", res[i]);
    CPPUNIT_ASSERT_MESSAGE(msg, 5 == res[i]);
  }

	cudaFree(inBuf.buf);
	cudaFree(outBuf.buf);
}

void TestBank::TestWrite()
{
  int i;
	const int dataSize = 5*20;
	const int resSize = 5*20;
  PR_GRANULE_T data[dataSize], *dataDev;
  PR_GRANULE_T res[resSize], *resDev;
  char msg[1024];
	pr_tx_args_s prArgs;

	cudaMalloc(&dataDev, sizeof(PR_GRANULE_T)*dataSize); // TODO: test errors
	cudaMalloc(&resDev, sizeof(PR_GRANULE_T)*resSize); // TODO: test errors
  memset(data, 0, sizeof(data));
  pr_buffer_s inBuf, outBuf;

  inBuf.buf = (void*)dataDev;
  inBuf.size = sizeof(data);
  outBuf.buf = (void*)resDev;
  outBuf.size = sizeof(res);

	PR_CPY_TO_DEV(dataDev, data, inBuf.size);
	PR_CPY_TO_DEV(resDev, &res, inBuf.size);

  PR_blockNum = 1;
  PR_threadNum = 1;
	PR_createStatistics(&prArgs);
	PR_prepareIO(&prArgs, inBuf, outBuf);
  PR_run(testWriteKernel, &prArgs, NULL);
	PR_waitKernel();
	PR_retrieveIO(&prArgs, NULL); // statistics
	PR_disposeIO(&prArgs);
	PR_CPY_TO_HOST(data, dataDev, inBuf.size);

  for (i = 0; i < 1; ++i) {
    sprintf(msg, "Expected 5 but got %i", data[i]);
    CPPUNIT_ASSERT_MESSAGE(msg, 5 == data[i]);
  }

  PR_blockNum = 5;
  PR_threadNum = 20;
	PR_createStatistics(&prArgs);
	PR_prepareIO(&prArgs, inBuf, outBuf);
  PR_run(testWriteKernel, &prArgs, NULL);
	PR_waitKernel();
	PR_retrieveIO(&prArgs, NULL); // statistics
	PR_disposeIO(&prArgs);
	PR_CPY_TO_HOST(data, dataDev, inBuf.size);

  for (i = 0; i < 5*20; ++i) {
    sprintf(msg, "Expected 5 but got %i", data[i]);
    CPPUNIT_ASSERT_MESSAGE(msg, 5 == data[i]);
  }

	cudaFree(dataDev);
	cudaFree(resDev);
}

void TestBank::TestReadWrite()
{
  int i;
	const int dataSize = 5*20;
	const int resSize = 5*20;
  PR_GRANULE_T data[5*20], *dataDev;
  PR_GRANULE_T res[5*20], *resDev;
  char msg[1024];
	pr_tx_args_s prArgs;

	cudaMalloc(&dataDev, sizeof(PR_GRANULE_T)*dataSize); // TODO: test errors
	cudaMalloc(&resDev, sizeof(PR_GRANULE_T)*resSize); // TODO: test errors
  memset(data, 0, sizeof(data));
  pr_buffer_s inBuf, outBuf;

  inBuf.buf = (void*)dataDev;
  inBuf.size = sizeof(data);
  outBuf.buf = (void*)resDev;
  outBuf.size = sizeof(res);

	PR_CPY_TO_DEV(dataDev, data, inBuf.size);
	PR_CPY_TO_DEV(resDev, res, outBuf.size);

  PR_blockNum = 1;
  PR_threadNum = 2;
	PR_createStatistics(&prArgs);
	PR_prepareIO(&prArgs, inBuf, outBuf);
  PR_run(testReadWriteKernel, &prArgs, NULL);
	PR_waitKernel();
	PR_retrieveIO(&prArgs, NULL); // statistics
	PR_disposeIO(&prArgs);
	PR_CPY_TO_HOST(res, resDev, outBuf.size);

  for (i = 0; i < 2; ++i) {
    sprintf(msg, "Expected 5 but got %i", res[i]);
    CPPUNIT_ASSERT_MESSAGE(msg, 5 == res[i]);
  }

  PR_blockNum = 5;
  PR_threadNum = 20;
	PR_createStatistics(&prArgs);
	PR_prepareIO(&prArgs, inBuf, outBuf);
  PR_run(testReadWriteKernel, &prArgs, NULL);
	PR_waitKernel();
	PR_retrieveIO(&prArgs, NULL); // statistics
	PR_disposeIO(&prArgs);
	PR_CPY_TO_HOST(res, resDev, outBuf.size);

  for (i = 0; i < 5*20; ++i) {
    sprintf(msg, "Expected 5 but got %i", res[i]);
    CPPUNIT_ASSERT_MESSAGE(msg, 5 == res[i]);
  }

	cudaFree(dataDev);
	cudaFree(resDev);
}

void TestBank::TestRandom()
{
  int i;
  const size_t nbGran = 10000;
  PR_GRANULE_T sum = 0, expected = nbGran * 50;
  PR_GRANULE_T data[nbGran], *dataDev;
  PR_GRANULE_T res[nbGran], *resDev;
  char msg[1024];
	pr_tx_args_s prArgs;

  pr_buffer_s inBuf, outBuf;

	CUDA_CHECK_ERROR(cudaMalloc(&dataDev, sizeof(PR_GRANULE_T)*nbGran), "");
	CUDA_CHECK_ERROR(cudaMalloc(&resDev, sizeof(PR_GRANULE_T)*nbGran), "");

  inBuf.buf = (void*)dataDev;
  inBuf.size = sizeof(data);
  outBuf.buf = (void*)resDev;
  outBuf.size = sizeof(res);

  PR_blockNum = 1;
  PR_threadNum = 1;
  for (i = 0; i < nbGran; ++i) { data[i] = 50; } // reset

	PR_CPY_TO_DEV(dataDev, data, inBuf.size);
	PR_CPY_TO_DEV(resDev, res, outBuf.size);

	PR_createStatistics(&prArgs);
	PR_prepareIO(&prArgs, inBuf, outBuf);
  PR_run(testRandomKernel, &prArgs, NULL);
	PR_waitKernel();
	PR_retrieveIO(&prArgs, NULL); // statistics
	PR_disposeIO(&prArgs);
	PR_CPY_TO_HOST(data, dataDev, inBuf.size);

  sum = 0; // ASSERT
  for (i = 0; i < nbGran; ++i) { sum += data[i]; }
  sprintf(msg, "-1- Expected %i but got %i", expected, data[i]);
  CPPUNIT_ASSERT_MESSAGE(msg, expected == sum);
  printf("(%ix%i) nb_aborts=%lli nb_commits=%lli\n", PR_blockNum,
    PR_threadNum, PR_nbAborts, PR_nbCommits);
  PR_nbAborts = 0;
  PR_nbCommits = 0;

  PR_blockNum = 2;
  PR_threadNum = 5;
  for (i = 0; i < nbGran; ++i) { data[i] = 50; } // reset

	PR_CPY_TO_DEV(dataDev, data, inBuf.size);
	PR_CPY_TO_DEV(resDev, res, outBuf.size);

	PR_createStatistics(&prArgs);
	PR_prepareIO(&prArgs, inBuf, outBuf);
  PR_run(testRandomKernel, &prArgs, NULL);
	PR_waitKernel();
	PR_retrieveIO(&prArgs, NULL); // statistics
	PR_disposeIO(&prArgs);
	PR_CPY_TO_HOST(data, dataDev, inBuf.size);

  sum = 0;
  for (i = 0; i < nbGran; ++i) { sum += data[i]; }
  sprintf(msg, "-2- Expected %i but got %i", expected, data[i]);
  CPPUNIT_ASSERT_MESSAGE(msg, expected == sum);
  printf("(%ix%i) nb_aborts=%lli nb_commits=%lli\n", PR_blockNum,
    PR_threadNum, PR_nbAborts, PR_nbCommits);
  PR_nbAborts = 0;
  PR_nbCommits = 0;

  PR_blockNum = 5;
  PR_threadNum = 20;
  for (i = 0; i < nbGran; ++i) { data[i] = 50; } // reset

	PR_CPY_TO_DEV(dataDev, data, inBuf.size);
	PR_CPY_TO_DEV(resDev, res, outBuf.size);

	PR_createStatistics(&prArgs);
	PR_prepareIO(&prArgs, inBuf, outBuf);
  PR_run(testRandomKernel, &prArgs, NULL);
	PR_waitKernel();
	PR_retrieveIO(&prArgs, NULL); // statistics
	PR_disposeIO(&prArgs);
	PR_CPY_TO_HOST(data, dataDev, inBuf.size);

  sum = 0;
  for (i = 0; i < nbGran; ++i) { sum += data[i]; }
  sprintf(msg, "-3- Expected %i but got %i", expected, data[i]);
  CPPUNIT_ASSERT_MESSAGE(msg, expected == sum);
  printf("(%ix%i) nb_aborts=%lli nb_commits=%lli\n", PR_blockNum,
    PR_threadNum, PR_nbAborts, PR_nbCommits);

	cudaFree(dataDev);
	cudaFree(resDev);
}

static PR_ENTRY_POINT void testReadKernel(PR_globalKernelArgs)
{
  PR_enterKernel();
  // size_t size = sizeof(int); // TODO: variable size, floats, ...
  PR_GRANULE_T *addr = &(((PR_GRANULE_T*)args.inBuf)[5]);
  PR_GRANULE_T rval;
  PR_txBegin();
  rval = PR_read(addr); // do not support reads in loops...
  PR_txCommit();
  // printf("CUDA idx=%i\n", PR_THREAD_IDX);
  ((int*)args.outBuf)[PR_THREAD_IDX] = rval;
  PR_exitKernel();
}

static PR_ENTRY_POINT void testWriteKernel(PR_globalKernelArgs)
{
  PR_enterKernel();
  PR_GRANULE_T *addr = &(((PR_GRANULE_T*)args.inBuf)[PR_THREAD_IDX]);
  // size_t size = sizeof(int); // TODO: variable size, floats, ...
  PR_GRANULE_T writeVal = 5;
  PR_txBegin();
  PR_read(addr); // TODO: need to read before write?
  PR_write(addr, writeVal);
  PR_txCommit();
  PR_exitKernel();
}

static PR_ENTRY_POINT void testReadWriteKernel(PR_globalKernelArgs)
{
  PR_enterKernel();
  PR_GRANULE_T *addr1 = &(((PR_GRANULE_T*)args.inBuf)[PR_THREAD_IDX]);
  PR_GRANULE_T *addr2 = &(((PR_GRANULE_T*)args.outBuf)[PR_THREAD_IDX]);
  // size_t size = sizeof(int); // TODO: variable size, floats, ...
  PR_GRANULE_T writeVal = 5;
  PR_txBegin();
  PR_read(addr1); // TODO: read without checking the value
  PR_txCommit();
  PR_txBegin();
  PR_write(addr2, writeVal); // no read before
  PR_txCommit();
  PR_exitKernel();
}

static PR_ENTRY_POINT void testEmptyTXKernel(PR_globalKernelArgs)
{
  PR_enterKernel();
  PR_txBegin();
  PR_txCommit();
  PR_txBegin();
  PR_txCommit();
  PR_txBegin();
  PR_txCommit();
}

static PR_ENTRY_POINT void testRandomKernel(PR_globalKernelArgs)
{
  PR_enterKernel();
  int i, j;
  unsigned nbGrans = args.inBuf_size / sizeof(PR_GRANULE_T);
  unsigned idx1 = PR_rand(nbGrans);
  unsigned idx2 = (idx1 + 1) % nbGrans;
  PR_GRANULE_T *addr1;
  PR_GRANULE_T *addr2;
  // size_t size = sizeof(int); // TODO: variable size, floats, ...
  PR_GRANULE_T read1;
  PR_GRANULE_T read2;

  for (i = 0; i < 200; ++i) {
		idx1 = PR_rand(nbGrans);
    PR_txBegin();
    for (j = 0; j < 1; ++j) { // TODO: change PR_MAX_RWSET_SIZE
      idx1 = (idx1 + 1) % nbGrans;
      idx2 = (idx1 + 1) % nbGrans;
      addr1 = &(((PR_GRANULE_T*)args.inBuf)[idx1]);
      addr2 = &(((PR_GRANULE_T*)args.inBuf)[idx2]);
      read1 = PR_read(addr1);
      read2 = PR_read(addr2);
      // printf("old - [%2i] (%2i)%2i - (%2i)%2i (is_abort=%i)\n", PR_THREAD_IDX, idx1, read1, idx2, read2, pr_args.is_abort);
      if (read1 > 0) {
        read1--;
        read2++;
        PR_write(addr1, read1);
        PR_write(addr2, read2);
      }
      // printf("new - [%2i] (%2i)%2i - (%2i)%2i (is_abort=%i)\n", PR_THREAD_IDX, idx1, *addr1, idx2, *addr2, pr_args.is_abort);
    }
    PR_txCommit();
  }
  //   PR_txBegin();
  //   read1 = PR_read(addr1);
  //   read2 = PR_read(addr2);
  //   PR_txCommit();
  //
  // printf("[%i]=%i <---> [%i]=%i\n", idx1, read1, idx2, read2);

  // int i, sum = 0;
  // printf("Sum = %i", sum);

  PR_exitKernel();
}

__global__ void setupKernel(curandState *state) {
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	curand_init(PR_seed, id, 0, &state[id]);
}

__device__ unsigned PR_i_rand(pr_tx_args_dev_host_s args, unsigned n)
{
	curandState *state = ((pr_dev_buff_ext_t*)args.pr_args_ext)->state;
	int id = PR_THREAD_IDX;
	int x = 0;
	/* Copy state to local memory for efficiency */
	curandState localState = state[id];
	x = curand(&localState);
	state[id] = localState;
	return x % n;
}
