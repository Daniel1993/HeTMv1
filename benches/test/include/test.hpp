#ifndef TEST_HPP_GUARD
#define TEST_HPP_GUARD

typedef struct kernel_input_ {
  int roundId;
  int *mempool;
} kernel_input_s;

void LaunchTestKernel(void *argsPtr);

#endif /* TEST_HPP_GUARD */
