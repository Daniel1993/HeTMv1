#ifndef SHARED_GUARD_H_
#define SHARED_GUARD_H_

#include <pthread.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "hetm.cuh"

#ifndef HETM_GPU_EN
// Set to 0 to ignore GPU
#define HETM_GPU_EN 1
#endif

#ifndef HETM_CPU_EN
// Set to 0 to ignore CPU
#define HETM_CPU_EN 1
#endif

void bindThread(long threadId);
void sigCatcher(int sig);

#endif /* SHARED_GUARD_H_ */
