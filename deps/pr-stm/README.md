# PR-STM

PR-STM: Priority Rule-based STM for GPU, please cite [1]. This repo is an attempt of providing a clean, modular and easy to use version of this TM library.

## Compiling

This project has the following dependencies:

cuda-utils

You can set the location of cuda-utils compiling with:

`$ make CUDA_UTIL_DIR=/path/to/cuda-utils`

## Internals

Important structs:

`pr_buffer_s`: use it to tell PR-STM transactions where to place the input/out.

`pr_tx_args_dev_host_s`: metadata in the CPU side, as in `PR_args_s`, there is a pointer `pr_args_ext` for users.

`PR_rwset_s`: R/W-set of the a PR-STM transaction: `addrs` is the accessed memory addresses, `values` is the new values (lazy apply), `versions` is used for conflict detection in the R-set, `size` is the current size of the R/W-set.

`PR_args_s`: contains metadata for PR-STM, the `(void*)pr_args_ext` is a pointer that can be user-defined, otherwise is `NULL`.

Constants:

`PR_LOCK_TABLE_SIZE`: size of the lock table, has an impact on false sharing.

`PR_GRANULE_T`: granule type. Provides the granularity at which PR-STM detects conflicts (TODO: multi-granularity).

`PR_LOCK_GRANULARITY`: hard-coded as `sizeof(PR_GRANULE_T)`

`PR_LOCK_GRAN_BITS`: hard-coded as `logBase2(PR_LOCK_GRANULARITY)`

`PR_MAX_RWSET_SIZE`: maximum size of the R/W-Set (TODO: create an extendable version)

## Extending

For performance reasons the library is header-only. The end-user must implement it by including `pr-stm-internal.cuh`. Example:
```C++
#include "pr-stm.cuh"

// change the behaviour (if needed) here. Example:
#ifndef   PR_AFTER_WRITEBACK_EXT
#define   PR_AFTER_WRITEBACK_EXT(args, i, addr, val) /* new behaviour here */
#endif /* PR_AFTER_WRITEBACK_EXT */

#include "pr-stm-internal.cuh"
```
Then link this file with the rest of your project.

MACROs that can be re-defined (TODO: not updated for a while, some MACROs no longer in use):

`PR_ARGS_S_EXT`: adds new fields in the struct that is passed around PR-STM GPU kernels. Initialize the struct in `PR_BEFORE_RUN_EXT`. Within the kernel, it is located in `args->pr_args_ext`.

`PR_DEV_BUFF_S_EXT`: adds new fields in the struct held in the CPU side.

`PR_BEFORE_BEGIN_EXT`: executed before starting the transaction (on abort this is re-executed).

`PR_AFTER_COMMIT_EXT`: executed after committing successfully the transaction (is not re-executed).

`PR_BEFORE_KERNEL_EXT`: executed on entering the PR-STM enabled kernel.

`PR_AFTER_KERNEL_EXT`: executed on exiting the PR-STM enabled kernel.

`PR_AFTER_VAL_LOCKS_EXT`: executed after all locks have been acquired (you can access the R/W-sets through `args->rset`, `args->wset`).

`PR_AFTER_WRITEBACK_EXT`: executed after each write is written-back to memory.

`PR_BEFORE_RUN_EXT`: executed before launching the kernel.

`PR_AFTER_RUN_EXT`:  executed after launching the kernel (note: if no stream is given, PR-STM serializes, i.e., blocks while the kernel executes in the GPU). When using streams it is useful to use `cudaStreamAddCallback` to execute some code after the stream finishes.

## PR-STM kernel example

A kernel using PR-STM looks like the following:
```C++
__global__ void someKernel(PR_globalKernelArgs)
{
  PR_enterKernel();
  ...
  
  PR_txBegin(); // WARNING: this is a loop! Do not use breaks to exit outer loops
  // Inside the transaction
  A = PR_read(address);
  // do stuff with A, e.g., B = A+2
  PR_write(address, B);
  PR_txCommit();
  
  ...
  PR_exitKernel();
}
```
If there is an abort, it can be early detected with (it is detected on `PR_txCommit` either way):
```C++
PR_txBegin();
...
if (pr_args.is_abort) break; // exits the PR_txBegin() loop
...
PR_txCommit();
```

## Calling a PR-STM kernel

An example is given below:
```C++
pr_tx_args_s prArgs;
PR_init();
PR_createStatistics(&prArgs); // Sets PR_nbAborts, PR_nbCommits and PR_kernelTime

// init inBuf, outBuf
PR_CPY_TO_DEV(inBuf.buf, hostInBuffer, inBuf.size);
PR_CPY_TO_DEV(outBuf.buf, hostOutBuffer, outBuf.size);
PR_prepareIO(&prArgs, inBuf, outBuf);
PR_run(someKernel, &prArgs, NULL); // NULL means default stream
PR_retrieveIO(&prArgs); // updates PR_nbAborts and PR_nbCommits
PR_CPY_TO_HOST(hostInBuffer, inBuf.buf, inBuf.size);
PR_CPY_TO_HOST(hostOutBuffer, outBuf.buf, outBuf.size);

PR_disposeIO(&prArgs);
PR_teardown();
```

## Running

Then run with (first parameter is number of blocks, second is threads):
`$ ./bank 40 256`

Statistic information is appended in the file *stats.txt*

__________
[1] Shen Q., Sharp C., Blewitt W., Ushaw G., Morgan G. (2015) PR-STM: Priority Rule Based Software Transactions for the GPU. In: Traff J., Hunold S., Versaci F. (eds) Euro-Par 2015: Parallel Processing. Euro-Par 2015. Lecture Notes in Computer Science, vol 9233. Springer, Berlin, Heidelberg
