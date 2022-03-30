#include "setupKernels.cuh"
#include "cmp_kernels.cuh"
#include "bankKernel.cuh"
#include "bank.hpp"

static void run_memcdReadTx(knlman_callback_params_s params);
static void run_memcdWriteTx(knlman_callback_params_s params);
static void run_finalTxLog2(knlman_callback_params_s params);

int HeTM_setup_memcdWriteTx(int nbBlocks, int nbThreads)
{
  PR_blockNum = nbBlocks;
  PR_threadNum = nbThreads;
  knlman_create("HeTM_memcdWriteTx", run_memcdWriteTx, 0);

  memman_alloc_dual("HeTM_memcdTx_input", sizeof(HeTM_memcdTx_input_s), 0);
  // memman_alloc_gpu("HeTM_memcdTx_output", sizeof(HeTM_memcdTx_input_s), 0);
  return 0;
}

int x(int nbBlocks, int nbThreads)
{
  PR_blockNum = nbBlocks;
  PR_threadNum = nbThreads;
  knlman_create("HeTM_memcdReadTx", run_memcdReadTx, 0);

  // already set-up HeTM_memcdTx_input

  return 0;
}

int HeTM_bankTx_cpy_IO() // TODO: not used
{
  for (int j = 0; HETM_NB_DEVICES; ++j) {
    memman_select_device(j);
    pr_tx_args_s *pr_args = getPrSTMmetaData(j);
    PR_curr_dev = j;
    PR_retrieveIO(pr_args);
  }
  return 0;
}

int HeTM_teardown_bankTx()
{
  knlman_select("HeTM_bankTx");
  knlman_destroy();
  return 0;
}

int HeTM_teardown_memcdWriteTx()
{
  knlman_select("HeTM_memcdWriteTx");
  knlman_destroy();
  return 0;
}

int HeTM_teardown_memcdReadTx()
{
  knlman_select("HeTM_memcdReadTx");
  knlman_destroy();
  return 0;
}

int HeTM_setup_finalTxLog2()
{
  knlman_create("HeTM_finalTxLog2", run_finalTxLog2, 0);
  return 0;
}

int HeTM_teardown_finalTxLog2()
{
  knlman_select("HeTM_finalTxLog2");
  knlman_destroy();
  return 0;
}

static void run_finalTxLog2(knlman_callback_params_s params)
{
  dim3 blocks(params.blocks.x, params.blocks.y, params.blocks.z);
  dim3 threads(params.threads.x, params.threads.y, params.threads.z);
  HeTM_knl_finalTxLog2_s *data = (HeTM_knl_finalTxLog2_s*)params.entryObj;

  /* Kernel Launch */
  HeTM_knl_finalTxLog2 <<< blocks, threads >>> (*data);

  // HeTM_knl_finalTxLog2<<<blocks, threads>>>(data->knlArgs);
}

static void run_memcdReadTx(knlman_callback_params_s params)
{
  HeTM_bankTx_s *data = (HeTM_bankTx_s*)params.entryObj; // TODO
  account_t *a = data->knlArgs.a;
  account_t *accounts = a;
  cuda_t *d = data->knlArgs.d;
  pr_buffer_s inBuf, outBuf;
  HeTM_memcdTx_input_s *input, *inputDev;

  // thread_local static unsigned short seed = 1234;

  for (int j = 0; j < HETM_NB_DEVICES; ++j) {
    memman_select_device(j);
    PR_curr_dev = j;

    CUDA_CHECK_ERROR(cudaDeviceSynchronize(), ""); // sync the previous run

    // memman_ad_hoc_free(NULL); // empties the previous parameters
    cudaFuncSetCacheConfig(memcdReadTx, cudaFuncCachePreferL1);

    if (a == NULL) {
      // This seems to swap the buffers if given a NULL array...
      accounts = d->dev_a;
      d->dev_a = d->dev_b;
      d->dev_b = accounts;
    }

    memman_select("HeTM_memcdTx_input");
    input = (HeTM_memcdTx_input_s*)memman_get_cpu(NULL);
    inputDev = (HeTM_memcdTx_input_s*)memman_get_gpu(NULL);

    input->key      = d->dev_a;
    // TODO: /sizeof(...)
    input->extraKey = input->key + (d->memcd_nbSets*d->memcd_nbWays);
    input->val      = input->extraKey + 3*(d->memcd_nbSets*d->memcd_nbWays);
    input->extraVal = input->val + (d->memcd_nbSets*d->memcd_nbWays);
    input->ts_CPU   = input->extraVal + 7*(d->memcd_nbSets*d->memcd_nbWays);
    input->ts_GPU   = input->ts_CPU + (d->memcd_nbSets*d->memcd_nbWays);
    input->state    = input->ts_GPU + (d->memcd_nbSets*d->memcd_nbWays);
    input->setUsage = input->state + (d->memcd_nbSets*d->memcd_nbWays);
    input->nbSets   = d->num_sets;
    input->nbWays   = d->num_ways;
    input->input_keys = GPUInputBuffer[j];
    input->input_vals = GPUInputBuffer[j];
    input->output     = (memcd_get_output_t*)GPUoutputBuffer[j];

    memman_select("memcd_global_ts");
    input->curr_clock = (int*)memman_get_gpu(NULL);

    memman_select("HeTM_memcdTx_input");
    memman_cpy_to_gpu(HeTM_memStream2[j], NULL, *hetm_batchCount[j]);

    // TODO:
    // inputDev = (HeTM_memcdTx_input_s*)memman_ad_hoc_alloc(NULL, &input, sizeof(HeTM_memcdTx_input_s));
    // memman_ad_hoc_cpy(NULL);

    // TODO: change PR-STM to use knlman
    // PR_blockNum = params.blocks.x;
    // PR_threadNum = params.threads.x;
    inBuf.buf = (void*)inputDev;
    inBuf.size = sizeof(HeTM_memcdTx_input_s);
    outBuf.buf = NULL;
    outBuf.size = 0;
    pr_tx_args_s *pr_args = getPrSTMmetaData(j);
    PR_prepareIO(pr_args, inBuf, outBuf);
    PR_run(memcdReadTx, pr_args);
  }
}

static void run_memcdWriteTx(knlman_callback_params_s params)
{
  HeTM_bankTx_s *data = (HeTM_bankTx_s*)params.entryObj;
  account_t *a = data->knlArgs.a;
  account_t *accounts = a;
  cuda_t *d = data->knlArgs.d;
  pr_buffer_s inBuf, outBuf;
  HeTM_memcdTx_input_s *input, *inputDev;

  for (j = 0; j < HETM_NB_DEVICES; ++j) {
    cudaFuncSetCacheConfig(memcdWriteTx, cudaFuncCachePreferL1);

    if (a == NULL) {
      // This seems to swap the buffers if given a NULL array...
      accounts = d->dev_a;
      d->dev_a = d->dev_b;
      d->dev_b = accounts;
    }

    memman_select("HeTM_memcdTx_input");
    input = (HeTM_memcdTx_input_s*)memman_get_cpu(NULL);
    inputDev = (HeTM_memcdTx_input_s*)memman_get_gpu(NULL);

    input->key   = d->dev_a;
    input->extraKey = input->key + (d->memcd_nbSets*d->memcd_nbWays);
    input->val      = input->extraKey + 3*(d->memcd_nbSets*d->memcd_nbWays);
    input->extraVal = input->val + (d->memcd_nbSets*d->memcd_nbWays);
    input->ts_CPU   = input->extraVal + 7*(d->memcd_nbSets*d->memcd_nbWays);
    input->ts_GPU   = input->ts_CPU + (d->memcd_nbSets*d->memcd_nbWays);
    input->state    = input->ts_GPU + (d->memcd_nbSets*d->memcd_nbWays);
    input->setUsage = input->state + (d->memcd_nbSets*d->memcd_nbWays);
    input->nbSets   = d->num_sets;
    input->nbWays   = d->num_ways;
    input->input_keys = GPUInputBuffer[j];
    input->input_vals = GPUInputBuffer[j];
    input->output     = (memcd_get_output_t*)GPUoutputBuffer[j];

    memman_select("memcd_global_ts");
    input->curr_clock = (int*)memman_get_gpu(NULL);

    memman_select("HeTM_memcdTx_input");
    memman_cpy_to_gpu(HeTM_memStream2[j], NULL, *hetm_batchCount[j]);

    // TODO: change PR-STM to use knlman
    // PR_blockNum = params.blocks.x;
    // PR_threadNum = params.threads.x;
    inBuf.buf = (void*)inputDev;
    inBuf.size = sizeof(HeTM_memcdTx_input_s);
    outBuf.buf = NULL;
    outBuf.size = 0;
    pr_tx_args_s *pr_args = getPrSTMmetaData(j);
    PR_prepareIO(pr_args, inBuf, outBuf);
    PR_run(memcdWriteTx, pr_args);
  }
}
