#include "hetm-log.h"
#include "setupKernels.cuh"
#include "cmp_kernels.cuh"
#include "bankKernel.cuh"
#include "memman.hpp"
#include "bank.hpp"

using namespace memman;
using namespace knlman;

KnlObj *HeTM_finalTxLog2;
KnlObj *HeTM_bankTx;
KnlObj *HeTM_memcdWriteTx;
KnlObj *HeTM_memcdReadTx;
MemObjOnDev HeTM_bankTxEntryObj;
MemObjOnDev HeTM_bankTxInput;
MemObjOnDev HeTM_memcdTx_input;
MemObjOnDev memcd_global_ts;

static void run_bankTx(knlman_callback_params_s params);
static void run_memcdReadTx(knlman_callback_params_s params);
static void run_memcdWriteTx(knlman_callback_params_s params);
static void run_finalTxLog2(knlman_callback_params_s params);

// static void *bankTx_input[HETM_NB_DEVICES];

int HeTM_setup_bankTx(int nbBlocks, int nbThreads)
{
  PR_global_data_s *d;
  for (int j = 0; j < Config::GetInstance()->NbGPUs(); j++)
  {
    MemObjBuilder b_input;
    MemObjBuilder b_entryObj;
    Config::GetInstance()->SelDev(j);
    PR_curr_dev = j;
    d = &(PR_global[PR_curr_dev]);
    d->PR_blockNum = nbBlocks;
    d->PR_threadNum = nbThreads;
    HeTM_bankTxInput.AddMemObj(new MemObj(b_input
      .SetOptions(0)
      ->SetSize(sizeof(HeTM_bankTx_input_s))
      ->AllocDevPtr()
      ->AllocHostPtr(),
      j
    ));
    HeTM_bankTxEntryObj.AddMemObj(new MemObj(b_entryObj
      .SetOptions(0)
      ->SetSize(sizeof(HeTM_bankTx_s))
      ->AllocDevPtr()
      ->AllocHostPtr(),
      j
    ));
  }
  
  KnlObjBuilder b;
  HeTM_bankTx = new KnlObj(b
    .SetCallback(run_bankTx)
    ->SetEntryObj(&HeTM_bankTxEntryObj));
  return 0;
}

int HeTM_setup_memcdWriteTx(int nbBlocks, int nbThreads)
{
  PR_global_data_s *d;
  for (int j = 0; j < Config::GetInstance()->NbGPUs(); j++)
  {
    MemObjBuilder b;
    Config::GetInstance()->SelDev(j);
    PR_curr_dev = j;
    d = &(PR_global[PR_curr_dev]);
    d->PR_blockNum = nbBlocks;
    d->PR_threadNum = nbThreads;
    HeTM_memcdTx_input.AddMemObj(new MemObj(b
      .SetOptions(0)
      ->SetSize(sizeof(HeTM_memcdTx_input_s))
      ->AllocDevPtr()
      ->AllocHostPtr(),
      j
    ));
  }
  KnlObjBuilder b;
  HeTM_memcdWriteTx = new KnlObj(b
    .SetCallback(run_memcdWriteTx)
    ->SetEntryObj(&HeTM_memcdTx_input));
  return 0;
}

int HeTM_setup_memcdReadTx(int nbBlocks, int nbThreads)
{
  PR_global_data_s *d;
  for (int j = 0; j < HETM_NB_DEVICES; j++) {
    PR_curr_dev = j;
    d = &(PR_global[PR_curr_dev]);
    d->PR_blockNum = nbBlocks;
    d->PR_threadNum = nbThreads;
  }
  KnlObjBuilder b;
  HeTM_memcdReadTx = new KnlObj(b
    .SetCallback(run_memcdReadTx)
    ->SetEntryObj(&HeTM_memcdTx_input));

  // already set-up HeTM_memcdTx_input

  return 0;
}

int HeTM_bankTx_cpy_IO() // TODO: not used
{
  for (int j = 0; j < Config::GetInstance()->NbGPUs(); ++j)
  {
    Config::GetInstance()->SelDev(j);
    PR_curr_dev = j;
    pr_tx_args_s *pr_args = getPrSTMmetaData(j);
    PR_retrieveIO(pr_args);
  }
  return 0;
}

int HeTM_teardown_bankTx()
{
  delete HeTM_bankTx;
  return 0;
}

int HeTM_teardown_memcdWriteTx()
{
  delete HeTM_memcdWriteTx;
  return 0;
}

int HeTM_teardown_memcdReadTx()
{
  delete HeTM_memcdReadTx;
  return 0; 
}

int HeTM_setup_finalTxLog2()
{
  KnlObjBuilder b;
  HeTM_finalTxLog2 = new KnlObj(b
    .SetCallback(run_finalTxLog2));
  return 0;
}

int HeTM_teardown_finalTxLog2()
{
  // TODO: delete entryObj
  delete HeTM_finalTxLog2;
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

static void run_bankTx(knlman_callback_params_s params)
{
  HeTM_bankTx_s *data = (HeTM_bankTx_s*)params.entryObj;
  int j = Config::GetInstance()->SelDev();
  // for (int j = 0; j < HETM_NB_DEVICES; ++j) {
    // account_t *a = data->knlArgs.a; // HOST
    // data->knlArgs.a = parsedData.bank[j].accounts;
    pr_buffer_s inBuf, outBuf;
    HeTM_bankTx_input_s *input, *inputDev;
    // cuda_t *d = &(data->knlArgs.d[j]);

    // memman_select_device(j); // already done
    PR_curr_dev = j;

    // thread_local static unsigned long seed = 0x3F12514A3F12514A;
    MemObj *m_bankTxInput = HeTM_bankTxInput.GetMemObj(j);
    input = (HeTM_bankTx_input_s*)m_bankTxInput->host;

    input->accounts = (int*)HeTM_mempool.GetMemObj(j)->dev; // parsedData.bank[j].accounts;
    input->is_intersection = isInterBatch;
    input->nbAccounts = HeTM_mempool.GetMemObj(j)->size / sizeof(PR_GRANULE_T);
    input->input_buffer = GPUInputBuffer[j];
    input->output_buffer = GPUoutputBuffer[j];

    // printf("dev %i, input_buffer = %p, output_buffer = %p\n", j, input->input_buffer, input->output_buffer);
    m_bankTxInput->CpyHtD(PR_getCurrentStream());

    // CUDA_CHECK_ERROR(cudaDeviceSynchronize(), ""); // sync the previous run

    // cudaFuncSetCacheConfig(bankTx, cudaFuncCachePreferL1);

    inputDev = (HeTM_bankTx_input_s*)m_bankTxInput->dev;
    inBuf.buf = (void*)inputDev;
    inBuf.size = sizeof(HeTM_bankTx_input_s);
    outBuf.buf = NULL;
    outBuf.size = 0;
    pr_tx_args_s *pr_args = getPrSTMmetaData(j);
    PR_prepareIO(pr_args, inBuf, outBuf);

    // TODO: change PR-STM to use knlman
    // PR_blockNum = params.blocks.x;
    // PR_threadNum = params.threads.x;

    // printf("PR_run(bankTx, pr_args) --> dev %i\n", j);
    PR_run(bankTx, pr_args);

    // PR_i_cudaPrepare((pr_args), bankTx);
    // PR_BEFORE_RUN_EXT((pr_args));
    // // PR_i_run(pr_args);
    // bankTx<<<PR_blockNum,PR_threadNum,0,(cudaStream_t)PR_getCurrentStream()>>>(HeTM_pr_args[j].dev);
    // PR_AFTER_RUN_EXT((pr_args));

    // CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "");
  // }
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

  for (int j = 0; j < Config::GetInstance()->NbGPUs(); ++j)
  {
    Config::GetInstance()->SelDev(j);
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

    MemObj *m_memcdTx_input = HeTM_memcdTx_input.GetMemObj(j);
    input = (HeTM_memcdTx_input_s*)m_memcdTx_input->host;
    inputDev = (HeTM_memcdTx_input_s*)m_memcdTx_input->dev;

    input->key   = d->dev_a;
    input->val   = input->key + (d->memcd_array_size/4); // TODO: /sizeof(...)
    input->ts    = input->val + (d->memcd_array_size/4); // TODO: /sizeof(...)
    input->state = input->ts  + (d->memcd_array_size/4); // TODO: /sizeof(...)
    input->nbSets = d->num_sets;
    input->nbWays = d->num_ways;
    input->input_keys = GPUInputBuffer[j];
    input->input_vals = GPUInputBuffer[j];
    input->output     = (memcd_get_output_t*)GPUoutputBuffer;

    MemObj *m_memcd_global_ts = memcd_global_ts.GetMemObj(j);
    input->curr_clock = (int*)m_memcd_global_ts->dev;

    m_memcdTx_input->CpyHtD(PR_getCurrentStream());

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

  for (int j = 0; j < HETM_NB_DEVICES; ++j)
  {
    Config::GetInstance()->SelDev(j);
    cudaFuncSetCacheConfig(memcdWriteTx, cudaFuncCachePreferL1);

    if (a == NULL) {
      // This seems to swap the buffers if given a NULL array...
      accounts = d->dev_a;
      d->dev_a = d->dev_b;
      d->dev_b = accounts;
    }

    MemObj *m_memcdTx_input = HeTM_memcdTx_input.GetMemObj(j);
    input = (HeTM_memcdTx_input_s*)m_memcdTx_input->host;
    inputDev = (HeTM_memcdTx_input_s*)m_memcdTx_input->dev;

    input->key   = d->dev_a;
    input->val   = input->key + (d->memcd_array_size/4)/sizeof(PR_GRANULE_T);
    input->ts    = input->val + (d->memcd_array_size/4)/sizeof(PR_GRANULE_T);
    input->state = input->ts  + (d->memcd_array_size/4)/sizeof(PR_GRANULE_T);
    input->nbSets = d->num_sets;
    input->nbWays = d->num_ways;
    input->input_keys = GPUInputBuffer[j];
    input->input_vals = GPUInputBuffer[j];
    input->output     = (memcd_get_output_t*)GPUoutputBuffer;

    MemObj *m_memcd_global_ts = memcd_global_ts.GetMemObj(j);
    input->curr_clock = (int*)m_memcd_global_ts->dev;
    m_memcdTx_input->CpyHtD(PR_getCurrentStream());

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
