#include "stdio.h"
#include "stdlib.h"

// TODO: change lib also needs to change this
// #include "gsort_lib.h"
#include "bb_lib.h"

#include "pr-stm.cuh"

#include "pr-stm-internal.cuh"
#include "pr-stm-internal.cu"

#define GRAPH_FILENAME_PATTERN "bank_dev%i_vert%i.in"

int prstm_track_rwset_max_size;
extern __thread pr_args_ext_t *rwset_GPU_log;

__global__ void buildConflMatrix(pr_args_ext_t *GPU_log);

void(*extra_pr_clbk_before_run_ext)(pr_tx_args_s *args);

void impl_pr_clbk_before_run_ext(pr_tx_args_s *args)
{
  if (!rwset_GPU_log) {
    PR_global_data_s *d = &(PR_global[PR_curr_dev]);
    int nbThreads = d->PR_blockNum * d->PR_threadNum;
    long rwset_size = nbThreads * prstm_track_rwset_max_size;
    PR_CHECK_CUDA_ERROR(cudaMallocManaged(&rwset_GPU_log, sizeof(pr_args_ext_t)), "Malloc tracking info");
    PR_CHECK_CUDA_ERROR(cudaMallocManaged(&(rwset_GPU_log->read_addrs), rwset_size), "Malloc tracking info");
    PR_CHECK_CUDA_ERROR(cudaMallocManaged(&(rwset_GPU_log->write_addrs), rwset_size), "Malloc tracking info");
    PR_CHECK_CUDA_ERROR(cudaMallocManaged(&(rwset_GPU_log->confl_mat), nbThreads*nbThreads*sizeof(*(rwset_GPU_log->confl_mat))), "Malloc tracking info");
		rwset_GPU_log->nbThreads = nbThreads;
		rwset_GPU_log->max_rwset_size = prstm_track_rwset_max_size / sizeof(PR_GRANULE_T*);
  }
	args->host.pr_args_ext = (void*)rwset_GPU_log;
	args->dev.pr_args_ext = rwset_GPU_log;
  if (extra_pr_clbk_before_run_ext) extra_pr_clbk_before_run_ext(args);
}

void impl_pr_clbk_after_run_ext(pr_tx_args_s *args)
{
  // TODO clean the datastructs
  int totThreads = rwset_GPU_log->nbThreads * rwset_GPU_log->nbThreads + 128;
  int perBlkThreads = 128;
  int nbBlocks = totThreads / perBlkThreads;

  PR_global_data_s *d = &(PR_global[PR_curr_dev]);
	cudaStream_t stream = d->PR_streams[d->PR_currentStream];

  // CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "");
  printf("buildConflMatrix<<<%i,%i>>>\n", nbBlocks, perBlkThreads);
  buildConflMatrix<<< nbBlocks, perBlkThreads, 0, stream >>>(rwset_GPU_log);

  CUDA_CHECK_ERROR(cudaDeviceSynchronize(), "");
  // printf("\n");

  int nbOfEdges = 0;
  char filename[128];
  sprintf(filename, GRAPH_FILENAME_PATTERN, PR_curr_dev, rwset_GPU_log->nbThreads);
  FILE *fp = fopen(filename, "w");
  for (int i = 0; i < rwset_GPU_log->nbThreads; ++i) {
    // printf("%2i| ", i);
    for (int j = 0; j < rwset_GPU_log->nbThreads; ++j) {
      int idx = i * rwset_GPU_log->nbThreads + j;
      if (rwset_GPU_log->confl_mat[idx]) {
        nbOfEdges++;
      }
      // printf("%i", rwset_GPU_log->confl_mat[idx]);
    }
    // printf("| \n");
  }

  // ------------------------ API
  struct timespec start, end;
  long long diff;

  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  graph G = fromSquareMat(rwset_GPU_log->nbThreads, rwset_GPU_log->confl_mat);
  BB_init_G((BB_parameters_s){
    .n = 15,
    .repOps = 15
  }, G); // TODO: need to break this
  BB_run();
  int *bestRes = BB_getBestSolution();
  while(*bestRes != -1) printf("%i ", *(bestRes++));
  printf("(size=%ld)\n", bestRes - BB_getBestSolution());
  BB_destroy();
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);

  diff = (end.tv_nsec - start.tv_nsec) + (end.tv_sec - start.tv_sec)*1e9;
  printf("matrix creation and FVS solver took %lli ns\n", diff);
  // ----------------------------

  fprintf(fp, "%i %i\n", rwset_GPU_log->nbThreads, nbOfEdges);
  for (int i = 0; i < rwset_GPU_log->nbThreads; ++i) {
    for (int j = 0; j < rwset_GPU_log->nbThreads; ++j) {
      int idx = i * rwset_GPU_log->nbThreads + j;
      if (rwset_GPU_log->confl_mat[idx]) {
        fprintf(fp, "%i %i\n", i, j);
      }
    }
  }
  fprintf(fp, "\n");
}

__global__ void buildConflMatrix(pr_args_ext_t *GPU_log)
{
	// NOTE: 1 thread per entry in the confl mat
	int id = PR_THREAD_IDX;
	int sizeMat = GPU_log->nbThreads;
	int row = id / sizeMat, col = id % sizeMat;

  // if (id == 0) printf("{GPU_log->max_rwset_size} = %i\n", GPU_log->max_rwset_size);

	if (row == col || row >= sizeMat) {
    // printf("thread %i (l=%i, c=%i, sizeMat=%i) returns\n", id, row, col, sizeMat);
    return;
  }

  // printf("[%i] test wSize=%llu(l=%i) rSize=%llu(c=%i)\n", id, (uintptr_t)GPU_log->write_addrs[row * GPU_log->max_rwset_size],
  //   row, (uintptr_t)GPU_log->read_addrs[col * GPU_log->max_rwset_size], col);
	for (int w = 0; w < (uintptr_t)GPU_log->write_addrs[row * GPU_log->max_rwset_size]; ++w) {
  	PR_GRANULE_T *w_addr = GPU_log->write_addrs[row * GPU_log->max_rwset_size + w + 1];
		for (int r = 0; r < (uintptr_t)GPU_log->read_addrs[col * GPU_log->max_rwset_size]; ++r) {
  		PR_GRANULE_T *r_addr = GPU_log->read_addrs[col * GPU_log->max_rwset_size + r + 1];
      // printf("[%i] test (r)%p =?= (w)%p\n", id, r_addr, w_addr);
  	  if (r_addr == w_addr) {
        GPU_log->confl_mat[id] = 1;
        return;
      }
		}
	}
}

