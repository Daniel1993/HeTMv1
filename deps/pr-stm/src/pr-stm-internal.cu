#include "pr-stm.cuh"

#include <curand_kernel.h>
#include <cuda_profiler_api.h>
#include "helper_cuda.h"
#include "helper_timer.h"

struct internal_stats_ {
	int curStream;
	int devId;
};

__thread pr_args_ext_t *rwset_GPU_log = NULL;
void (*pr_clbk_before_run_ext)(pr_tx_args_s*);
void (*pr_clbk_after_run_ext)(pr_tx_args_s*);

PR_HOST void PR_noStatistics(pr_tx_args_s *args)
{
	args->host.nbAborts[PR_curr_dev] = NULL;
	args->host.nbCommits[PR_curr_dev] = NULL;
	args->dev.nbAborts[PR_curr_dev] = NULL;
	args->dev.nbCommits[PR_curr_dev] = NULL;
}

PR_HOST void PR_createStatistics(pr_tx_args_s *args)
{
	PR_global_data_s *d = &(PR_global[PR_curr_dev]);
	int nbThreads = d->PR_blockNum * d->PR_threadNum;
	int sizeArray = nbThreads * sizeof(int);
	int nbDevices, deviceId;

	args->host.nbAborts[PR_curr_dev] = NULL; // not used
	args->host.nbCommits[PR_curr_dev] = NULL;

	PR_CHECK_CUDA_ERROR(cudaGetDeviceCount(&nbDevices), "");
	deviceId = PR_curr_dev % nbDevices;
	PR_CHECK_CUDA_ERROR(cudaSetDevice(deviceId), "");

	PR_ALLOC(args->dev.nbAborts[PR_curr_dev], sizeArray*d->PR_streamCount);
	PR_ALLOC(args->dev.nbCommits[PR_curr_dev], sizeArray*d->PR_streamCount);

	PR_resetStatistics(args);
}

PR_HOST void PR_resetStatistics(pr_tx_args_s *args)
{
	PR_global_data_s *d = &(PR_global[PR_curr_dev]);
	int nbThreads = d->PR_blockNum * d->PR_threadNum;
	int sizeArray = nbThreads * sizeof(int);
	cudaStream_t stream = d->PR_streams[d->PR_currentStream];
	int nbDevices, deviceId;

	PR_CHECK_CUDA_ERROR(cudaGetDeviceCount(&nbDevices), "");
	deviceId = PR_curr_dev % nbDevices;
	PR_CHECK_CUDA_ERROR(cudaSetDevice(deviceId), "");

	// printf(" <<<<<<<<<<<< cudaMemsetAsync on stream %p on dev %i\n", stream, deviceId);
	PR_CHECK_CUDA_ERROR(cudaMemsetAsync(args->dev.nbAborts[PR_curr_dev], 0, sizeArray*d->PR_streamCount, stream), "");
	PR_CHECK_CUDA_ERROR(cudaMemsetAsync(args->dev.nbCommits[PR_curr_dev], 0, sizeArray*d->PR_streamCount, stream), "");
	d->PR_nbCommits = 0;
	d->PR_nbAborts = 0;
}

PR_HOST void PR_prepareIO(
	pr_tx_args_s *args,
	pr_buffer_s inBuf,
	pr_buffer_s outBuf
) {
	// input
	args->dev.inBuf = inBuf.buf;
	args->dev.inBuf_size = inBuf.size;
	args->host.inBuf = NULL; // Not needed
	args->host.inBuf_size = 0;

	// output
	args->dev.outBuf = outBuf.buf;
	args->dev.outBuf_size = outBuf.size;
	args->host.outBuf = NULL; // Not needed
	args->host.outBuf_size = 0;
}

PR_HOST void PR_i_cudaPrepare(
	pr_tx_args_s *args,
	void(*callback)(pr_tx_args_dev_host_s)
) {
	PR_global_data_s *d = &(PR_global[PR_curr_dev]);
	args->host.mtx = d->PR_lockTableHost;
	args->host.current_stream = d->PR_currentStream;

	// dev is a CPU-local struct that is passed to the kernel
	args->dev.devId = PR_curr_dev;
	args->dev.mtx  = d->PR_lockTableDev;
	args->dev.current_stream = d->PR_currentStream;
	args->callback = callback;
}

PR_HOST void PR_retrieveIO(pr_tx_args_s *args)
{
	PR_global_data_s *d = &(PR_global[PR_curr_dev]);
	int i, nbThreads = d->PR_blockNum * d->PR_threadNum;
	int sizeArray = nbThreads * sizeof(int);
	// int sizeMtx = PR_LOCK_TABLE_SIZE * sizeof(int);
	static thread_local int *hostNbAborts;
	static thread_local int *hostNbCommits;
	static thread_local int last_size;
	cudaStream_t stream = d->PR_streams[d->PR_currentStream];
	int nbDevices, deviceId;

	PR_CHECK_CUDA_ERROR(cudaGetDeviceCount(&nbDevices), "");
	deviceId = PR_curr_dev % nbDevices;
	PR_CHECK_CUDA_ERROR(cudaSetDevice(deviceId), "");

	if (hostNbAborts == NULL) {
		cudaMallocHost(&hostNbAborts, sizeArray*d->PR_streamCount);
		cudaMallocHost(&hostNbCommits, sizeArray*d->PR_streamCount);
		last_size = sizeArray;
	}

	if (last_size != sizeArray) {
		// TODO
		hostNbAborts = (int*)realloc(hostNbAborts, sizeArray*d->PR_streamCount);
		hostNbCommits = (int*)realloc(hostNbCommits, sizeArray*d->PR_streamCount);
		last_size = sizeArray;
	}

	// TODO: this is only done in the end --> need to copy all streams!

	void *devNbAborts = (int*)args->dev.nbAborts;
	void *devNbCommits = (int*)args->dev.nbCommits;

	// PR_CPY_TO_HOST(PR_lockTableHost, PR_lockTableDev, sizeMtx); // TODO: only for debug
	PR_CPY_TO_HOST_ASYNC(hostNbAborts, devNbAborts, sizeArray*d->PR_streamCount, stream);
	PR_CPY_TO_HOST_ASYNC(hostNbCommits, devNbCommits, sizeArray*d->PR_streamCount, stream);
	d->PR_nbAborts = 0;
	d->PR_nbCommits = 0;
	CUDA_CHECK_ERROR(cudaStreamSynchronize(PR_getCurrentStream()), "");
	for (i = 0; i < nbThreads*d->PR_streamCount; ++i) {
		d->PR_nbAborts += hostNbAborts[i];
		d->PR_nbCommits += hostNbCommits[i];
	}
	// printf("PR_nbAborts = %lli\n", PR_nbAborts);
}

PR_HOST cudaStream_t PR_getCurrentStream()
{
	PR_global_data_s *d = &(PR_global[PR_curr_dev]);
	return d->PR_streams[d->PR_currentStream];
}

PR_HOST void PR_disposeIO(pr_tx_args_s *args)
{
	PR_CHECK_CUDA_ERROR(cudaFree(args->dev.nbAborts), "");
	PR_CHECK_CUDA_ERROR(cudaFree(args->dev.nbCommits), "");
}

PR_HOST void
PR_i_afterStream(cudaStream_t stream, cudaError_t status, void *data)
{
	struct internal_stats_ *is = (struct internal_stats_ *)data;
	int devId = is->devId;
	int curStream = is->curStream;
	PR_global_data_s *d = &(PR_global[devId]);
	__atomic_store_n(&d->PR_isDone[curStream], 1, __ATOMIC_RELEASE);
	__atomic_store_n(&d->PR_isStart[curStream], 0, __ATOMIC_RELEASE);

	// printf(
	// 	"[dev%i,curStrm=%i] PR_i_afterStream: \nd->PR_isDone=%i d->PR_isStart=%i PR_sumNbCommits=%li\n",
	// 	devId, curStream, d->PR_isDone[curStream], d->PR_isStart[curStream], *d->PR_sumNbCommits
	// );
	free(data);
}

PR_HOST void
PR_i_afterStats(cudaStream_t stream, cudaError_t status, void *data)
{
	struct internal_stats_ *is = (struct internal_stats_ *)data;
	int devId = is->devId;
	int curStream = is->curStream;
	PR_global_data_s *d = &(PR_global[devId]);

	d->PR_nbCommitsLastKernel = *d->PR_sumNbCommits - d->PR_nbCommitsStrm[curStream];
	d->PR_nbAbortsLastKernel  = *d->PR_sumNbAborts  - d->PR_nbAbortsStrm[curStream];

	d->PR_nbCommitsSinceCheckpoint += d->PR_nbCommitsLastKernel;
	d->PR_nbAbortsSinceCheckpoint  += d->PR_nbAbortsLastKernel;

	// printf(
	// 	"[dev%i,curStrm=%lu] PR_i_afterStats: \nPR_nbCommitsSinceCheckpoint=%li PR_nbCommitsLastKernel=%li PR_sumNbCommits=%li\n",
	// 	devId, curStream, d->PR_nbCommitsSinceCheckpoint, d->PR_nbCommitsLastKernel, *d->PR_sumNbCommits
	// );

	d->PR_nbCommitsStrm[curStream] = *d->PR_sumNbCommits;
	d->PR_nbAbortsStrm[curStream]  = *d->PR_sumNbAborts;
	__atomic_thread_fence(__ATOMIC_RELEASE);
	free(data);
}

PR_HOST void
PR_i_run(pr_tx_args_s *args)
{
	PR_global_data_s *d = &(PR_global[PR_curr_dev]);
	// StopWatchInterface *kernelTime = NULL;
	int curStream = d->PR_currentStream;
	cudaStream_t stream = d->PR_streams[curStream];

	// cudaFuncSetCacheConfig(args->callback, cudaFuncCachePreferL1);

	__atomic_store_n(&d->PR_isDone[curStream], 0, __ATOMIC_RELEASE);
	__atomic_store_n(&d->PR_isStart[curStream], 1, __ATOMIC_RELEASE);

	struct internal_stats_ *is = (struct internal_stats_ *)malloc(sizeof(struct internal_stats_));
	
	is->curStream = (int)curStream;
	is->devId = (int)PR_curr_dev;

	// pass struct by value
	PR_CHECK_CUDA_ERROR(cudaEventRecord(d->PR_eventKernelStart, stream), ""); // TODO: does not work in multi-thread
	// if (stream != NULL) {
		// int dev;
		// cudaGetDevice(&dev);
		// printf("Run kernel in device %i (cudaGetDevice = %i)\n", PR_curr_dev, dev);
		args->callback<<< d->PR_blockNum, d->PR_threadNum, 0, stream >>>(args->dev);
	// } else {
	// 	// serializes
	// 	args->callback<<< PR_blockNum, PR_threadNum >>>(args->dev);
	// }
	// printf("         [dev%i,curStrm=%i] PR_i_run\n", PR_curr_dev, curStream );
	PR_CHECK_CUDA_ERROR(cudaEventRecord(d->PR_eventKernelStop, stream), "");
	PR_CHECK_CUDA_ERROR(cudaStreamAddCallback(
		stream, PR_i_afterStream, (void*)is, 0
	), "");
	
	// cudaThreadSynchronize();
}

// Extension wrapper
PR_HOST void PR_init(PR_init_params_s params)
{
	int nbStreams = params.nbStreams;
	int nbMtxs = params.lockTableSize;

	PR_global_data_s *d = &(PR_global[PR_curr_dev]);
	const size_t sizeMtx = nbMtxs * sizeof(int);

	// user must init for each device
	int nbDevices, deviceId;
	PR_CHECK_CUDA_ERROR(cudaGetDeviceCount(&nbDevices), "");
	deviceId = PR_curr_dev % nbDevices;
	PR_CHECK_CUDA_ERROR(cudaSetDevice(deviceId), "");

	d->PR_streamCount = nbStreams;
	if (d->PR_streams == NULL) {
		d->PR_streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * nbStreams);
		for (int i = 0; i < nbStreams; ++i) {
			PR_CHECK_CUDA_ERROR(cudaStreamCreate(d->PR_streams + i), "stream");
			// printf(" <<<<<<<<<<<< created stream %p on dev %i\n", d->PR_streams[i], deviceId);
		}
	}

	if (d->PR_lockTableDev == NULL) {
		PR_CHECK_CUDA_ERROR(cudaEventCreate(&d->PR_eventKernelStart), "");
		PR_CHECK_CUDA_ERROR(cudaEventCreate(&d->PR_eventKernelStop), "");
		d->PR_lockTableHost = NULL; // TODO: host locks not needed
		// memset(PR_lockTableHost, 0, PR_LOCK_TABLE_SIZE * sizeof(int));
		PR_ALLOC(d->PR_lockTableDev, sizeMtx);
		PR_CHECK_CUDA_ERROR(cudaMemset(d->PR_lockTableDev, 0, sizeMtx), "memset");
	}

	if (d->PR_sumNbCommitsDev == NULL) {
		size_t sizeCount = sizeof(uint64_t)*2;
		CUDA_DEV_ALLOC (d->PR_sumNbCommitsDev, sizeCount);
		CUDA_HOST_ALLOC(d->PR_sumNbCommits,    sizeCount);
		d->PR_sumNbAbortsDev = d->PR_sumNbCommitsDev + 1; // Aborts are after commits in memory layout
		d->PR_sumNbAborts    = d->PR_sumNbCommits    + 1;

		PR_CHECK_CUDA_ERROR(cudaMemset(d->PR_sumNbCommitsDev, 0, sizeCount), "");

		*d->PR_sumNbCommits = 0;
		*d->PR_sumNbAborts  = 0;
	}
}

PR_HOST void PR_run(void(*callback)(pr_tx_args_dev_host_s), pr_tx_args_s *pr_args)
{
	int nbDevices, deviceId;

	PR_CHECK_CUDA_ERROR(cudaGetDeviceCount(&nbDevices), "");
	deviceId = PR_curr_dev % nbDevices;
	PR_CHECK_CUDA_ERROR(cudaSetDevice(deviceId), "");

	PR_i_cudaPrepare(pr_args, callback);

  	if (pr_clbk_before_run_ext)
		pr_clbk_before_run_ext(pr_args);
	PR_i_run(pr_args);
  	if (pr_clbk_after_run_ext)
	  	pr_clbk_after_run_ext(pr_args);
}

PR_HOST void PR_teardown()
{
	PR_global_data_s *d = &(PR_global[PR_curr_dev]);
	// cudaFreeHost((void*)PR_lockTableHost);
	int nbDevices, deviceId;

	PR_CHECK_CUDA_ERROR(cudaGetDeviceCount(&nbDevices), "");
	deviceId = PR_curr_dev % nbDevices;
	PR_CHECK_CUDA_ERROR(cudaSetDevice(deviceId), "");

	cudaFree((void*)d->PR_lockTableDev);
	for (int i = 0; i < d->PR_streamCount; ++i) {
		PR_CHECK_CUDA_ERROR(cudaStreamDestroy(d->PR_streams[i]), "stream");
	}
	free(d->PR_streams);
}

PR_HOST void PR_useNextStream(pr_tx_args_s *args)
{
	PR_global_data_s *d = &(PR_global[PR_curr_dev]);
	static size_t countSize = sizeof(uint64_t)*2;
	uintptr_t curStream = d->PR_currentStream;
	int nbDevices, deviceId;

	PR_CHECK_CUDA_ERROR(cudaGetDeviceCount(&nbDevices), "");
	deviceId = PR_curr_dev % nbDevices;
	PR_CHECK_CUDA_ERROR(cudaSetDevice(deviceId), "");

	PR_CHECK_CUDA_ERROR(cudaGetLastError(), "before PR_reduceCommitAborts");

	PR_reduceCommitAborts<<<d->PR_blockNum, d->PR_threadNum, 0, d->PR_streams[d->PR_currentStream]>>>
		(0, curStream, args->dev, (uint64_t*)d->PR_sumNbCommitsDev, (uint64_t*)d->PR_sumNbAbortsDev);

	PR_CHECK_CUDA_ERROR(cudaGetLastError(), "PR_reduceCommitAborts");

	struct internal_stats_ *is = (struct internal_stats_ *)malloc(sizeof(struct internal_stats_));
	
	is->curStream = (int)curStream;
	is->devId = (int)PR_curr_dev;

	// not first time, copy previous data
	CUDA_CPY_TO_HOST_ASYNC(
		d->PR_sumNbCommits,
		d->PR_sumNbCommitsDev,
		countSize, // also copies aborts
		d->PR_streams[curStream]
	);
	CUDA_CHECK_ERROR(cudaStreamAddCallback(
		d->PR_streams[curStream], PR_i_afterStats, (void*)is, 0
	), "");

	PR_CHECK_CUDA_ERROR(cudaMemsetAsync(d->PR_sumNbCommitsDev, 0, countSize, d->PR_streams[d->PR_currentStream]), "");

	// cudaDeviceSynchronize();
	// printf("[dev%i] commits=%lu aborts=%lu\n", PR_curr_dev,
	// 	*(uint64_t*)d->PR_sumNbCommits, *(uint64_t*)d->PR_sumNbAborts);

	d->PR_currentStream = (curStream + 1) % d->PR_streamCount;
}

PR_HOST void PR_checkpointAbortsCommits()
{
	PR_global_data_s *d = &(PR_global[PR_curr_dev]);
	d->PR_nbAbortsSinceCheckpoint = 0;
	d->PR_nbCommitsSinceCheckpoint = 0;
}
