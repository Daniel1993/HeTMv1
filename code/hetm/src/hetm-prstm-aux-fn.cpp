#include "hetm-log.h"
#include "memman.hpp"
#include "pr-stm-wrapper.cuh"

using namespace memman;

MemObjOnDev HeTM_gpuLog;

void hetm_impl_pr_clbk_before_run_ext(pr_tx_args_s *args)
{
	HeTM_GPU_log_s *GPU_log;
	/* Fail on multiple allocs (memman_select needed) */
	
	Config::GetInstance()->SelDev(PR_curr_dev);
	MemObjBuilder b;
	MemObj *m = HeTM_gpuLog.GetMemObj(PR_curr_dev);
	if (!m)
	{
		HeTM_gpuLog.AddMemObj(m = new MemObj(
			b.SetSize(sizeof(HeTM_GPU_log_s))
			->SetOptions(MEMMAN_NONE)
			->AllocDevPtr()
			->AllocHostPtr(), PR_curr_dev
		));
	}
	GPU_log = (HeTM_GPU_log_s*)m->host;
	/* TODO: explicit log only */
	/* ---------------------- */
	MemObj *m_gpu_rset = HeTM_gpu_rset.GetMemObj(PR_curr_dev);
	MemObj *m_gpu_wset = HeTM_gpu_wset.GetMemObj(PR_curr_dev);
	MemObj *m_gpu_rset_cache = HeTM_gpu_rset_cache.GetMemObj(PR_curr_dev);
	MemObj *m_gpu_wset_cache = HeTM_gpu_wset_cache.GetMemObj(PR_curr_dev);
	MemObj *m_mempool = HeTM_mempool.GetMemObj(PR_curr_dev);

	GPU_log->devId                  = PR_curr_dev;
	GPU_log->bmap_rset_devptr       = m_gpu_rset->dev;
	GPU_log->bmap_wset_devptr       = m_gpu_wset->dev;
	GPU_log->bmap_wset_cache_devptr = m_gpu_wset_cache->dev;
	GPU_log->bmap_rset_cache_devptr = m_gpu_rset_cache->dev;
	GPU_log->devMemPoolBasePtr      = m_mempool->dev;
	GPU_log->hostMemPoolBasePtr     = m_mempool->host;
	GPU_log->state                  = (long*)HeTM_shared_data[PR_curr_dev].devCurandState; /* TODO: application specific */
	GPU_log->batchCount             = *hetm_batchCount;
	// printf("Sent to GPU batch %li\n", GPU_log->batchCount);
	GPU_log->isGPUOnly              = (HeTM_gshared_data.isCPUEnabled == 0);
	/* ---------------------- */

	// memman_select("HeTM_mempool_bmap");
	// memman_cpy_to_gpu(HeTM_memStream2[PR_curr_dev], NULL, *hetm_batchCount);
	// memman_bmap_s *bmap = (memman_bmap_s*)memman_get_cpu(NULL);
	// GPU_log->bmap = (memman_bmap_s*)memman_get_gpu(NULL);
	// if ((HeTM_gshared_data.batchCount & 0xff) == 1) {
	// 	CUDA_CHECK_ERROR(cudaMemsetAsync(bmap->dev, 0, bmap->div, (cudaStream_t)HeTM_memStream2[PR_curr_dev]), "");
	// }

	/* ---------------------- */
	args->host.pr_args_ext = (void*)GPU_log;
	// memman_select("HeTM_gpuLog");
	args->dev.pr_args_ext = m->dev;
	// memman_get_gpu(NULL);
	m->CpyHtD(PR_getCurrentStream());
	// memman_cpy_to_gpu(HeTM_memStream2[PR_curr_dev], NULL, *hetm_batchCount);
	// CUDA_CHECK_ERROR(cudaStreamSynchronize((cudaStream_t)HeTM_memStream2[PR_curr_dev]), "");
}

void hetm_impl_pr_clbk_after_run_ext(pr_tx_args_s *args) { }
