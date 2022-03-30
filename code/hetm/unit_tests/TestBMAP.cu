#include "TestBMAP.cuh"
#include "pr-stm.cuh"
#include "hetm.cuh"
#include "hetm-aux.cuh"
#include "hetm-log.h"
#include "stm.h"
#include "stm-wrapper.h"
#include "pr-stm-wrapper.cuh"
#include "memman.hpp"
#include "knlman.hpp"

CPPUNIT_TEST_SUITE_REGISTRATION(TestBMAP);

using namespace std;

typedef PR_GRANULE_T granule_t;
static const int NB_GPUS              = 2; // TODO: compile with HETM_NB_DEVICES=2
static const size_t GRAN_BITMAP_CACHE = 1<<16;
static const size_t MEMPOOL_SIZE      = 1<<18;
static const char *STR_MEMPOOL        = "HeTM_mempool";
static const char *STR_BMAP_GPU       = "HeTM_mempool_bmap";
static const char *STR_CPU_WS         = "HeTM_cpu_wset";
static const char *STR_CPU_WS_CACHE   = "HeTM_cpu_wset_cache";

static granule_t *mempool_base_addr_cpu[NB_GPUS];
static granule_t *mempool_base_addr_gpu[NB_GPUS];
static granule_t *cpu_buffer_of_gpu_mempool[NB_GPUS];
// static memman_bmap_s *bmap_struct[NB_GPUS];
// static memman_bmap_s *bmap_struct_gpu[NB_GPUS];
static void *bmap_cpu[NB_GPUS];
static void *bmap_gpu[NB_GPUS];

// static memman_s bmap_buffer;

void SetWrtPositionInCPU(size_t nbPos, granule_t **addrs);
void SetRdPositionInCPU(size_t nbPos, granule_t **addrs);

static void run_kernelGPUsetPos(knlman_callback_params_s params);

typedef struct knlEntryObj_ {
    int devId;
    size_t nbPos;
    granule_t *base;
    granule_t **rd_addrs;
    granule_t **wt_addrs;
} knlEntryObj_s;

// static knlman_s kernelObj;
// static knlEntryObj_s *argsDev0, *argsDev1;
const size_t size_rwset = 1;
static granule_t **rd_addr_cpu ;
static granule_t **rd_addr_gpu0;
static granule_t **rd_addr_gpu1;
static granule_t **wt_addr_cpu ;
static granule_t **wt_addr_gpu0;
static granule_t **wt_addr_gpu1;

TestBMAP::TestBMAP()  { }
TestBMAP::~TestBMAP() { }

void TestBMAP::setUp()
{
    printf("\n\nSetUp\n\n");
    HeTM_init((HeTM_init_s){
		.policy       = HETM_GPU_INV,
		.nbCPUThreads = 1,
		.nbGPUBlocks  = 1,
		.nbGPUThreads = 1,
		.isCPUEnabled = 1,
		.isGPUEnabled = 1,
		.mempool_size = MEMPOOL_SIZE*sizeof(granule_t),
        .mempool_opts = MEMMAN_NONE // MEMMAN_UNIF
	});
    for (int i = 0; i < NB_GPUS; ++i)
    {
        // memman_select_device(i);
        // memman_select(STR_MEMPOOL);
        // mempool_base_addr_cpu[i] = (granule_t*)memman_get_cpu(NULL);
        // mempool_base_addr_gpu[i] = (granule_t*)memman_get_gpu(NULL);

        // initialized in hetm-threading.cu (HeTM_thread_data is thread local)
        HeTM_thread_data[i] = &(HeTM_shared_data[i].threadsInfo[0]); // only 1 thread
        HeTM_thread_data[i]->devId = i;
    }
    // knlmanObjCreate(&kernelObj, "kernelGPUsetPos", run_kernelGPUsetPos, 0);

    cudaMemset(mempool_base_addr_cpu[0], 0, MEMPOOL_SIZE*sizeof(granule_t));
    // memman_select_device(0);
    cudaMemset(mempool_base_addr_gpu[0], 0, MEMPOOL_SIZE*sizeof(granule_t));
    // memman_select_device(1);
    cudaMemset(mempool_base_addr_gpu[1], 0, MEMPOOL_SIZE*sizeof(granule_t));

    // cudaMallocManaged(&argsDev0, sizeof(knlEntryObj_s));
    // cudaMallocManaged(&argsDev1, sizeof(knlEntryObj_s));
    cudaMallocManaged(&cpu_buffer_of_gpu_mempool[0], MEMPOOL_SIZE*sizeof(granule_t*));
    cudaMallocManaged(&cpu_buffer_of_gpu_mempool[1], MEMPOOL_SIZE*sizeof(granule_t*));
    cudaMallocManaged(&rd_addr_cpu, size_rwset*sizeof(granule_t*));
    cudaMallocManaged(&rd_addr_gpu0, size_rwset*sizeof(granule_t*));
    cudaMallocManaged(&rd_addr_gpu1, size_rwset*sizeof(granule_t*));
    cudaMallocManaged(&wt_addr_cpu, size_rwset*sizeof(granule_t*));
    cudaMallocManaged(&wt_addr_gpu0, size_rwset*sizeof(granule_t*));
    cudaMallocManaged(&wt_addr_gpu1, size_rwset*sizeof(granule_t*));
}

void TestBMAP::tearDown()
{
    printf("\n\nTearDown\n\n");
    // cudaFree(argsDev0);
    // cudaFree(argsDev1);
    cudaFree(cpu_buffer_of_gpu_mempool[0]);
    cudaFree(cpu_buffer_of_gpu_mempool[1]);
    cudaFree(rd_addr_cpu);
    cudaFree(rd_addr_gpu0);
    cudaFree(rd_addr_gpu1);
    cudaFree(wt_addr_cpu);
    cudaFree(wt_addr_gpu0);
    cudaFree(wt_addr_gpu1);
    // knlmanObjDestroy(&kernelObj);
    HeTM_destroy();
}

// static void launchKernel(knlman_s *kernelObj, int devId, int thrs, int blks, void *args)
// {
//     kernelObj->select(kernelObj);
//     kernelObj->setNbBlocks(blks, 1, 1);
//     kernelObj->setThrsPerBlock(thrs, 1, 1);
//     kernelObj->setDevice(devId);
//     kernelObj->setStream(NULL); // Default stream
//     kernelObj->setArgs(args);
//     kernelObj->run();
// }

void TestBMAP::TestRWDifferentPositions()
{
    granule_t *_cpu = mempool_base_addr_cpu[0];
    granule_t *_gpu0 = mempool_base_addr_gpu[0];
    granule_t *_gpu1 = mempool_base_addr_gpu[1];

    CPPUNIT_ASSERT(mempool_base_addr_cpu[0] == mempool_base_addr_cpu[1]); // CPU keeps only 1 memory pool

    PR_global[0].PR_blockNum  = 1;
    PR_global[0].PR_threadNum = 1;
    PR_global[1].PR_blockNum  = 1;
    PR_global[1].PR_threadNum = 1;

    // argsDev0->rd_addrs = rd_addr_gpu0;
    // argsDev0->wt_addrs = wt_addr_gpu0;
    // argsDev1->rd_addrs = rd_addr_gpu1;
    // argsDev1->wt_addrs = wt_addr_gpu1;

    rd_addr_cpu[0]  = _cpu+0;
    rd_addr_gpu0[0] = _gpu0+1;
    rd_addr_gpu1[0] = _gpu1+2;

    wt_addr_cpu[0]  = _cpu+0;
    wt_addr_gpu0[0] = _gpu0+1;
    wt_addr_gpu1[0] = _gpu1+2;

    // cudaMemcpy(argsDev0->rd_addrs, rd_addr_gpu0, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);
    // cudaMemcpy(argsDev0->wt_addrs, wt_addr_gpu0, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);
    // cudaMemcpy(argsDev1->rd_addrs, rd_addr_gpu1, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);
    // cudaMemcpy(argsDev1->wt_addrs, rd_addr_gpu1, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);

    HeTM_gshared_data.batchCount = 13;

    PR_curr_dev = 0;
    // argsDev0->nbPos = size_rwset;
    // argsDev0->base = _gpu0;
    // argsDev0->devId = 0;
    // launchKernel(&kernelObj, /*devId*/0, /*thrs*/1, /*blks*/1, argsDev0);

    PR_curr_dev = 1;
    // argsDev1->nbPos = size_rwset;
    // argsDev1->base = _gpu1;
    // argsDev1->devId = 1;
    // launchKernel(&kernelObj, /*devId*/1, /*thrs*/1, /*blks*/1, argsDev1);

    SetRdPositionInCPU(size_rwset, rd_addr_cpu);
    SetWrtPositionInCPU(size_rwset, wt_addr_cpu);

    // memman_select_device(0); // also checks if the device exists
    cudaDeviceSynchronize();

    // memman_select_device(1);
    cudaDeviceSynchronize(); // when emulating the second device this is redundant

    // do the comparison among GPUs
    notifyBatchIsDone(); // GPU controller

    // inspect write set of GPU0
    // memman_select_device(0);
    // memman_select("HeTM_gpu_wset");
    // memman_cpy_to_cpu(NULL, NULL, 0);
    // unsigned char *gpu0_wrtset = (unsigned char *)memman_get_cpu(NULL);
    cudaDeviceSynchronize();
    // printf("[GPU0] wrote in pos %i = %i\n", 1, gpu0_wrtset[1]);
    // printf("[GPU0] wrote in pos %i = %i\n", 2, gpu0_wrtset[2]);

    // inspect write set of GPU1
    // memman_select_device(1);
    // memman_select("HeTM_gpu_wset");
    // memman_cpy_to_cpu(NULL, NULL, 0);
    // unsigned char *gpu1_wrtset = (unsigned char *)memman_get_cpu(NULL);
    cudaDeviceSynchronize();
    // printf("[GPU1] wrote in pos %i = %i\n", 1, gpu1_wrtset[1]);
    // printf("[GPU1] wrote in pos %i = %i\n", 2, gpu1_wrtset[2]);

    // for debugging
    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[0], mempool_base_addr_gpu[0], sizeof(granule_t)*MEMPOOL_SIZE);
    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[1], mempool_base_addr_gpu[1], sizeof(granule_t)*MEMPOOL_SIZE);

    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][1] == 1 && "GPU0 id not in GPU0 mempool");
    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][2] == 2 && "GPU1 id not in GPU1 mempool");

    // do the comparison with the CPU
    pollIsRoundComplete(/*nonBlock*/1); // TODO: this should not hang!

    // for debugging
    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[0], mempool_base_addr_gpu[0], sizeof(granule_t)*MEMPOOL_SIZE);
    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[1], mempool_base_addr_gpu[1], sizeof(granule_t)*MEMPOOL_SIZE);

    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][1] == 1 && "GPU0 id not in GPU0 mempool");
    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][2] == 2 && "GPU1 id not in GPU1 mempool");

    syncGPUdataset((void*)1); // contains mergeMatricesAndRunFVS();, 1 to set nonBlock

    // copies the merged dataset
    // memman_select_device(0);
    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[0], mempool_base_addr_gpu[0], sizeof(granule_t)*MEMPOOL_SIZE);
    cudaDeviceSynchronize();
    // memman_select_device(1);
    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[1], mempool_base_addr_gpu[1], sizeof(granule_t)*MEMPOOL_SIZE);
    cudaDeviceSynchronize();


    CPPUNIT_ASSERT(mempool_base_addr_cpu[0][0] == 3 && "CPU id not in mempool");
    CPPUNIT_ASSERT(mempool_base_addr_cpu[0][1] == 1 && "GPU0 id not in mempool");
    CPPUNIT_ASSERT(mempool_base_addr_cpu[0][2] == 2 && "GPU1 id not in mempool");

    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][0] == 3 && "CPU id not in GPU0 mempool");
    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][1] == 1 && "GPU0 id not in GPU0 mempool");
    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][2] == 2 && "GPU1 id not in GPU0 mempool");

    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][0] == 3 && "CPU id not in GPU1 mempool");
    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][1] == 1 && "GPU0 id not in GPU1 mempool");
    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][2] == 2 && "GPU1 id not in GPU1 mempool");

    // TODO: apply the dataset in DRAM
    printf("CPU is done with the merging\n");
}

void TestBMAP::TestWrtDifferentReadSamePositions()
{
    granule_t *_cpu = mempool_base_addr_cpu[0];
    granule_t *_gpu0 = mempool_base_addr_gpu[0];
    granule_t *_gpu1 = mempool_base_addr_gpu[1];

    CPPUNIT_ASSERT(mempool_base_addr_cpu[0] == mempool_base_addr_cpu[1]); // CPU keeps only 1 memory pool


    PR_global[0].PR_blockNum  = 1;
    PR_global[0].PR_threadNum = 1;
    PR_global[1].PR_blockNum  = 1;
    PR_global[1].PR_threadNum = 1;

    // argsDev0->rd_addrs = rd_addr_gpu0;
    // argsDev0->wt_addrs = wt_addr_gpu0;
    // argsDev1->rd_addrs = rd_addr_gpu1;
    // argsDev1->wt_addrs = wt_addr_gpu1;

    rd_addr_cpu[0]  = _cpu+0; // writes also read
    rd_addr_gpu0[0] = _gpu0+0;
    rd_addr_gpu1[0] = _gpu1+0;

    wt_addr_cpu[0]  = _cpu+1;
    wt_addr_gpu0[0] = _gpu0+2;
    wt_addr_gpu1[0] = _gpu1+3;

    // cudaMemcpy(argsDev0->rd_addrs, rd_addr_gpu0, size*sizeof(granule_t*), cudaMemcpyHostToDevice);
    // cudaMemcpy(argsDev0->wt_addrs, wt_addr_gpu0, size*sizeof(granule_t*), cudaMemcpyHostToDevice);
    // cudaMemcpy(argsDev1->rd_addrs, rd_addr_gpu1, size*sizeof(granule_t*), cudaMemcpyHostToDevice);
    // cudaMemcpy(argsDev1->wt_addrs, rd_addr_gpu1, size*sizeof(granule_t*), cudaMemcpyHostToDevice);

    HeTM_gshared_data.batchCount = 14;

    PR_curr_dev = 0;
    // argsDev0->nbPos = size_rwset;
    // argsDev0->base = _gpu0;
    // argsDev0->devId = 0;
    // launchKernel(&kernelObj, /*devId*/0, /*thrs*/1, /*blks*/1, argsDev0);

    PR_curr_dev = 1;
    // argsDev1->nbPos = size_rwset;
    // argsDev1->base = _gpu1;
    // argsDev1->devId = 1;
    // launchKernel(&kernelObj, /*devId*/1, /*thrs*/1, /*blks*/1, argsDev1);

    SetRdPositionInCPU(size_rwset, rd_addr_cpu);
    SetWrtPositionInCPU(size_rwset, wt_addr_cpu);

    // memman_select_device(0); // also checks if the device exists
    cudaDeviceSynchronize();

    // memman_select_device(1);
    cudaDeviceSynchronize(); // when emulating the second device this is redundant

    // do the comparison among GPUs
    notifyBatchIsDone(); // GPU controller

    // inspect write set of GPU0
    // memman_select_device(0);
    // memman_select("HeTM_gpu_wset");
    // memman_cpy_to_cpu(NULL, NULL, 0);
    // unsigned char *gpu0_wrtset = (unsigned char *)memman_get_cpu(NULL);
    cudaDeviceSynchronize();
    // printf("[GPU0] wrote in pos %i = %i\n", 2, gpu0_wrtset[2]);
    // printf("[GPU0] wrote in pos %i = %i\n", 4, gpu0_wrtset[4]);

    // inspect write set of GPU1
    // memman_select_device(1);
    // memman_select("HeTM_gpu_wset");
    // memman_cpy_to_cpu(NULL, NULL, 0);
    // unsigned char *gpu1_wrtset = (unsigned char *)memman_get_cpu(NULL);
    cudaDeviceSynchronize();
    // printf("[GPU1] wrote in pos %i = %i\n", 2, gpu1_wrtset[2]);
    // printf("[GPU1] wrote in pos %i = %i\n", 4, gpu1_wrtset[4]);

    // do the comparison with the CPU
    pollIsRoundComplete(/*nonBlock*/1); // TODO: this should not hang!

    syncGPUdataset((void*)1); // contains mergeMatricesAndRunFVS();

    CPPUNIT_ASSERT(mempool_base_addr_cpu[0][1] == 3 && "CPU id not in CPU mempool");
    CPPUNIT_ASSERT(mempool_base_addr_cpu[0][2] == 1 && "GPU0 id not in CPU mempool");
    CPPUNIT_ASSERT(mempool_base_addr_cpu[0][3] == 2 && "GPU1 id not in CPU mempool");

    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[0], mempool_base_addr_gpu[0], sizeof(granule_t)*MEMPOOL_SIZE);
    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[1], mempool_base_addr_gpu[1], sizeof(granule_t)*MEMPOOL_SIZE);

    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][1] == 3 && "CPU id not in GPU0 mempool");
    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][2] == 1 && "GPU0 id not in GPU0 mempool");
    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][3] == 2 && "GPU1 id not in GPU0 mempool");

    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][1] == 3 && "CPU id not in GPU1 mempool");
    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][2] == 1 && "GPU0 id not in GPU1 mempool");
    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][3] == 2 && "GPU1 id not in GPU1 mempool");

    // TODO: apply the dataset in DRAM
    printf("CPU is done with the merging\n");
}


void TestBMAP::TestAllConflict()
{
    granule_t *_cpu = mempool_base_addr_cpu[0];
    granule_t *_gpu0 = mempool_base_addr_gpu[0];
    granule_t *_gpu1 = mempool_base_addr_gpu[1];
    size_t size_gpu_wset;

    CPPUNIT_ASSERT(mempool_base_addr_cpu[0] == mempool_base_addr_cpu[1]); // CPU keeps only 1 memory pool

    // memman_select_device(1);
    // memman_select("HeTM_gpu_wset");
    // unsigned char *gpu1_wrtset = (unsigned char *)memman_get_cpu(&size_gpu_wset);
    // unsigned char *gpu1_wrtset_dev = (unsigned char *)memman_get_gpu(NULL);

    // memman_select_device(0);
    // memman_select("HeTM_gpu_wset");
    // unsigned char *gpu0_wrtset = (unsigned char *)memman_get_cpu(NULL);
    // unsigned char *gpu0_wrtset_dev = (unsigned char *)memman_get_gpu(NULL);

    PR_global[0].PR_blockNum  = 1;
    PR_global[0].PR_threadNum = 1;
    PR_global[1].PR_blockNum  = 1;
    PR_global[1].PR_threadNum = 1;

    // argsDev0->rd_addrs = rd_addr_gpu0;
    // argsDev0->wt_addrs = wt_addr_gpu0;
    // argsDev1->rd_addrs = rd_addr_gpu1;
    // argsDev1->wt_addrs = wt_addr_gpu1;

    rd_addr_cpu[0]  = _cpu+0; // writes also read
    rd_addr_gpu0[0] = _gpu0+0;
    rd_addr_gpu1[0] = _gpu1+0;

    wt_addr_cpu[0]  = _cpu+0;
    wt_addr_gpu0[0] = _gpu0+0;
    wt_addr_gpu1[0] = _gpu1+0;

    // cudaMemcpy(argsDev0->rd_addrs, rd_addr_gpu0, size*sizeof(granule_t*), cudaMemcpyHostToDevice);
    // cudaMemcpy(argsDev0->wt_addrs, wt_addr_gpu0, size*sizeof(granule_t*), cudaMemcpyHostToDevice);
    // cudaMemcpy(argsDev1->rd_addrs, rd_addr_gpu1, size*sizeof(granule_t*), cudaMemcpyHostToDevice);
    // cudaMemcpy(argsDev1->wt_addrs, rd_addr_gpu1, size*sizeof(granule_t*), cudaMemcpyHostToDevice);

    HeTM_gshared_data.batchCount = 15;

    PR_curr_dev = 0;
    // argsDev0->nbPos = size_rwset;
    // argsDev0->base = _gpu0;
    // argsDev0->devId = 0;
    // launchKernel(&kernelObj, /*devId*/0, /*thrs*/1, /*blks*/1, argsDev0);

    PR_curr_dev = 1;
    // argsDev1->nbPos = size_rwset;
    // argsDev1->base = _gpu1;
    // argsDev1->devId = 1;
    // launchKernel(&kernelObj, /*devId*/1, /*thrs*/1, /*blks*/1, argsDev1);

    SetRdPositionInCPU(size_rwset, rd_addr_cpu);
    SetWrtPositionInCPU(size_rwset, wt_addr_cpu);

    // memman_select_device(0); // also checks if the device exists
    cudaDeviceSynchronize();

    // memman_select_device(1);
    cudaDeviceSynchronize(); // when emulating the second device this is redundant

    // do the comparison among GPUs
    notifyBatchIsDone(); // GPU controller

    // inspect write set of GPU0
    // memman_select_device(0);
    // CUDA_CPY_TO_HOST(gpu0_wrtset, gpu0_wrtset_dev, size_gpu_wset);
    // memman_select("HeTM_gpu_wset");
    // memman_cpy_to_cpu(NULL, NULL, 0);
    cudaDeviceSynchronize();
    // printf("[GPU0] wrote in pos %i = %i\n", 0, gpu0_wrtset[0]);
    // printf("[GPU0] wrote in pos %i = %i\n", 1, gpu0_wrtset[1]);

    // inspect write set of GPU1
    // memman_select_device(1);
    // CUDA_CPY_TO_HOST(gpu1_wrtset, gpu1_wrtset_dev, size_gpu_wset);
    // memman_select("HeTM_gpu_wset");
    // memman_cpy_to_cpu(NULL, NULL, 0);
    cudaDeviceSynchronize();
    // printf("[GPU1] wrote in pos %i = %i\n", 0, gpu1_wrtset[0]);
    // printf("[GPU1] wrote in pos %i = %i\n", 1, gpu1_wrtset[1]);

    // do the comparison with the CPU
    pollIsRoundComplete(/*nonBlock*/1); // TODO: this should not hang!

    syncGPUdataset((void*)1); // contains mergeMatricesAndRunFVS();

    // TODO: what is the result if all conflict? who wins?
    // CPPUNIT_ASSERT(mempool_base_addr_cpu[0][1] == 3 && "CPU id not in CPU mempool");
    // CPPUNIT_ASSERT(mempool_base_addr_cpu[0][2] == 1 && "GPU0 id not in CPU mempool");
    // CPPUNIT_ASSERT(mempool_base_addr_cpu[0][3] == 2 && "GPU1 id not in CPU mempool");

    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[0], mempool_base_addr_gpu[0], sizeof(granule_t)*MEMPOOL_SIZE);
    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[1], mempool_base_addr_gpu[1], sizeof(granule_t)*MEMPOOL_SIZE);

    // CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][1] == 3 && "CPU id not in GPU0 mempool");
    // CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][2] == 1 && "GPU0 id not in GPU0 mempool");
    // CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][3] == 2 && "GPU1 id not in GPU0 mempool");

    // CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][1] == 3 && "CPU id not in GPU1 mempool");
    // CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][2] == 1 && "GPU0 id not in GPU1 mempool");
    // CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][3] == 2 && "GPU1 id not in GPU1 mempool");

    // TODO: apply the dataset in DRAM
    printf("CPU is done with the merging\n");
}

void TestBMAP::TestGPUsConflict()
{
    granule_t *_cpu = mempool_base_addr_cpu[0];
    granule_t *_gpu0 = mempool_base_addr_gpu[0];
    granule_t *_gpu1 = mempool_base_addr_gpu[1];

    CPPUNIT_ASSERT(mempool_base_addr_cpu[0] == mempool_base_addr_cpu[1]); // CPU keeps only 1 memory pool


    PR_global[0].PR_blockNum  = 1;
    PR_global[0].PR_threadNum = 1;
    PR_global[1].PR_blockNum  = 1;
    PR_global[1].PR_threadNum = 1;

    // argsDev0->rd_addrs = rd_addr_gpu0;
    // argsDev0->wt_addrs = wt_addr_gpu0;
    // argsDev1->rd_addrs = rd_addr_gpu1;
    // argsDev1->wt_addrs = wt_addr_gpu1;

    rd_addr_cpu[0]  = _cpu+1; // writes also read
    rd_addr_gpu0[0] = _gpu0+0;
    rd_addr_gpu1[0] = _gpu1+0;

    wt_addr_cpu[0]  = _cpu+1;
    wt_addr_gpu0[0] = _gpu0+0;
    wt_addr_gpu1[0] = _gpu1+0;

    // cudaMemcpy(argsDev0->rd_addrs, rd_addr_gpu0, size*sizeof(granule_t*), cudaMemcpyHostToDevice);
    // cudaMemcpy(argsDev0->wt_addrs, wt_addr_gpu0, size*sizeof(granule_t*), cudaMemcpyHostToDevice);
    // cudaMemcpy(argsDev1->rd_addrs, rd_addr_gpu1, size*sizeof(granule_t*), cudaMemcpyHostToDevice);
    // cudaMemcpy(argsDev1->wt_addrs, rd_addr_gpu1, size*sizeof(granule_t*), cudaMemcpyHostToDevice);

    HeTM_gshared_data.batchCount = 16;

    PR_curr_dev = 0;
    // argsDev0->nbPos = size_rwset;
    // argsDev0->base = _gpu0;
    // argsDev0->devId = 0;
    // launchKernel(&kernelObj, /*devId*/0, /*thrs*/1, /*blks*/1, argsDev0);

    PR_curr_dev = 1;
    // argsDev1->nbPos = size_rwset;
    // argsDev1->base = _gpu1;
    // argsDev1->devId = 1;
    // launchKernel(&kernelObj, /*devId*/1, /*thrs*/1, /*blks*/1, argsDev1);

    SetRdPositionInCPU(size_rwset, rd_addr_cpu);
    SetWrtPositionInCPU(size_rwset, wt_addr_cpu);

    // memman_select_device(0); // also checks if the device exists
    cudaDeviceSynchronize();

    // memman_select_device(1);
    cudaDeviceSynchronize(); // when emulating the second device this is redundant

    // do the comparison among GPUs
    notifyBatchIsDone(); // GPU controller

    // inspect write set of GPU0
    // memman_select_device(0);
    // memman_select("HeTM_gpu_wset");
    // memman_cpy_to_cpu(NULL, NULL, 0);
    // unsigned char *gpu0_wrtset = (unsigned char *)memman_get_cpu(NULL);
    cudaDeviceSynchronize();
    // printf("[GPU0] wrote in pos %i = %i\n", 2, gpu0_wrtset[2]);
    // printf("[GPU0] wrote in pos %i = %i\n", 4, gpu0_wrtset[4]);

    // inspect write set of GPU1
    // memman_select_device(1);
    // memman_select("HeTM_gpu_wset");
    // memman_cpy_to_cpu(NULL, NULL, 0);
    // unsigned char *gpu1_wrtset = (unsigned char *)memman_get_cpu(NULL);
    cudaDeviceSynchronize();
    // printf("[GPU1] wrote in pos %i = %i\n", 2, gpu1_wrtset[2]);
    // printf("[GPU1] wrote in pos %i = %i\n", 4, gpu1_wrtset[4]);

    // do the comparison with the CPU
    pollIsRoundComplete(/*nonBlock*/1); // TODO: this should not hang!

    syncGPUdataset((void*)1); // contains mergeMatricesAndRunFVS();

    // TODO: what is the result if all conflict? who wins?
    // CPPUNIT_ASSERT(mempool_base_addr_cpu[0][1] == 3 && "CPU id not in CPU mempool");
    // CPPUNIT_ASSERT(mempool_base_addr_cpu[0][2] == 1 && "GPU0 id not in CPU mempool");
    // CPPUNIT_ASSERT(mempool_base_addr_cpu[0][3] == 2 && "GPU1 id not in CPU mempool");

    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[0], mempool_base_addr_gpu[0], sizeof(granule_t)*MEMPOOL_SIZE);
    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[1], mempool_base_addr_gpu[1], sizeof(granule_t)*MEMPOOL_SIZE);

    // CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][1] == 3 && "CPU id not in GPU0 mempool");
    // CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][2] == 1 && "GPU0 id not in GPU0 mempool");
    // CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][3] == 2 && "GPU1 id not in GPU0 mempool");

    // CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][1] == 3 && "CPU id not in GPU1 mempool");
    // CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][2] == 1 && "GPU0 id not in GPU1 mempool");
    // CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][3] == 2 && "GPU1 id not in GPU1 mempool");

    // TODO: apply the dataset in DRAM
    printf("CPU is done with the merging\n");
}

void TestBMAP::TestCache()
{

}

void SetWrtPositionInCPU(size_t nbPos, granule_t **addrs)
{
    for (int i = 0; i < nbPos; ++i) {
        // some arguments are not meant for BMAP
        stm_log_newentry(/*vers log*/NULL, (long*)addrs[i], /*val*/0, /*vers*/0);
        *addrs[i] = NB_GPUS+1; // CPUid
    }
}

void SetRdPositionInCPU(size_t nbPos, granule_t **addrs)
{
    for (int i = 0; i < nbPos; ++i) {
        stm_log_read_entry((long*)addrs[i]);
    }
}

__device__ void SetWrtPositionInGPU(PR_txCallDefArgs, size_t nbPos, granule_t **addrs)
{
    HeTM_GPU_log_s *GPU_log = (HeTM_GPU_log_s*)pr_args.pr_args_ext;
    // printf("[GPU%i] batch = %li\n", GPU_log->devId, GPU_log->batchCount);
    for (int i = 0; i < nbPos; ++i) {
        // memman_access_addr_dev(GPU_log->bmap, args->wset.addrs[i], GPU_log->batchCount);

        SET_ON_RS_BMAP(addrs[i]); // writes also read

        // uintptr_t _wsetAddr = (uintptr_t)(addrs[i]);
        // uintptr_t _devwBAddr = (uintptr_t)GPU_log->devMemPoolBasePtr;
        // uintptr_t _wpos = (_wsetAddr - _devwBAddr) >> (PR_LOCK_GRAN_BITS+HETM_REDUCED_RS);
        // void *_WSetBitmap = GPU_log->bmap_wset_devptr;
        // unsigned char *BM_bitmap = (unsigned char*)_WSetBitmap;
        // BM_bitmap[_wpos] = (unsigned char)GPU_log->batchCount;
        SET_ON_WS_BMAP(addrs[i]); // above: expansion of this macro
        printf("[dev%i]Set write %lu\n", GPU_log->devId, _wpos);
        *(addrs[i]) = GPU_log->devId+1;
    }
}

__device__ void SetRdPositionInGPU(PR_txCallDefArgs, size_t nbPos, granule_t **addrs)
{
    HeTM_GPU_log_s *GPU_log = (HeTM_GPU_log_s*)pr_args.pr_args_ext;
    for (int i = 0; i < nbPos; ++i) {
        // memman_access_addr_dev(GPU_log->bmap, args->wset.addrs[i], GPU_log->batchCount);
        // SET_ON_RS_BMAP(pr_args.wset.addrs[i]);
        SET_ON_RS_BMAP(addrs[i]);
        // printf("Set read %lu\n", _pos);
    }
}

__global__ void kernel_setPositions(PR_globalKernelArgs)
{
    int tid = PR_THREAD_IDX;
    PR_enterKernel(tid);
    knlEntryObj_s *entryObj = (knlEntryObj_s*)pr_args.inBuf;
    SetRdPositionInGPU(PR_txCallArgs, entryObj->nbPos, entryObj->rd_addrs);
    SetWrtPositionInGPU(PR_txCallArgs, entryObj->nbPos, entryObj->wt_addrs);
    PR_exitKernel();
}

static void run_kernelGPUsetPos(knlman_callback_params_s params)
{
    dim3 blocks(params.blocks.x, params.blocks.y, params.blocks.z);
    dim3 threads(params.threads.x, params.threads.y, params.threads.z);
    cudaStream_t stream = (cudaStream_t)params.stream;
    knlEntryObj_s *entryObj = (knlEntryObj_s*)params.entryObj;
    pr_buffer_s inBuf, outBuf;

    pr_tx_args_s *pr_args = getPrSTMmetaData(entryObj->devId);
    //   params.currDev 

    inBuf.buf = entryObj;
    inBuf.size = sizeof(knlEntryObj_s);
    outBuf.buf = NULL;
    outBuf.size = 0;
    PR_prepareIO(pr_args, inBuf, outBuf);
    PR_run(kernel_setPositions, pr_args);
}

