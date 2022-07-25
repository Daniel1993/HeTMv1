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
using namespace memman;
using namespace knlman;

typedef PR_GRANULE_T granule_t;
static const int NB_GPUS              = 2; // TODO: compile with HETM_NB_DEVICES=2
static const size_t MEMPOOL_SIZE      = BMAP_GRAN*4; 
static const char *STR_MEMPOOL        = "HeTM_mempool";
static const char *STR_BMAP_GPU       = "HeTM_mempool_bmap";
static const char *STR_CPU_WS         = "HeTM_cpu_wset";
static const char *STR_CPU_WS_CACHE   = "HeTM_cpu_wset_cache";

static granule_t *mempool_base_addr_cpu[NB_GPUS];
static granule_t *mempool_base_addr_gpu[NB_GPUS];
static granule_t *cpu_buffer_of_gpu_mempool[NB_GPUS];

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
static KnlObj *test_knl;
static MemObjOnDev test_knl_args;
static knlEntryObj_s *argsDev0, *argsDev1;
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
        .mempool_opts = MEMMAN_NONE
	});
    
    for (int i = 0; i < NB_GPUS; ++i)
    {
        // memman_select_device(i);
        // memman_select(STR_MEMPOOL);
        MemObj *m = HeTM_mempool.GetMemObj(i);
        mempool_base_addr_cpu[i] = (granule_t*)m->host;
        mempool_base_addr_gpu[i] = (granule_t*)m->dev;

        // initialized in hetm-threading.cu (HeTM_thread_data is thread local)
        HeTM_thread_data[i] = &(HeTM_shared_data[i].threadsInfo[0]); // only 1 thread
        HeTM_thread_data[i]->devId = i;
    }
    // knlmanObjCreate(&kernelObj, "kernelGPUsetPos", run_kernelGPUsetPos, 0);

    cudaMemset(mempool_base_addr_cpu[0], 0, MEMPOOL_SIZE*sizeof(granule_t));
    
    Config::GetInstance()->SelDev(0);
    cudaMemset(mempool_base_addr_gpu[0], 0, MEMPOOL_SIZE*sizeof(granule_t));
    
    Config::GetInstance()->SelDev(1);
    cudaMemset(mempool_base_addr_gpu[1], 0, MEMPOOL_SIZE*sizeof(granule_t));

    cudaMallocManaged(&argsDev0, sizeof(knlEntryObj_s));
    cudaMallocManaged(&argsDev1, sizeof(knlEntryObj_s));
    MemObjBuilder b0;
    MemObjBuilder b1;
    KnlObjBuilder b;
    test_knl_args.AddMemObj(new MemObj(b0
        .SetSize(sizeof(knlEntryObj_s))
        ->SetDevPtr(argsDev0)
        ->SetHostPtr(argsDev0), 0));
    test_knl_args.AddMemObj(new MemObj(b1
        .SetSize(sizeof(knlEntryObj_s))
        ->SetDevPtr(argsDev1)
        ->SetHostPtr(argsDev1), 1));
    test_knl = new KnlObj(b
        .SetBlocks({ .x = 1, .y = 1, .z = 1})
        ->SetThreads({ .x = 1, .y = 1, .z = 1})
        ->SetCallback(run_kernelGPUsetPos)
        ->SetEntryObj(&test_knl_args));

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
    cudaFree(argsDev0);
    cudaFree(argsDev1);
    cudaFree(cpu_buffer_of_gpu_mempool[0]);
    cudaFree(cpu_buffer_of_gpu_mempool[1]);
    cudaFree(rd_addr_cpu);
    cudaFree(rd_addr_gpu0);
    cudaFree(rd_addr_gpu1);
    cudaFree(wt_addr_cpu);
    cudaFree(wt_addr_gpu0);
    cudaFree(wt_addr_gpu1);
    delete test_knl_args.GetMemObj(0);
    delete test_knl_args.GetMemObj(1);
    delete test_knl;
    // knlmanObjDestroy(&kernelObj);
    HeTM_destroy();
}

static void launchKernel(
    int devId
) {
    // Config::GetInstance()->SelDev(devId);
    test_knl->Run(devId);
    // kernelObj->select(kernelObj);
    // kernelObj->setNbBlocks(blks, 1, 1);
    // kernelObj->setThrsPerBlock(thrs, 1, 1);
    // kernelObj->setDevice(devId);
    // kernelObj->setStream(NULL); // Default stream
    // kernelObj->setArgs(args);
    // kernelObj->run();
}

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

    argsDev0->rd_addrs = rd_addr_gpu0;
    argsDev0->wt_addrs = wt_addr_gpu0;
    argsDev1->rd_addrs = rd_addr_gpu1;
    argsDev1->wt_addrs = wt_addr_gpu1;

    rd_addr_cpu[0]  = _cpu+0;
    rd_addr_gpu0[0] = _gpu0+1;
    rd_addr_gpu1[0] = _gpu1+2;

    wt_addr_cpu[0]  = _cpu+0;
    wt_addr_gpu0[0] = _gpu0+1;
    wt_addr_gpu1[0] = _gpu1+2;

    cudaMemcpy(argsDev0->rd_addrs, rd_addr_gpu0, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(argsDev0->wt_addrs, wt_addr_gpu0, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(argsDev1->rd_addrs, rd_addr_gpu1, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(argsDev1->wt_addrs, rd_addr_gpu1, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);

    HeTM_gshared_data.batchCount = 13;

    PR_curr_dev = 0;
    argsDev0->nbPos = size_rwset;
    argsDev0->base = _gpu0;
    argsDev0->devId = 0;
    launchKernel(/*devId*/0);

    PR_curr_dev = 1;
    argsDev1->nbPos = 1;
    argsDev1->base = _gpu1;
    argsDev1->devId = 1;
    launchKernel(/*devId*/1);

    SetRdPositionInCPU(size_rwset, rd_addr_cpu);
    SetWrtPositionInCPU(size_rwset, wt_addr_cpu);

    Config::GetInstance()->SelDev(0);
    cudaDeviceSynchronize();

    Config::GetInstance()->SelDev(1);
    cudaDeviceSynchronize(); // when emulating the second device this is redundant

    // do the comparison among GPUs
    notifyBatchIsDone(); // GPU controller

    // inspect write set of GPU0
    Config::GetInstance()->SelDev(0);
    MemObj *m_gpu0_wrtset = HeTM_gpu_wset.GetMemObj(0);
    m_gpu0_wrtset->CpyDtH();
    unsigned char *gpu0_wrtset = (unsigned char *)m_gpu0_wrtset->host;
    cudaDeviceSynchronize();
    printf("[GPU0] wrote in pos %i = %i\n", 1, gpu0_wrtset[1]);
    printf("[GPU0] wrote in pos %i = %i\n", 2, gpu0_wrtset[2]);

    // inspect write set of GPU1
    Config::GetInstance()->SelDev(1);
    MemObj *m_gpu1_wrtset = HeTM_gpu_wset.GetMemObj(1);
    m_gpu1_wrtset->CpyDtH();
    unsigned char *gpu1_wrtset = (unsigned char *)m_gpu1_wrtset->host;
    cudaDeviceSynchronize();
    printf("[GPU1] wrote in pos %i = %i\n", 1, gpu1_wrtset[1]);
    printf("[GPU1] wrote in pos %i = %i\n", 2, gpu1_wrtset[2]);

    // for debugging
    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[0], mempool_base_addr_gpu[0], sizeof(granule_t)*MEMPOOL_SIZE);
    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[1], mempool_base_addr_gpu[1], sizeof(granule_t)*MEMPOOL_SIZE);

    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][1] == 1 && "GPU0 id not in GPU0 mempool");
    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][2] == 2 && "GPU1 id not in GPU1 mempool");

    // do the comparison with the CPU
    cpyCPUwrtsetToGPU(1);
    // sleep(2);

    // for debugging
    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[0], mempool_base_addr_gpu[0], sizeof(granule_t)*MEMPOOL_SIZE);
    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[1], mempool_base_addr_gpu[1], sizeof(granule_t)*MEMPOOL_SIZE);

    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][1] == 1 && "GPU0 id not in GPU0 mempool");
    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][2] == 2 && "GPU1 id not in GPU1 mempool");

    // syncGPUdataset((void*)1); // contains mergeMatricesAndRunFVS();
    mergeMatricesAndRunFVS(1);

    // copies the merged dataset
    waitNextBatch(1);
    doGPUStateReset();

    Config::GetInstance()->SelDev(0);
    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[0], mempool_base_addr_gpu[0], sizeof(granule_t)*MEMPOOL_SIZE);
    cudaDeviceSynchronize();
    Config::GetInstance()->SelDev(1);
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

void TestBMAP::TestRWDifferentPositions2()
{
    granule_t *_cpu = mempool_base_addr_cpu[0];
    granule_t *_gpu0 = mempool_base_addr_gpu[0];
    granule_t *_gpu1 = mempool_base_addr_gpu[1];

    CPPUNIT_ASSERT(mempool_base_addr_cpu[0] == mempool_base_addr_cpu[1]); // CPU keeps only 1 memory pool

    PR_global[0].PR_blockNum  = 1;
    PR_global[0].PR_threadNum = 1;
    PR_global[1].PR_blockNum  = 1;
    PR_global[1].PR_threadNum = 1;

    argsDev0->rd_addrs = rd_addr_gpu0;
    argsDev0->wt_addrs = wt_addr_gpu0;
    argsDev1->rd_addrs = rd_addr_gpu1;
    argsDev1->wt_addrs = wt_addr_gpu1;

    rd_addr_cpu[0]  = _cpu+1+BMAP_GRAN;
    rd_addr_gpu0[0] = _gpu0+2+BMAP_GRAN;
    rd_addr_gpu1[0] = _gpu1+3+BMAP_GRAN;

    wt_addr_cpu[0]  = _cpu+1+BMAP_GRAN;
    wt_addr_gpu0[0] = _gpu0+2+BMAP_GRAN;
    wt_addr_gpu1[0] = _gpu1+3+BMAP_GRAN;

    cudaMemcpy(argsDev0->rd_addrs, rd_addr_gpu0, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(argsDev0->wt_addrs, wt_addr_gpu0, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(argsDev1->rd_addrs, rd_addr_gpu1, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(argsDev1->wt_addrs, wt_addr_gpu1, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);

    HeTM_gshared_data.batchCount = 14;

    PR_curr_dev = 0;
    argsDev0->nbPos = size_rwset;
    argsDev0->base = _gpu0;
    argsDev0->devId = 0;
    launchKernel(/*devId*/0);

    PR_curr_dev = 1;
    argsDev1->nbPos = 1;
    argsDev1->base = _gpu1;
    argsDev1->devId = 1;
    launchKernel(/*devId*/1);

    SetRdPositionInCPU(size_rwset, rd_addr_cpu);
    SetWrtPositionInCPU(size_rwset, wt_addr_cpu);

    Config::GetInstance()->SelDev(0);
    cudaDeviceSynchronize();

    Config::GetInstance()->SelDev(1);
    cudaDeviceSynchronize(); // when emulating the second device this is redundant

    // do the comparison among GPUs
    notifyBatchIsDone(); // GPU controller

    // inspect write set of GPU0
    Config::GetInstance()->SelDev(0);
    MemObj *m_gpu0_wrtset = HeTM_gpu_wset.GetMemObj(0);
    m_gpu0_wrtset->CpyDtH();
    unsigned char *gpu0_wrtset = (unsigned char *)m_gpu0_wrtset->host;
    cudaDeviceSynchronize();
    printf("[GPU0] wrote in pos %i = %i\n", 2+BMAP_GRAN, (int) gpu0_wrtset[(2+BMAP_GRAN)
#ifdef BMAP_ENC_1BIT
    /8
#endif
    ]);

    // inspect write set of GPU1
    Config::GetInstance()->SelDev(1);
    MemObj *m_gpu1_wrtset = HeTM_gpu_wset.GetMemObj(1);
    m_gpu1_wrtset->CpyDtH();
    unsigned char *gpu1_wrtset = (unsigned char *)m_gpu1_wrtset->host;
    cudaDeviceSynchronize();
    printf("[GPU1] wrote in pos %i = %i\n", 3+BMAP_GRAN, (int) gpu1_wrtset[(3+BMAP_GRAN)
#ifdef BMAP_ENC_1BIT
    /8
#endif
    ]);

    // for debugging
    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[0], mempool_base_addr_gpu[0], sizeof(granule_t)*MEMPOOL_SIZE);
    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[1], mempool_base_addr_gpu[1], sizeof(granule_t)*MEMPOOL_SIZE);

    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][2+BMAP_GRAN] == 1 && "GPU0 id not in GPU0 mempool");
    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][3+BMAP_GRAN] == 2 && "GPU1 id not in GPU1 mempool");

    // do the comparison with the CPU
    cpyCPUwrtsetToGPU(1);
    // sleep(2);

    // for debugging
    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[0], mempool_base_addr_gpu[0], sizeof(granule_t)*MEMPOOL_SIZE);
    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[1], mempool_base_addr_gpu[1], sizeof(granule_t)*MEMPOOL_SIZE);

    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][2+BMAP_GRAN] == 1 && "GPU0 id not in GPU0 mempool");
    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][3+BMAP_GRAN] == 2 && "GPU1 id not in GPU1 mempool");

    // syncGPUdataset((void*)1); // contains mergeMatricesAndRunFVS();
    mergeMatricesAndRunFVS(1);

    // copies the merged dataset
    waitNextBatch(1);
    doGPUStateReset();

    Config::GetInstance()->SelDev(0);
    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[0], mempool_base_addr_gpu[0], sizeof(granule_t)*MEMPOOL_SIZE);
    cudaDeviceSynchronize();
    Config::GetInstance()->SelDev(1);
    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[1], mempool_base_addr_gpu[1], sizeof(granule_t)*MEMPOOL_SIZE);
    cudaDeviceSynchronize();

    CPPUNIT_ASSERT(mempool_base_addr_cpu[0][1+BMAP_GRAN] == 3 && "CPU id not in mempool");
    CPPUNIT_ASSERT(mempool_base_addr_cpu[0][2+BMAP_GRAN] == 1 && "GPU0 id not in mempool");
    CPPUNIT_ASSERT(mempool_base_addr_cpu[0][3+BMAP_GRAN] == 2 && "GPU1 id not in mempool");

    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][1+BMAP_GRAN] == 3 && "CPU id not in GPU0 mempool");
    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][2+BMAP_GRAN] == 1 && "GPU0 id not in GPU0 mempool");
    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][3+BMAP_GRAN] == 2 && "GPU1 id not in GPU0 mempool");

    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][1+BMAP_GRAN] == 3 && "CPU id not in GPU1 mempool");
    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][2+BMAP_GRAN] == 1 && "GPU0 id not in GPU1 mempool");
    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][3+BMAP_GRAN] == 2 && "GPU1 id not in GPU1 mempool");

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

    argsDev0->rd_addrs = rd_addr_gpu0;
    argsDev0->wt_addrs = wt_addr_gpu0;
    argsDev1->rd_addrs = rd_addr_gpu1;
    argsDev1->wt_addrs = wt_addr_gpu1;

    rd_addr_cpu[0]  = _cpu+0; // writes also read
    rd_addr_gpu0[0] = _gpu0+0;
    rd_addr_gpu1[0] = _gpu1+0;

    wt_addr_cpu[0]  = _cpu+1;
    wt_addr_gpu0[0] = _gpu0+2;
    wt_addr_gpu1[0] = _gpu1+3;

    cudaMemcpy(argsDev0->rd_addrs, rd_addr_gpu0, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(argsDev0->wt_addrs, wt_addr_gpu0, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(argsDev1->rd_addrs, rd_addr_gpu1, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(argsDev1->wt_addrs, wt_addr_gpu1, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);

    HeTM_gshared_data.batchCount = 11;

    PR_curr_dev = 0;
    argsDev0->nbPos = size_rwset;
    argsDev0->base = _gpu0;
    argsDev0->devId = 0;
    launchKernel(/*devId*/0);

    PR_curr_dev = 1;
    argsDev1->nbPos = size_rwset;
    argsDev1->base = _gpu1;
    argsDev1->devId = 1;
    launchKernel(/*devId*/1);

    SetRdPositionInCPU(size_rwset, rd_addr_cpu);
    SetWrtPositionInCPU(size_rwset, wt_addr_cpu);

    Config::GetInstance()->SelDev(0);
    cudaDeviceSynchronize();

    Config::GetInstance()->SelDev(1);
    cudaDeviceSynchronize(); // when emulating the second device this is redundant

    // do the comparison among GPUs
    notifyBatchIsDone(); // GPU controller

    // inspect write set of GPU0
    Config::GetInstance()->SelDev(0);
    MemObj *m_gpu0_wrtset = HeTM_gpu_wset.GetMemObj(0);
    m_gpu0_wrtset->CpyDtH();
    unsigned char *gpu0_wrtset = (unsigned char *)m_gpu0_wrtset->host;
    cudaDeviceSynchronize();
    printf("[GPU0] wrote in pos %i = %i\n", 2, gpu0_wrtset[2]);
    printf("[GPU0] wrote in pos %i = %i\n", 4, gpu0_wrtset[4]);

    // inspect write set of GPU1
    Config::GetInstance()->SelDev(1);
    MemObj *m_gpu1_wrtset = HeTM_gpu_wset.GetMemObj(1);
    m_gpu1_wrtset->CpyDtH();
    unsigned char *gpu1_wrtset = (unsigned char *)m_gpu1_wrtset->host;
    cudaDeviceSynchronize();
    printf("[GPU1] wrote in pos %i = %i\n", 2, gpu1_wrtset[2]);
    printf("[GPU1] wrote in pos %i = %i\n", 4, gpu1_wrtset[4]);

    // do the comparison with the CPU
    cpyCPUwrtsetToGPU(1);
    // sleep(2);

    // syncGPUdataset((void*)1); // contains mergeMatricesAndRunFVS();
    mergeMatricesAndRunFVS(1);

    waitNextBatch(1);
    doGPUStateReset();

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

    Config::GetInstance()->SelDev(1);
    MemObj *m_gpu1_wrtset = HeTM_gpu_wset.GetMemObj(1);
    size_gpu_wset = m_gpu1_wrtset->size;
    unsigned char *gpu1_wrtset = (unsigned char *)m_gpu1_wrtset->host;
    // unsigned char *gpu1_wrtset_dev = (unsigned char *)m_gpu1_wrtset->dev;

    Config::GetInstance()->SelDev(0);
    MemObj *m_gpu0_wrtset = HeTM_gpu_wset.GetMemObj(0);
    unsigned char *gpu0_wrtset = (unsigned char *)m_gpu0_wrtset->host;
    // unsigned char *gpu0_wrtset_dev = (unsigned char *)m_gpu0_wrtset->dev;

    PR_global[0].PR_blockNum  = 1;
    PR_global[0].PR_threadNum = 1;
    PR_global[1].PR_blockNum  = 1;
    PR_global[1].PR_threadNum = 1;

    argsDev0->rd_addrs = rd_addr_gpu0;
    argsDev0->wt_addrs = wt_addr_gpu0;
    argsDev1->rd_addrs = rd_addr_gpu1;
    argsDev1->wt_addrs = wt_addr_gpu1;

    rd_addr_cpu[0]  = _cpu+0; // writes also read
    rd_addr_gpu0[0] = _gpu0+0;
    rd_addr_gpu1[0] = _gpu1+0;

    wt_addr_cpu[0]  = _cpu+0;
    wt_addr_gpu0[0] = _gpu0+0;
    wt_addr_gpu1[0] = _gpu1+0;

    cudaMemcpy(argsDev0->rd_addrs, rd_addr_gpu0, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(argsDev0->wt_addrs, wt_addr_gpu0, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(argsDev1->rd_addrs, rd_addr_gpu1, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(argsDev1->wt_addrs, wt_addr_gpu1, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);

    HeTM_gshared_data.batchCount = 15;

    PR_curr_dev = 0;
    argsDev0->nbPos = size_rwset;
    argsDev0->base = _gpu0;
    argsDev0->devId = 0;
    launchKernel(/*devId*/0);

    PR_curr_dev = 1;
    argsDev1->nbPos = size_rwset;
    argsDev1->base = _gpu1;
    argsDev1->devId = 1;
    launchKernel(/*devId*/1);

    SetRdPositionInCPU(size_rwset, rd_addr_cpu);
    SetWrtPositionInCPU(size_rwset, wt_addr_cpu);

    Config::GetInstance()->SelDev(0);
    cudaDeviceSynchronize();

    Config::GetInstance()->SelDev(1);
    cudaDeviceSynchronize(); // when emulating the second device this is redundant

    // do the comparison among GPUs
    notifyBatchIsDone(); // GPU controller

    // inspect write set of GPU0
    Config::GetInstance()->SelDev(0);
    m_gpu0_wrtset = HeTM_gpu_wset.GetMemObj(0);
    m_gpu0_wrtset->CpyDtH();
    gpu0_wrtset = (unsigned char *)m_gpu0_wrtset->host;
    cudaDeviceSynchronize();
    printf("[GPU0] wrote in pos %i = %i\n", 0, gpu0_wrtset[0]);
    printf("[GPU0] wrote in pos %i = %i\n", 1, gpu0_wrtset[1]);

    // inspect write set of GPU1
    Config::GetInstance()->SelDev(1);
    m_gpu1_wrtset = HeTM_gpu_wset.GetMemObj(1);
    m_gpu1_wrtset->CpyDtH();
    gpu1_wrtset = (unsigned char *)m_gpu1_wrtset->host;
    cudaDeviceSynchronize();
    printf("[GPU1] wrote in pos %i = %i\n", 0, gpu1_wrtset[0]);
    printf("[GPU1] wrote in pos %i = %i\n", 1, gpu1_wrtset[1]);

    // do the comparison with the CPU
    cpyCPUwrtsetToGPU(1);
    // sleep(2);

    // syncGPUdataset((void*)1); // contains mergeMatricesAndRunFVS();
    mergeMatricesAndRunFVS(1);

    waitNextBatch(1);
    doGPUStateReset();

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

    argsDev0->rd_addrs = rd_addr_gpu0;
    argsDev0->wt_addrs = wt_addr_gpu0;
    argsDev1->rd_addrs = rd_addr_gpu1;
    argsDev1->wt_addrs = wt_addr_gpu1;

    rd_addr_cpu[0]  = _cpu+1; // writes also read
    rd_addr_gpu0[0] = _gpu0+0;
    rd_addr_gpu1[0] = _gpu1+0;

    wt_addr_cpu[0]  = _cpu+1;
    wt_addr_gpu0[0] = _gpu0+0;
    wt_addr_gpu1[0] = _gpu1+0;

    cudaMemcpy(argsDev0->rd_addrs, rd_addr_gpu0, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(argsDev0->wt_addrs, wt_addr_gpu0, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(argsDev1->rd_addrs, rd_addr_gpu1, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(argsDev1->wt_addrs, wt_addr_gpu1, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);

    HeTM_gshared_data.batchCount = 16;

    PR_curr_dev = 0;
    argsDev0->nbPos = size_rwset;
    argsDev0->base = _gpu0;
    argsDev0->devId = 0;
    launchKernel(/*devId*/0);

    PR_curr_dev = 1;
    argsDev1->nbPos = size_rwset;
    argsDev1->base = _gpu1;
    argsDev1->devId = 1;
    launchKernel(/*devId*/1);

    SetRdPositionInCPU(size_rwset, rd_addr_cpu);
    SetWrtPositionInCPU(size_rwset, wt_addr_cpu);

    Config::GetInstance()->SelDev(0);
    cudaDeviceSynchronize();

    Config::GetInstance()->SelDev(1);
    cudaDeviceSynchronize(); // when emulating the second device this is redundant

    // do the comparison among GPUs
    notifyBatchIsDone(); // GPU controller

    // inspect write set of GPU0
    Config::GetInstance()->SelDev(0);
    MemObj *m_gpu0_wrtset = HeTM_gpu_wset.GetMemObj(0);
    m_gpu0_wrtset->CpyDtH();
    unsigned char *gpu0_wrtset = (unsigned char *)m_gpu0_wrtset->host;
    cudaDeviceSynchronize();
    printf("[GPU0] wrote in pos %i = %i\n", 0, gpu0_wrtset[0]);

    // inspect write set of GPU1
    Config::GetInstance()->SelDev(1);
    MemObj *m_gpu1_wrtset = HeTM_gpu_wset.GetMemObj(1);
    m_gpu1_wrtset->CpyDtH();
    unsigned char *gpu1_wrtset = (unsigned char *)m_gpu1_wrtset->host;
    cudaDeviceSynchronize();
    printf("[GPU1] wrote in pos %i = %i\n", 0, gpu1_wrtset[0]);

    // do the comparison with the CPU
    cpyCPUwrtsetToGPU(1);
    // sleep(2);

    // syncGPUdataset((void*)1); // contains mergeMatricesAndRunFVS();
    mergeMatricesAndRunFVS(1);

    waitNextBatch(1);
    doGPUStateReset();

    // TODO: what is the result if all conflict? who wins?
    CPPUNIT_ASSERT(mempool_base_addr_cpu[0][1] == 3 && "CPU id not in CPU mempool");
    CPPUNIT_ASSERT((mempool_base_addr_cpu[0][0] == 1 || mempool_base_addr_cpu[0][0] == 2) && "id of neither GPU0 nor GPU1 in CPU mempool");

    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[0], mempool_base_addr_gpu[0], sizeof(granule_t)*MEMPOOL_SIZE);
    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[1], mempool_base_addr_gpu[1], sizeof(granule_t)*MEMPOOL_SIZE);

    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][1] == 3 && "CPU id not in GPU0 mempool");
    CPPUNIT_ASSERT((cpu_buffer_of_gpu_mempool[0][0] == 1 || cpu_buffer_of_gpu_mempool[0][0] == 2) && "id of neither GPU0 nor GPU1 in GPU0 mempool");

    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][1] == 3 && "CPU id not in GPU1 mempool");
    CPPUNIT_ASSERT((cpu_buffer_of_gpu_mempool[1][0] == 1 || cpu_buffer_of_gpu_mempool[1][0] == 2) && "id of neither GPU0 nor GPU1 in GPU1 mempool");

    // TODO: apply the dataset in DRAM
    printf("CPU is done with the merging\n");
}

void TestBMAP::TestGPUsConflictNotFirstChunk()
{
    granule_t *_cpu = mempool_base_addr_cpu[0];
    granule_t *_gpu0 = mempool_base_addr_gpu[0];
    granule_t *_gpu1 = mempool_base_addr_gpu[1];

    CPPUNIT_ASSERT(mempool_base_addr_cpu[0] == mempool_base_addr_cpu[1]); // CPU keeps only 1 memory pool


    PR_global[0].PR_blockNum  = 1;
    PR_global[0].PR_threadNum = 1;
    PR_global[1].PR_blockNum  = 1;
    PR_global[1].PR_threadNum = 1;

    argsDev0->rd_addrs = rd_addr_gpu0;
    argsDev0->wt_addrs = wt_addr_gpu0;
    argsDev1->rd_addrs = rd_addr_gpu1;
    argsDev1->wt_addrs = wt_addr_gpu1;

    rd_addr_cpu[0]  = _cpu+1; // writes also read
    rd_addr_gpu0[0] = _gpu0+BMAP_GRAN+10;
    rd_addr_gpu1[0] = _gpu1+BMAP_GRAN+10;

    wt_addr_cpu[0]  = _cpu+1;
    wt_addr_gpu0[0] = _gpu0+BMAP_GRAN+10;
    wt_addr_gpu1[0] = _gpu1+BMAP_GRAN+10;

    cudaMemcpy(argsDev0->rd_addrs, rd_addr_gpu0, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(argsDev0->wt_addrs, wt_addr_gpu0, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(argsDev1->rd_addrs, rd_addr_gpu1, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(argsDev1->wt_addrs, wt_addr_gpu1, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);

    HeTM_gshared_data.batchCount = 19;

    PR_curr_dev = 0;
    argsDev0->nbPos = size_rwset;
    argsDev0->base = _gpu0;
    argsDev0->devId = 0;
    launchKernel(/*devId*/0);

    PR_curr_dev = 1;
    argsDev1->nbPos = size_rwset;
    argsDev1->base = _gpu1;
    argsDev1->devId = 1;
    launchKernel(/*devId*/1);

    SetRdPositionInCPU(size_rwset, rd_addr_cpu);
    SetWrtPositionInCPU(size_rwset, wt_addr_cpu);

    Config::GetInstance()->SelDev(0);
    cudaDeviceSynchronize();

    Config::GetInstance()->SelDev(1);
    cudaDeviceSynchronize(); // when emulating the second device this is redundant

    // do the comparison among GPUs
    notifyBatchIsDone(); // GPU controller

    // inspect write set of GPU0
    Config::GetInstance()->SelDev(0);
    MemObj *m_gpu0_wrtset = HeTM_gpu_wset.GetMemObj(0);
    m_gpu0_wrtset->CpyDtH();
    unsigned char *gpu0_wrtset = (unsigned char *)m_gpu0_wrtset->host;
    cudaDeviceSynchronize();
    printf("[GPU0] wrote in pos %i = %i\n", BMAP_GRAN+10, gpu0_wrtset[BMAP_GRAN+10]);

    // inspect write set of GPU1
    Config::GetInstance()->SelDev(1);
    MemObj *m_gpu1_wrtset = HeTM_gpu_wset.GetMemObj(1);
    m_gpu1_wrtset->CpyDtH();
    unsigned char *gpu1_wrtset = (unsigned char *)m_gpu1_wrtset->host;
    cudaDeviceSynchronize();
    printf("[GPU1] wrote in pos %i = %i\n", BMAP_GRAN+10, gpu1_wrtset[BMAP_GRAN+10]);

    // do the comparison with the CPU
    cpyCPUwrtsetToGPU(1);
    // sleep(2);

    // syncGPUdataset((void*)1); // contains mergeMatricesAndRunFVS();
    mergeMatricesAndRunFVS(1);

    waitNextBatch(1);
    doGPUStateReset();

    // TODO: what is the result if all conflict? who wins?
    CPPUNIT_ASSERT(mempool_base_addr_cpu[0][1] == 3 && "CPU id not in CPU mempool");
    CPPUNIT_ASSERT((mempool_base_addr_cpu[0][BMAP_GRAN+10] == 1 || mempool_base_addr_cpu[0][BMAP_GRAN+10] == 2) && "id of neither GPU0 nor GPU1 in CPU mempool");

    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[0], mempool_base_addr_gpu[0], sizeof(granule_t)*MEMPOOL_SIZE);
    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[1], mempool_base_addr_gpu[1], sizeof(granule_t)*MEMPOOL_SIZE);

    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][1] == 3 && "CPU id not in GPU0 mempool");
    CPPUNIT_ASSERT((cpu_buffer_of_gpu_mempool[0][BMAP_GRAN+10] == 1 || cpu_buffer_of_gpu_mempool[0][BMAP_GRAN+10] == 2) && "id of neither GPU0 nor GPU1 in GPU0 mempool");

    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][1] == 3 && "CPU id not in GPU1 mempool");
    CPPUNIT_ASSERT((cpu_buffer_of_gpu_mempool[1][BMAP_GRAN+10] == 1 || cpu_buffer_of_gpu_mempool[1][BMAP_GRAN+10] == 2) && "id of neither GPU0 nor GPU1 in GPU1 mempool");

    // TODO: apply the dataset in DRAM
    printf("CPU is done with the merging\n");
}

void TestBMAP::TestDisjointChunkWrites()
{
    granule_t *_cpu = mempool_base_addr_cpu[0];
    granule_t *_gpu0 = mempool_base_addr_gpu[0];
    granule_t *_gpu1 = mempool_base_addr_gpu[1];

    CPPUNIT_ASSERT(mempool_base_addr_cpu[0] == mempool_base_addr_cpu[1]); // CPU keeps only 1 memory pool

    PR_global[0].PR_blockNum  = 1;
    PR_global[0].PR_threadNum = 1;
    PR_global[1].PR_blockNum  = 1;
    PR_global[1].PR_threadNum = 1;

    argsDev0->rd_addrs = rd_addr_gpu0;
    argsDev0->wt_addrs = wt_addr_gpu0;
    argsDev1->rd_addrs = rd_addr_gpu1;
    argsDev1->wt_addrs = wt_addr_gpu1;

    rd_addr_cpu[0]  = _cpu  + 1; // writes also read
    rd_addr_gpu0[0] = _gpu0 + 1 + BMAP_GRAN;
    rd_addr_gpu1[0] = _gpu1 + 1 + (2*BMAP_GRAN);

    wt_addr_cpu[0]  = _cpu  + 1;
    wt_addr_gpu0[0] = _gpu0 + 1 + BMAP_GRAN;
    wt_addr_gpu1[0] = _gpu1 + 1 + (2*BMAP_GRAN);

    cudaMemcpy(argsDev0->rd_addrs, rd_addr_gpu0, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(argsDev0->wt_addrs, wt_addr_gpu0, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(argsDev1->rd_addrs, rd_addr_gpu1, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(argsDev1->wt_addrs, wt_addr_gpu1, size_rwset*sizeof(granule_t*), cudaMemcpyHostToDevice);

    HeTM_gshared_data.batchCount = 17;

    PR_curr_dev = 0;
    argsDev0->nbPos = size_rwset;
    argsDev0->base = _gpu0;
    argsDev0->devId = 0;
    launchKernel(/*devId*/0);

    PR_curr_dev = 1;
    argsDev1->nbPos = size_rwset;
    argsDev1->base = _gpu1;
    argsDev1->devId = 1;
    launchKernel(/*devId*/1);

    SetRdPositionInCPU(size_rwset, rd_addr_cpu);
    SetWrtPositionInCPU(size_rwset, wt_addr_cpu);

    Config::GetInstance()->SelDev(0);
    cudaDeviceSynchronize();

    Config::GetInstance()->SelDev(1);
    cudaDeviceSynchronize(); // when emulating the second device this is redundant

    // do the comparison among GPUs
    notifyBatchIsDone(); // GPU controller

    // inspect write set of GPU0
    Config::GetInstance()->SelDev(0);
    MemObj *m_gpu0_wrtset = HeTM_gpu_wset.GetMemObj(0);
    m_gpu0_wrtset->CpyDtH();
    unsigned char *gpu0_wrtset = (unsigned char *)m_gpu0_wrtset->host;
    cudaDeviceSynchronize();
    printf("[GPU0] wrote in pos %i = %i\n", 2, gpu0_wrtset[2]);
    printf("[GPU0] wrote in pos %i = %i\n", 4, gpu0_wrtset[4]);

    // inspect write set of GPU1
    Config::GetInstance()->SelDev(1);
    MemObj *m_gpu1_wrtset = HeTM_gpu_wset.GetMemObj(1);
    m_gpu1_wrtset->CpyDtH();
    unsigned char *gpu1_wrtset = (unsigned char *)m_gpu1_wrtset->host;
    cudaDeviceSynchronize();
    printf("[GPU1] wrote in pos %i = %i\n", 2, gpu1_wrtset[2]);
    printf("[GPU1] wrote in pos %i = %i\n", 4, gpu1_wrtset[4]);

    // do the comparison with the CPU
    cpyCPUwrtsetToGPU(1);
    // sleep(2);

    // syncGPUdataset((void*)1); // contains mergeMatricesAndRunFVS();
    mergeMatricesAndRunFVS(1);

    waitNextBatch(1);
    doGPUStateReset();

    Config::GetInstance()->SelDev(0);
    cudaDeviceSynchronize();

    // inspect write set of GPU1
    Config::GetInstance()->SelDev(1);
    cudaDeviceSynchronize();

     printf("BMAP_GRAN = %i BMAP_GRAN_bytes = %i\n", BMAP_GRAN, BMAP_GRAN*4);

    // TODO: what is the result if all conflict? who wins?
    CPPUNIT_ASSERT(mempool_base_addr_cpu[0][1] == 3 && "CPU id not in CPU mempool");
    CPPUNIT_ASSERT(mempool_base_addr_cpu[0][1+BMAP_GRAN] == 1 && "GPU0 id not in CPU mempool");
    CPPUNIT_ASSERT(mempool_base_addr_cpu[0][1+(2*BMAP_GRAN)] == 2 && "GPU1 id not in CPU mempool");

    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[0], mempool_base_addr_gpu[0], sizeof(granule_t)*MEMPOOL_SIZE);
    CUDA_CPY_TO_HOST(cpu_buffer_of_gpu_mempool[1], mempool_base_addr_gpu[1], sizeof(granule_t)*MEMPOOL_SIZE);

    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][1] == 3 && "CPU id not in GPU0 mempool");
    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][1+BMAP_GRAN] == 1 && "GPU0 id not in GPU0 mempool");
    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[0][1+(2*BMAP_GRAN)] == 2 && "GPU1 id not in GPU0 mempool");

    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][1] == 3 && "CPU id not in GPU1 mempool");
    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][1+BMAP_GRAN] == 1 && "GPU0 id not in GPU1 mempool");
    CPPUNIT_ASSERT(cpu_buffer_of_gpu_mempool[1][1+(2*BMAP_GRAN)] == 2 && "GPU1 id not in GPU1 mempool");

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

        SET_ON_RS_BMAP(addrs[i], GPU_log); // writes also read

        // uintptr_t _wsetAddr = (uintptr_t)(addrs[i]);
        // uintptr_t _devwBAddr = (uintptr_t)GPU_log->devMemPoolBasePtr;
        // uintptr_t _wpos = (_wsetAddr - _devwBAddr) >> (PR_LOCK_GRAN_BITS+HETM_REDUCED_RS);
        // void *_WSetBitmap = GPU_log->bmap_wset_devptr;
        // unsigned char *BM_bitmap = (unsigned char*)_WSetBitmap;
        // BM_bitmap[_wpos] = (unsigned char)GPU_log->batchCount;
        SET_ON_WS_BMAP(addrs[i], GPU_log); // above: expansion of this macro
        printf("[dev%i] --- Set write %lu\n", GPU_log->devId,
            ((uintptr_t)(addrs[i]) - (uintptr_t)GPU_log->devMemPoolBasePtr) >> (PR_LOCK_GRAN_BITS+HETM_REDUCED_RS));
        *(addrs[i]) = GPU_log->devId+1;
    }
}

__device__ void SetRdPositionInGPU(PR_txCallDefArgs, size_t nbPos, granule_t **addrs)
{
    HeTM_GPU_log_s *GPU_log = (HeTM_GPU_log_s*)pr_args.pr_args_ext;
    for (int i = 0; i < nbPos; ++i)
    {
        // memman_access_addr_dev(GPU_log->bmap, args->wset.addrs[i], GPU_log->batchCount);
        // SET_ON_RS_BMAP(pr_args.wset.addrs[i]);
        SET_ON_RS_BMAP(addrs[i], GPU_log);
        printf("[dev%i] --- Set read %lu\n", GPU_log->devId,
            ((uintptr_t)(addrs[i]) - (uintptr_t)GPU_log->devMemPoolBasePtr) >> (PR_LOCK_GRAN_BITS+HETM_REDUCED_RS));
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
    // cudaStream_t stream = (cudaStream_t)params.stream;
    pr_buffer_s inBuf, outBuf;

    pr_tx_args_s *pr_args = getPrSTMmetaData(params.devId);
    //   params.currDev 

    inBuf.buf = params.entryObj->dev;
    inBuf.size = params.entryObj->size;
    outBuf.buf = NULL;
    outBuf.size = 0;
    PR_prepareIO(pr_args, inBuf, outBuf);
    PR_run(kernel_setPositions, pr_args);
}

