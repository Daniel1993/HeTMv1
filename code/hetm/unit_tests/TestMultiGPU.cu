#include "TestMultiGPU.cuh"
#include "hetm-log.h"
#include "pr-stm.cuh"
#include "hetm.cuh"
#include "hetm-aux.cuh"
#include "stm.h"
#include "stm-wrapper.h"

CPPUNIT_TEST_SUITE_REGISTRATION(TestMultiGPU);

using namespace std;
using namespace memman;
using namespace knlman;

const int MAX_THREADS         = 1;
const int NB_CPU_THREADS      = 1;
const int NB_GPU_BLOCKS       = 1;
const int NB_GPU_THRS_PER_BLK = 1;

typedef struct callback_args_ {
	int *dataset;
	int *accesses;
	int nb_accesses;
} callback_args_t;

typedef struct gpu_callback_args_ {
	int *dataset;
	int *accesses;
	int nb_accesses;
} gpu_callback_args_t;

const int NB_GPUS = HETM_NB_DEVICES; // TODO: needs to be defined as a macro in the lib
const size_t DATA_SIZE = 4096 * 4 * sizeof(int);
int *data_cpu;
int *data_gpu;

void internal_runGPUBatch(int *accesses, int nb_accesses);
__global__ void gpu_tx_access(PR_globalKernelArgs);
void cpu_tx_access(int id, void *args);
static void choose_policy(int, void*) { /* TODO: drop policy */ }

void TestMultiGPU::setUp()
{
}

static void setupHeTM() 
{
	printf("[%s]\n", __PRETTY_FUNCTION__);
	printf("HeTM_shared_data[0].threadsInfo = %p\n", HeTM_shared_data[0].threadsInfo);
	HeTM_init((HeTM_init_s){
		.policy       = HETM_GPU_INV,
		.nbCPUThreads = NB_CPU_THREADS,
		.nbGPUBlocks  = NB_GPU_BLOCKS,
		.nbGPUThreads = NB_GPU_THRS_PER_BLK,
		.isCPUEnabled = 1,
		.isGPUEnabled = 0,
		.mempool_size = DATA_SIZE
	});
	printf("HeTM_shared_data[0].threadsInfo = %p\n", HeTM_shared_data[0].threadsInfo);
	HeTM_choose_policy(choose_policy);

	HeTM_run_sync = 1;
	HeTM_alloc(0, (void**)&data_cpu, (void**)&data_gpu, DATA_SIZE);
	memset(data_cpu, 0, DATA_SIZE);
	// TODO: exitGPU, exitCPU
}

void TestMultiGPU::tearDown() { }

static void teardownHeTM()
{
	printf("[%s]\n", __PRETTY_FUNCTION__);
	HeTM_set_is_stop(1);
	// HeTM_join_CPU_threads();
	HeTM_destroy(); // TODO: it will probably crash
}

static void gpu_init(int id)
{
	runBeforeGPU(id, (void*)HeTM_thread_data[0]);
	runGPUBeforeBatch(id, (void*)HeTM_thread_data[0]);
	doGPUStateReset();
	setCountAfterBatch();
}

static void gpu_done(int id)
{
	waitGPUBatchEnd();
	/*CPU*/	resetInterGPUConflFlag();
	notifyBatchIsDone();
	
	// TODO: this is from the CPU side (unit tests do not have parallelism)
	/*CPU*/	while (HeTM_shared_data[0].threadsWaitingSync < 1)
	/*CPU*/		pollIsRoundComplete(1);
	
	waitGPUCMPEnd(0);
	// TODO: removed barriers
	mergeGPUDataset();
	checkIsExit();
	RUN_ASYNC(syncGPUdataset, HeTM_thread_data[0]);
	RUN_ASYNC(waitGPUdataset, HeTM_thread_data[0]);

	runGPUAfterBatch(id, (void*)HeTM_thread_data[0]);
}

static void gpu_run(int id, void *argsPtr)
{
	callback_args_t *args = (callback_args_t*)argsPtr;
	gpu_init(id);
	// while-loop until timeout
	internal_runGPUBatch(args->accesses, args->nb_accesses); // does not block
	gpu_done(id);
}

static void cpu_init(int id)
{
	// Start CPU STM thread
	stm_init_thread();
	Config::GetInstance(HETM_NB_DEVICES);

	for (int j = 0; j < Config::GetInstance()->NbGPUs(); ++j)
	{
		Config::GetInstance()->SelDev(j);
		HeTM_thread_s *thr = &(HeTM_shared_data[j].threadsInfo[id]);
		HeTM_thread_data[j] = thr;

		thr->stream = NULL;// knlman_get_current_stream();
		thr->wSetLog = stm_thread_local_log;

		thr->logChunkEventCounter = 0;
		thr->logChunkEventStore = 0;

		stm_log_init();
	}
}

static void cpu_done(int id)
{
		// TODO: statistics
	// HeTM_thread_data[0]->curNbTxs++;
	// if (HeTM_get_GPU_status(0) == HETM_BATCH_DONE) {
    //   // transaction done while comparing
    //   HeTM_thread_data[0]->curNbTxsNonBlocking++;
    // }
}

static void cpu_run(int id, void *argsPtr)
{
	callback_args_t *args = (callback_args_t*)argsPtr;
	cpu_init(id);
	cpu_tx_access(id, argsPtr); // does 1 transaction
}

TestMultiGPU::TestMultiGPU()  { }
TestMultiGPU::~TestMultiGPU() { }

void TestMultiGPU::TestOnOff()
{
	// const int nb_accesses_cpu = 6;
	// const int nb_accesses_gpu = 6;
	// int accesses_cpu[nb_accesses_cpu] = { 1,  2,  3,  4,  5,  10};
	// int accesses_gpu[nb_accesses_gpu] = {10, 11, 12, 13, 14, 15};
	// callback_args_t args_cpu;
	// callback_args_t args_gpu;
	// args_cpu.accesses = accesses_cpu;
	// args_gpu.accesses = accesses_gpu;
	// args_cpu.nb_accesses = nb_accesses_cpu;
	// args_gpu.nb_accesses = nb_accesses_gpu;

	// setupHeTM();
	// printf("%s: init\n", __func__);

	// args_cpu.dataset = data_cpu;
	// args_gpu.dataset = data_gpu;

	// // test conflict
	// cpu_run(0, (void*)&args_cpu);
	// gpu_run(0, (void*)&args_gpu);

	// CPPUNIT_ASSERT(true);
	// teardownHeTM();
}


// ----------------------------------------
// Internal stuff

void internal_runGPUBatch(int *accesses, int nb_accesses)
{
	int *dataset = data_cpu;
	gpu_callback_args_t *args_send_to_gpu;

	cudaMallocManaged(&args_send_to_gpu, sizeof(gpu_callback_args_t));
	args_send_to_gpu->dataset = data_gpu;
	args_send_to_gpu->nb_accesses = nb_accesses;
	cudaMallocManaged(&(args_send_to_gpu->accesses), sizeof(gpu_callback_args_t));
	cudaMemcpy(args_send_to_gpu->accesses, accesses, nb_accesses*sizeof(int), cudaMemcpyDefault);

	runPrSTMCallback(
		NB_GPU_BLOCKS, NB_GPU_THRS_PER_BLK, gpu_tx_access,
		(void*)args_send_to_gpu, sizeof(gpu_callback_args_t),
		NULL, 0);
}

__global__ void gpu_tx_access(PR_globalKernelArgs)
{
	int id = PR_THREAD_IDX;
	gpu_callback_args_t *a = (gpu_callback_args_t*)args.inBuf;
	int *dataset = a->dataset;
	int *accesses = a->accesses;
	int nb_accesses = a->nb_accesses;
	int dummy = 0;

	PR_enterKernel(id);
	for (int i = 0; i < nb_accesses; ++i) {
		dummy += PR_read(&(dataset[accesses[i]]));
		PR_write(&(dataset[accesses[i]]), id);
	}
	PR_exitKernel();
}

void cpu_tx_access(int id, void *args)
{
	callback_args_t *a = (callback_args_t*)args;
	int *dataset = a->dataset;
	int *accesses = a->accesses;
	int nb_accesses = a->nb_accesses;
	int dummy = 0;

	TM_START(/*txid*/0, 0);
	for (int i = 0; i < nb_accesses; ++i) {
		dummy += TM_LOAD(&(dataset[accesses[i]]));
		TM_STORE (&(dataset[accesses[i]]), id);
	}
	TM_COMMIT;
}

