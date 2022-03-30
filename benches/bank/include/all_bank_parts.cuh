#ifndef ALL_BANK_PARTS_H_GUARD_
#define ALL_BANK_PARTS_H_GUARD_

#include "hetm.cuh"
#include "bank.hpp"
#include "pr-stm-wrapper.cuh"
#include "memman.hpp"

/**
 * --- WORKLOADS ---
 * BANK_PART == 1 --> split the dataset with GPU_PART/CPU_PART
 * BANK_PART == 2 --> Uniform at random inter-leaved
 * BANK_PART == 3 --> Zipf interleaved
 * BANK_PART == 4 --> Zipf separate (else)
 */


#define INTEREST_RATE 0.5

#define COMPUTE_TRANSFER(val) \
	val // TODO: do math that does not kill the final result

/* ################################################################### *
 * GLOBALS
 * ################################################################### */

const size_t NB_OF_BUFFERS = 512; // 2 good + 2 bad

const static int NB_CPU_TXS_PER_THREAD = 16384;
const static int PAGE_SIZE = 4096; // TODO: this is also defined in hetm proj (CACHE_GRANULE_SIZE)

extern thread_data_t parsedData;
extern int isInterBatch;
extern unsigned long long input_seed;
extern int bank_cpu_sample_data;
extern size_t currMaxCPUoutputBufferSize;
extern size_t maxGPUoutputBufferSize;
extern size_t size_of_GPU_input_buffer;
extern size_t size_of_CPU_input_buffer;
extern FILE *GPU_input_file;
extern FILE *CPU_input_file;

// extern memman_s GPU_input_buffer_good;
// extern memman_s GPU_input_buffer_bad;
// extern memman_s GPU_input_buffer;
// extern memman_s GPU_output_buffer;
extern memman::MemObjOnDev GPU_input_buffer_good;
extern memman::MemObjOnDev GPU_input_buffer_bad;
extern memman::MemObjOnDev GPU_input_buffer;
extern memman::MemObjOnDev GPU_output_buffer;

extern int *GPUoutputBuffer[HETM_NB_DEVICES];
extern int *CPUoutputBuffer;
extern int *GPUInputBuffer[HETM_NB_DEVICES];
extern int *CPUInputBuffer;

__constant__ __device__ extern long  BANK_SIZE;
// __constant__ __device__ const int BANK_NB_TRANSFERS;
__constant__ __device__ extern int   HASH_NUM;
__constant__ __device__ extern int   num_ways;
__constant__ __device__ extern int   num_sets;
__constant__ __device__ extern int   txsPerGPUThread;
__constant__ __device__ extern int   txsInKernel;
__constant__ __device__ extern int   hprob;
__constant__ __device__ extern int   prec_read_intensive;
__constant__ __device__ extern float hmult;
__constant__ __device__ extern int   read_intensive_size;

__constant__ __device__ extern thread_data_t devParsedData;
__constant__ __device__ extern int PR_maxNbRetries;

void bank_part1_init();
void bank_part2_init();
void bank_part3_init();
void bank_part4_init();
void bank_part5_init();
void bank_part6_init();
void bank_part7_init();
void bank_part8_init();
void bank_part9_init();
void bank_part10_init();

void bank_part1_cpu_run(int id, thread_data_t *d);
void bank_part2_cpu_run(int id, thread_data_t *d);
void bank_part3_cpu_run(int id, thread_data_t *d);
void bank_part4_cpu_run(int id, thread_data_t *d);
void bank_part5_cpu_run(int id, thread_data_t *d);
void bank_part6_cpu_run(int id, thread_data_t *d);
void bank_part7_cpu_run(int id, thread_data_t *d);
void bank_part8_cpu_run(int id, thread_data_t *d);
void bank_part9_cpu_run(int id, thread_data_t *d);
void bank_part10_cpu_run(int id, thread_data_t *d);

__device__ void bank_part1_gpu_run(int id, PR_txCallDefArgs);
__device__ void bank_part2_gpu_run(int id, PR_txCallDefArgs);
__device__ void bank_part3_gpu_run(int id, PR_txCallDefArgs);
__device__ void bank_part4_gpu_run(int id, PR_txCallDefArgs);
__device__ void bank_part5_gpu_run(int id, PR_txCallDefArgs);
__device__ void bank_part6_gpu_run(int id, PR_txCallDefArgs);
__device__ void bank_part7_gpu_run(int id, PR_txCallDefArgs);
__device__ void bank_part8_gpu_run(int id, PR_txCallDefArgs);
__device__ void bank_part9_gpu_run(int id, PR_txCallDefArgs);
__device__ void bank_part10_gpu_run(int id, PR_txCallDefArgs);

__device__ void readIntensive_tx(PR_txCallDefArgs, int txCount);
__device__ void readOnly_tx(PR_txCallDefArgs, int txCount);
__device__ void update_tx2(PR_txCallDefArgs, int txCount);
__device__ void readOnly_tx2(PR_txCallDefArgs, int txCount);
__device__ void update_tx(PR_txCallDefArgs, int txCount);
__device__ void update_tx_simple(PR_txCallDefArgs, int txCount);
__device__ void updateReadOnly_tx(PR_txCallDefArgs, int txCount);

// ----------------------------------------
// cpu_txs
// -------

int transfer(account_t *accounts, volatile unsigned *positions, int count, int amount);
int transfer_simple(account_t *accounts, volatile unsigned *positions, int count, int amount);
int readIntensive(account_t *accounts, volatile unsigned *positions, int count, int amount);
int readOnly(account_t *accounts, volatile unsigned *positions, int count, int amount);
int readOnly2(account_t *accounts, volatile unsigned *positions, int isInter, int count, int tid, int nbAccounts);

int transfer2(account_t *accounts, volatile unsigned *positions, int isInter, int count, int tid, int nbAccounts);
int readOnly2(account_t *accounts, volatile unsigned *positions, int isInter, int count, int tid, int nbAccounts);

int total(bank_t *bank, int transactional);
void reset(bank_t *bank);

#endif /* ALL_BANK_PARTS_H_GUARD_ */