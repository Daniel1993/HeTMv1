#include "all_bank_parts.cuh"
#include <assert.h>

#include "bankKernel.cuh" // test only

using namespace memman;

void bank_part1_init()
{
	volatile size_t buffer_last = size_of_GPU_input_buffer/sizeof(int) * NB_OF_BUFFERS;
	volatile unsigned rnd = 0;
	size_t good_buffers_last = size_of_CPU_input_buffer/sizeof(int) * NB_OF_BUFFERS;
	size_t bad_buffers_last = 2*size_of_CPU_input_buffer/sizeof(int) * NB_OF_BUFFERS;
	int *cpu_ptr;
	int nbGPUs = Config::GetInstance()->NbGPUs();

	input_seed = 123423;
	// printf (" \n ----- input_seed = %lu ----- \n ", input_seed);
	
	// --------------------
	// GPU buffers
	// -----------

	for (int j = 0; j < nbGPUs; ++j) 
	{
		cpu_ptr = (int*)GPU_input_buffer_good.GetMemObj(j)->host; //GPU_input_buffer_good.getCPUPtr();

		rnd = RAND_R_FNC(input_seed);
		// ----------- Low conflict buffers
		for (int i = 0; i < buffer_last; ++i)
		{
			cpu_ptr[i] = GPU_ACCESS(j, rnd, parsedData.nb_accounts-20);
			rnd = RAND_R_FNC(input_seed);
		}

		cpu_ptr = (int*)GPU_input_buffer_bad.GetMemObj(j)->host;

		rnd = RAND_R_FNC(input_seed);
		// ----------- High conflict buffers
		for (int i = 0; i < buffer_last; ++i) {
			if (i % 512 == 0) {
				cpu_ptr[i] = 0; // deterministic intersection
				continue;
			}
			cpu_ptr[i] = INTERSECT_ACCESS_GPU(j, rnd, parsedData.nb_accounts-20);
			rnd = RAND_R_FNC(input_seed);
		}
	}

	// --------------------
	// CPU buffers
	// -----------
	rnd = RAND_R_FNC(input_seed);
	for (int i = 0; i < good_buffers_last; ++i)
	{
		CPUInputBuffer[i] = CPU_ACCESS(rnd, parsedData.nb_accounts-20);
		rnd = RAND_R_FNC(input_seed);
	}

	rnd = RAND_R_FNC(input_seed);
	for (int i = good_buffers_last; i < bad_buffers_last; ++i)
	{
		if (i % 128 == 0) {
			CPUInputBuffer[i] = 0; // deterministic intersection
			continue;
		}
		CPUInputBuffer[i] = INTERSECT_ACCESS_CPU(rnd, parsedData.nb_accounts-20);
		rnd = RAND_R_FNC(input_seed);
	}
}

void bank_part1_cpu_run(int id, thread_data_t *d)
{
	volatile unsigned accounts_vec[d->read_intensive_size+2];
	account_t *accounts = d->bank->accounts;
	// int nb_accounts = d->bank->size;
	int rndOpt = RAND_R_FNC(d->seed);
	int rndOpt2 = RAND_R_FNC(d->seed);
	static thread_local int curr_tx = 0;
	static thread_local volatile int resReadOnly = 0;

	int good_buffers_start = 0;
	int bad_buffers_start = size_of_CPU_input_buffer/sizeof(int) * NB_OF_BUFFERS;
	int buffers_start = isInterBatch ? bad_buffers_start : good_buffers_start;

	int index = buffers_start+id*NB_CPU_TXS_PER_THREAD + curr_tx;
	accounts_vec[0] = CPUInputBuffer[index];
	curr_tx += 1;
	curr_tx = curr_tx % NB_CPU_TXS_PER_THREAD;
	// int index1 = buffers_start+id*NB_CPU_TXS_PER_THREAD + curr_tx;
	
	assert(0 < accounts_vec[0] && parsedData.nb_accounts > accounts_vec[0] && "invalid account idx");

	if (rndOpt % 100 < d->nb_read_intensive) {
		for (int i = 1; i < d->read_intensive_size+1; ++i) {
			accounts_vec[i] = accounts_vec[i-1]+1;
		}
		if (rndOpt2 % 100000000 < (d->prec_write_txs * 1000000)) {
			// 1% readIntensive
			readIntensive(accounts, accounts_vec, d->read_intensive_size, 1);
		} else {
			resReadOnly += readOnly(accounts, accounts_vec, d->read_intensive_size, 1);
		}
	} else {
		for (int i = 1; i < d->read_intensive_size+1; ++i) {
			accounts_vec[i] = accounts_vec[i-1]+1;
			assert(0 < accounts_vec[i] && parsedData.nb_accounts > accounts_vec[i] && "invalid account idx");
		}
		if (rndOpt2 % 100000000 < (d->prec_write_txs * 1000000)) {
			transfer(accounts, accounts_vec, d->read_intensive_size, 1);
		} else {
			resReadOnly += readOnly(accounts, accounts_vec, d->read_intensive_size, 1);
		}
	}
}


__device__ void bank_part1_gpu_run(int tid, PR_txCallDefArgs)
{
	__shared__ int rndVoteOption[32];
	__shared__ int rndVoteOption2[32];

	int i = 0; //how many transactions one thread need to commit
	// HeTM_bankTx_input_s *input = (HeTM_bankTx_input_s*)args.inBuf;
	int option;
	int option2;

	if (threadIdx.x % 32 == 0) {
		rndVoteOption[threadIdx.x / 32] = PR_rand(INT_MAX);
		rndVoteOption2[threadIdx.x / 32] = PR_rand(INT_MAX);
	}
	__syncthreads();
	option = rndVoteOption[threadIdx.x / 32];
	option2 = rndVoteOption2[threadIdx.x / 32];
	// HeTM_GPU_log_s *GPU_log = (HeTM_GPU_log_s*)args.pr_args_ext;

	for (i = 0; i < txsPerGPUThread; ++i) { // each thread need to commit x transactions
		__syncthreads();
		if (option % 100 < prec_read_intensive) {
			if (option2 % 100000000 < (devParsedData.prec_write_txs * 1000000)) { // prec read-only
				readIntensive_tx(PR_txCallArgs, i);
			} else {
				readOnly_tx(PR_txCallArgs, i);
			}
		} else {
			if (option2 % 100000000 < (devParsedData.prec_write_txs * 1000000)) { // prec read-only
				update_tx(PR_txCallArgs, i);
			} else {
				readOnly_tx(PR_txCallArgs, i);
			}
		}
	}
}
