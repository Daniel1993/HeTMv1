#include "all_bank_parts.cuh"
#include "zipf_dist.h"

using namespace memman;

void bank_part4_init()
{
	volatile int buffer_last = size_of_GPU_input_buffer/sizeof(int) * NB_OF_BUFFERS;
	volatile unsigned rnd = 0;
	int nbGPUs = Config::GetInstance()->NbGPUs();
	int *cpu_ptr;

	for (int j = 0; j < nbGPUs; ++j) 
	{
		// --------------------
		// GPU buffers
		// -----------

		// ----------- Low conflict buffers
		cpu_ptr = (int*)GPU_input_buffer_good.GetMemObj(j)->host;

		unsigned maxGen = parsedData.nb_accounts;
		zipf_setup(maxGen, 0.8);
		for (int i = 0; i < buffer_last; ++i) {
			rnd = zipf_gen();
			cpu_ptr[i] = rnd;
		}

		// ----------- High conflict buffers
		cpu_ptr = (int*)GPU_input_buffer_bad.GetMemObj(j)->host;

		for (int i = 0; i < buffer_last; ++i)
		{
			rnd = zipf_gen();
			cpu_ptr[i] = rnd;
		}

		// --------------------
		// CPU buffers
		// -----------

		int good_buffers_last = size_of_CPU_input_buffer/sizeof(int) * NB_OF_BUFFERS;
		int bad_buffers_last = 2*size_of_CPU_input_buffer/sizeof(int) * NB_OF_BUFFERS;

		for (int i = 0; i < good_buffers_last; ++i)
		{
			rnd = zipf_gen();
			CPUInputBuffer[i] = parsedData.nb_accounts - rnd;
		}

		for (int i = good_buffers_last + 1; i < bad_buffers_last; ++i)
		{
			rnd = zipf_gen();
			CPUInputBuffer[i] = parsedData.nb_accounts - rnd;
		}
	}
}

void bank_part4_cpu_run(int id, thread_data_t *d)
{
	account_t *accounts = d->bank->accounts;
	volatile unsigned accounts_vec[d->read_intensive_size+2];
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
	int index1 = buffers_start+id*NB_CPU_TXS_PER_THREAD + curr_tx;
	accounts_vec[1] = CPUInputBuffer[index1];

	if (rndOpt % 100 < d->nb_read_intensive) {
		for (int i = 1; i < d->read_intensive_size+1; ++i) {
			accounts_vec[i] = accounts_vec[i-1]+1;
		}
		if (rndOpt2 % 100000000 < (d->prec_write_txs * 1000000)) {
			readIntensive(accounts, accounts_vec, d->read_intensive_size, 1);
		} else {
			resReadOnly += readOnly(accounts, accounts_vec, d->read_intensive_size, 1);
		}
	} else {
		for (int i = 1; i < d->read_intensive_size+1; ++i) {
			accounts_vec[i] = accounts_vec[i-1]+1;
		}
		if (rndOpt2 % 100000000 < (d->prec_write_txs * 1000000)) {
			transfer(accounts, accounts_vec, d->read_intensive_size, 1);
		} else {
			// resReadOnly += transferReadOnly(accounts, accounts_vec, d->trfs, 1);
			resReadOnly += readOnly(accounts, accounts_vec, d->read_intensive_size, 1);
		}
	}
}

__device__ void bank_part4_gpu_run(int tid, PR_txCallDefArgs)
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
				// updateReadOnly_tx(PR_txCallArgs, i);
			}
		}
	}
}
