#include "all_bank_parts.cuh"

using namespace memman;

void bank_part7_init()
{
	volatile int buffer_last = size_of_GPU_input_buffer/sizeof(int) * NB_OF_BUFFERS;
	volatile unsigned rnd = 0;
	int nbGPUs = Config::GetInstance()->NbGPUs();
	int *cpu_ptr;

	for (int j = 0; j < nbGPUs; ++j)
	{
		cpu_ptr = (int*)GPU_input_buffer_good.GetMemObj(j)->host;

		rnd = RAND_R_FNC(input_seed);
		// ----------- Low conflict buffers
		for (int i = 0; i < buffer_last; ++i) {
			cpu_ptr[i] = GPU_ACCESS(j, rnd, parsedData.nb_accounts-20);
			rnd = RAND_R_FNC(input_seed);
		}

		cpu_ptr = (int*)GPU_input_buffer_bad.GetMemObj(j)->host;

		rnd = RAND_R_FNC(input_seed);
		// ----------- High conflict buffers
		for (int i = 0; i < buffer_last; ++i) {
			if (i % 128 == 0) {
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
	int good_buffers_last = size_of_CPU_input_buffer/sizeof(int) * NB_OF_BUFFERS;
	int bad_buffers_last = 2*size_of_CPU_input_buffer/sizeof(int) * NB_OF_BUFFERS;

	rnd = 0;
	for (int i = 0; i < good_buffers_last; ++i) {
		CPUInputBuffer[i] = CPU_ACCESS(rnd, parsedData.nb_accounts-20);
		rnd += parsedData.read_intensive_size + 16;
	}

	rnd = 0;
	for (int i = good_buffers_last; i < bad_buffers_last; ++i) {
		if (i % 128 == 0) {
			CPUInputBuffer[i] = 0; // deterministic intersection
			continue;
		}
		CPUInputBuffer[i] = INTERSECT_ACCESS_CPU(i, parsedData.nb_accounts-20);
		rnd += parsedData.read_intensive_size + 16;
	}
}

void bank_part7_cpu_run(int id, thread_data_t *d)
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
		/* ** READ_INTENSIVE BRANCH (-R > 0) ** */
		// BANK_PREPARE_READ_INTENSIVE(d->id, d->seed, rnd, d->hmult, d->nb_threads, accounts_vec, nb_accounts);
		// TODO: change the buffers
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
		/* ** READ_INTENSIVE BRANCH (current in use with -R 0) ** */
		// BANK_PREPARE_TRANSFER(d->id, d->seed, rnd, d->hmult, d->nb_threads, accounts_vec, nb_accounts);
		for (int i = 1; i < d->read_intensive_size+1; ++i) {
			accounts_vec[i] = accounts_vec[i-1]+1;
		}
		// for (int i = 1; i < d->trfs*2; ++i)
		// 	accounts_vec[i] = accounts_vec[i-1] + parsedData.access_offset;
		if (rndOpt2 % 100000000 < (d->prec_write_txs * 1000000)) {
			transfer(accounts, accounts_vec, d->read_intensive_size, 1);
		} else {
			// resReadOnly += transferReadOnly(accounts, accounts_vec, d->trfs, 1);
			resReadOnly += readOnly(accounts, accounts_vec, d->read_intensive_size, 1);
		}
	}
}

__device__ void bank_part7_gpu_run(int tid, PR_txCallDefArgs)
{
	return;
}
