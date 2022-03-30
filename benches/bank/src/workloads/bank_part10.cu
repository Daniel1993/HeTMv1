#include "all_bank_parts.cuh"

using namespace memman;

void bank_part10_init()
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

		// int nbPages = parsedData.nb_accounts / PAGE_SIZE;
		// if (parsedData.nb_accounts % PAGE_SIZE) nbPages++;

		// TODO: I should use more input buffers: some caching effects may show up
		rnd = RAND_R_FNC(input_seed);
		for (int i = 0; i < buffer_last; ++i) {
			unsigned cnfl_rnd = RAND_R_FNC(input_seed);
			unsigned pos = RAND_R_FNC(input_seed);
#if BANK_INTRA_CONFL > 0
			// no conflict
			if (cnfl_rnd % 100 >= BANK_INTRA_CONFL*100)
				cpu_ptr[i] = GPU_ACCESS(j, pos, parsedData.nb_accounts-20);
			else
#endif
				cpu_ptr[i] = GPU_ACCESS(j, pos % 128, parsedData.nb_accounts-20);
		}

		// ----------- High conflict buffers
		cpu_ptr = (int*)GPU_input_buffer_bad.GetMemObj(j)->host;

		rnd = RAND_R_FNC(input_seed);
		for (int i = 0; i < buffer_last; ++i) {
			if (i % 512 == 0) {
				cpu_ptr[i] = 0; // deterministic intersection
				continue;
			}
			unsigned cnfl_rnd = RAND_R_FNC(input_seed);
			unsigned pos = RAND_R_FNC(input_seed);
#if BANK_INTRA_CONFL > 0
			// no conflict
			if (cnfl_rnd % 100 >= BANK_INTRA_CONFL*100)
				cpu_ptr[i] = GPU_ACCESS(j, pos, parsedData.nb_accounts-20);
			else
#endif
				cpu_ptr[i] = GPU_ACCESS(j, pos % 128, parsedData.nb_accounts-20);
		}
	}


	// --------------------
	// CPU buffers
	// -----------
	int good_buffers_last = size_of_CPU_input_buffer/sizeof(int) * NB_OF_BUFFERS;
	int bad_buffers_last = 2*size_of_CPU_input_buffer/sizeof(int) * NB_OF_BUFFERS;

	// TODO: make the CPU access the end --> target_page * PAGE_SIZE - access

	// int nbPages = parsedData.nb_accounts / PAGE_SIZE;
	// if (parsedData.nb_accounts % PAGE_SIZE) nbPages++;
	// TODO: if the last page is incomplete it is never accessed

	unsigned long reset_rnd = RAND_R_FNC(input_seed);
	rnd = reset_rnd;
	for (int i = 0; i < good_buffers_last; ++i) {
		unsigned cnfl_rnd = RAND_R_FNC(input_seed);
		unsigned pos = RAND_R_FNC(input_seed);
#if BANK_INTRA_CONFL > 0
		if (cnfl_rnd % 100 >= BANK_INTRA_CONFL*100)
			// no conflict
			CPUInputBuffer[i] = CPU_ACCESS(pos, parsedData.nb_accounts-20);
		else
#endif
			CPUInputBuffer[i] = CPU_ACCESS(pos % 128, parsedData.nb_accounts-20);
	}

	reset_rnd = RAND_R_FNC(input_seed);
	rnd = reset_rnd;
	for (int i = good_buffers_last; i < bad_buffers_last; ++i) {
		if (i % 128 == 0) {
			CPUInputBuffer[i] = 0; // deterministic intersection
			continue;
		}
		unsigned cnfl_rnd = RAND_R_FNC(input_seed);
		unsigned pos = RAND_R_FNC(input_seed);
#if BANK_INTRA_CONFL > 0
		if (cnfl_rnd % 100 >= BANK_INTRA_CONFL*100)
			// no conflict
			CPUInputBuffer[i] = CPU_ACCESS(pos, parsedData.nb_accounts-20);
		else
#endif
			CPUInputBuffer[i] = CPU_ACCESS(pos % 128, parsedData.nb_accounts-20);
	}
}

void bank_part10_cpu_run(int id, thread_data_t *d)
{
	account_t *accounts = d->bank->accounts;
	volatile unsigned accounts_vec[d->read_intensive_size+2];
	int nb_accounts = d->bank->size;
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
		if (rndOpt2 % 100000000 < (d->prec_write_txs * 1000000)) {
			transfer2(accounts, accounts_vec, isInterBatch, d->read_intensive_size, id, nb_accounts);
		} else {
			// resReadOnly += transferReadOnly(accounts, accounts_vec, d->trfs, 1);
			resReadOnly += readOnly2(accounts, accounts_vec, isInterBatch, d->read_intensive_size, id, nb_accounts);
		}
	}
}

__device__ void bank_part10_gpu_run(int tid, PR_txCallDefArgs)
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
			// TODO:
			if (option2 % 100000000 < (devParsedData.prec_write_txs * 1000000)) { // prec read-only
				readIntensive_tx(PR_txCallArgs, i);
			} else {
				readOnly_tx(PR_txCallArgs, i);
			}
		} else {
			// if (option2 % 100000000 < (devParsedData.prec_write_txs * 1000000)) {
				update_tx2(PR_txCallArgs, i);
			// } else {
			// 	readOnly_tx2(PR_txCallArgs, i);
			// }
		}
	}
}

