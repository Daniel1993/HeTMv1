#include "all_bank_parts.cuh"

using namespace memman;

void bank_part3_init()
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

		GPU_input_file = fopen(parsedData.GPUInputFile, "r");
		printf("Opening %s\n", parsedData.GPUInputFile);

		for (int i = 0; i < buffer_last; ++i)
		{
			// int access = GPU_ACCESS(rnd, (parsedData.nb_accounts-BANK_NB_TRANSFERS-1));
			if (!fscanf(GPU_input_file, "%i\n", &rnd))
				printf("Error reading from file\n");
			
			cpu_ptr[i] = rnd % parsedData.nb_accounts;
			cpu_ptr[i] &= (unsigned)-2;
		}

		// ----------- High conflict buffers
		cpu_ptr = (int*)GPU_input_buffer_bad.GetMemObj(j)->host;

		// cpu_ptr[0] = 0; // deterministic abort
		for (int i = 0; i < buffer_last; ++i)
		{
			if (!fscanf(GPU_input_file, "%i\n", &rnd))
				printf("Error reading from file\n");
			
			cpu_ptr[i] = rnd % parsedData.nb_accounts;
		}

		// --------------------
		// CPU buffers
		// -----------
		int good_buffers_last = size_of_CPU_input_buffer/sizeof(int) * NB_OF_BUFFERS;
		int bad_buffers_last = 2*size_of_CPU_input_buffer/sizeof(int) * NB_OF_BUFFERS;

		CPU_input_file = fopen(parsedData.CPUInputFile, "r");
		printf("Opening %s\n", parsedData.CPUInputFile);

		for (int i = 0; i < good_buffers_last; ++i)
		{
			// unsigned rnd = (*zipf_dist)(generator); // RAND_R_FNC(input_seed);
			if (fscanf(CPU_input_file, "%i\n", &rnd) == EOF)
				printf("Error reading from file\n");
			
			CPUInputBuffer[i] = rnd % (parsedData.nb_accounts - 20);
			CPUInputBuffer[i] |= 1;
		}
		// CPUInputBuffer[good_buffers_last] = 0; // deterministic abort
		for (int i = good_buffers_last; i < bad_buffers_last; ++i)
		{
			// unsigned rnd = (*zipf_dist)(generator); // RAND_R_FNC(input_seed);
			if (fscanf(CPU_input_file, "%i\n", &rnd) == EOF)
				printf("Error reading from file\n");
			
			CPUInputBuffer[i] = rnd % (parsedData.nb_accounts - 20);
			// CPUInputBuffer[i] |= 1;
		}
	}
}

void bank_part3_cpu_run(int id, thread_data_t *d)
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

	// if (accounts_vec[0] == 0) {
	// 	printf("conflict access stm_baseMemPool=%p accounts=%p\n", stm_baseMemPool, accounts);
	// }

	static thread_local unsigned offset = 4096 * (id*1000);
	offset += (4096 % nb_accounts) & ~0xFFF;
	accounts_vec[0] = accounts_vec[0] % 4096;
	accounts_vec[0] += offset;
	accounts_vec[0] %= (nb_accounts - 40);
	accounts_vec[0] |= 1;

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

__device__ void bank_part3_gpu_run(int tid, PR_txCallDefArgs)
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
