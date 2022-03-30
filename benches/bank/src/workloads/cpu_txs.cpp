#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define __USE_GNU

#include "bank.hpp"
#include <cmath>
#include "hetm-cmp-kernels.cuh"
#include "setupKernels.cuh"
#include "bank_aux.h"
#include "CheckAllFlags.h"
#include "input_handler.h"
#include "all_bank_parts.cuh"

#ifndef HETM_BANK_PART_SCALE
#define HETM_BANK_PART_SCALE 10
#endif /* HETM_BANK_PART_SCALE */

thread_local static unsigned someSeed = 0x034e3115;

int
transfer2(
	account_t *accounts,
	volatile unsigned *positions,
	int isInter,
	int count,
	int tid,
	int nbAccounts
) {
	int n;

#if BANK_PART == 10
	void *pos[count*HETM_BANK_PART_SCALE];
#else
	void *pos[count];
#endif
	// void *pos_write;
	int accountIdx;
	unsigned seedCopy = someSeed;

	double count_amount = 0.0;

	unsigned input = positions[0];
	unsigned randNum;

	unsigned seedState;

	seedCopy += tid ^ 1234;
	RAND_R_FNC(someSeed);

	seedCopy  = someSeed;
	seedState = someSeed;

	for (n = 0; n < count; n++) {
		randNum = RAND_R_FNC(seedCopy);
		if (!isInter) {
			accountIdx = CPU_ACCESS(randNum, nbAccounts);
		} else {
			accountIdx = randNum % nbAccounts;
		}
		pos[n] = accounts + accountIdx;
		// __builtin_prefetch(pos[n], 1, 1);
	}

  TM_START(/*txid*/0, RW);

	seedCopy = seedState;

	int forLoop = count;

#if BANK_PART == 10
	forLoop *= HETM_BANK_PART_SCALE;
#endif

	for (n = 0; n < forLoop; n++) {
		randNum = RAND_R_FNC(seedCopy);
		if (!isInter) {
			accountIdx = CPU_ACCESS(randNum, nbAccounts);
		} else {
#if BANK_PART == 9
			// deterministic abort
			accountIdx = /*(n == 0) ? tid * 64 : */INTERSECT_ACCESS_CPU(randNum, nbAccounts);
#else
			accountIdx = INTERSECT_ACCESS_CPU(randNum, nbAccounts);
#endif /* BANK_PART == 9 */
		}
		pos[n] = accounts + accountIdx;
		int someLoad = TM_LOAD(pos[n]);
		count_amount += someLoad;
		// int someLoad = TM_LOAD(pos[n]);
		// count_amount += someLoad;
  }

	seedCopy = seedState;

#if BANK_PART == 10
	forLoop = count;
#endif

	// TODO: store must be controlled with parsedData.access_controller
	// -----------------
	for (n = 0; n < count; n++) {
		randNum = RAND_R_FNC(seedCopy);
		if (!isInter) {
			accountIdx = CPU_ACCESS(randNum, nbAccounts);
		} else {
#if BANK_PART == 9
			// deterministic abort
			accountIdx = /*(n == 0) ? tid * 64 : */INTERSECT_ACCESS_CPU(randNum, nbAccounts);
#else
			accountIdx = INTERSECT_ACCESS_CPU(randNum, nbAccounts);
#endif /* BANK_PART == 9 */
		}
		TM_STORE(accounts + accountIdx, count_amount * input);
		// TM_STORE(pos[n], count_amount * input);
	}

	TM_COMMIT;

	someSeed = seedCopy;

	return 0;
}

int
readOnly2(
	account_t *accounts,
	volatile unsigned *positions,
	int isInter,
	int count,
	int tid,
	int nbAccounts
) {
	int n;
	// void *pos[count];
	// void *pos_write;
	int accountIdx;

	double count_amount = 0.0;

	unsigned input = positions[0];
	unsigned randNum;

	someSeed += tid ^ 1234;
	RAND_R_FNC(someSeed);

  TM_START(/*txid*/0, RW);
#if BANK_PART == 10
	count *= HETM_BANK_PART_SCALE;
#endif

	for (n = 0; n < count; n++) {
		randNum = RAND_R_FNC(someSeed);
		if (!isInter) {
			accountIdx = CPU_ACCESS(randNum, nbAccounts);
		} else {
#if BANK_PART == 9
			// deterministic abort
			accountIdx = /*(n == 0) ? tid * 64 : */INTERSECT_ACCESS_CPU(randNum, nbAccounts);
#else
			accountIdx = INTERSECT_ACCESS_CPU(randNum, nbAccounts);
#endif /* BANK_PART == 9 */
		}
		int someLoad = TM_LOAD(accounts + accountIdx);
		count_amount += someLoad * input;
  }

	TM_COMMIT;

	return count_amount;
}

int
transfer(
	account_t *accounts,
	volatile unsigned *positions,
	int count, int amount
) {
	uintptr_t load1, load2;
  	int n;
	void *src;
	void *dst;
	void *pos[count+1];
	void *pos_write;
	void *pos_write2;

#if BANK_PART == 7
	pos_write = &accounts[HeTM_thread_data[0]->id * 16];
	asm volatile("" ::: "memory");
#elif BANK_PART == 8
	static thread_local unsigned local_seed = (0x1234 << 10) + HeTM_thread_data[0]->id;
	unsigned rnd = RAND_R_FNC(local_seed);
	unsigned partionSize = parsedData.nb_accounts / parsedData.nb_threadsCPU;
	unsigned rnd_pos = (rnd % partionSize) * HeTM_thread_data[0]->id;
	rnd_pos = rnd_pos / parsedData.access_controller;
	pos_write = &accounts[rnd_pos];
#else
	for (n = 0; n < count; n += 2) {
		pos[n] = &accounts[positions[n]];
		pos[n+1] = &accounts[positions[n+1]];
		__builtin_prefetch(pos[n], 0, 0);
		__builtin_prefetch(pos[n+1], 0, 0);
		// printf("prefetch %i and %i\n", positions[n], positions[n+1]);
	}
	// int writeAccountIdx = src;
	int halfAccounts = parsedData.nb_accounts / 2;
	int writeAccountIdx = (positions[0] - halfAccounts) / parsedData.access_controller + halfAccounts;
	pos_write = &accounts[writeAccountIdx];
	int writeAccountIdx2 = (positions[1] - halfAccounts) / parsedData.access_controller + halfAccounts;
	pos_write2 = &accounts[writeAccountIdx2];

#endif /* BANK_PART == 7 */

  /* Allow overdrafts */
  TM_START(/*txid*/0, RW);
#if BANK_PART == 7 || BANK_PART == 8
	TM_STORE((int*)pos_write, 1234);
	// TM_STORE((int*)pos_write + 1, 1234);
	// TM_STORE((int*)pos_write + 2, 1234);
	// TM_STORE((int*)pos_write + 3, 1234);
	// TM_STORE((int*)pos_write + 4, 1234);
	// TM_STORE((int*)pos_write + 5, 1234);
	// TM_STORE((int*)pos_write + 6, 1234);
	// TM_STORE((int*)pos_write + 7, 1234);
	// TM_STORE((int*)pos_write + 8, 1234);
	// TM_STORE((int*)pos_write + 9, 1234);
	// TM_STORE((int*)pos_write + 10, 1234);
	// TM_STORE((int*)pos_write + 11, 1234);
	// TM_STORE((int*)pos_write + 12, 1234);
	// TM_STORE((int*)pos_write + 13, 1234);
	// TM_STORE((int*)pos_write + 14, 1234);
	// TM_STORE((int*)pos_write + 15, 1234);
#else
	for (n = 0; n < count; n += 2) {
		src = pos[n];
		dst = pos[n+1];

		// Problem: TinySTM works with the granularity of 8B, PR-STM works with 4B
		load1 = TM_LOAD(src);
		load1 -= COMPUTE_TRANSFER(amount);

		load2 = TM_LOAD(dst);
		load2 += COMPUTE_TRANSFER(amount);
	}

	// TODO: store must be controlled with parsedData.access_controller
	// -----------------

	TM_STORE(pos_write, load1); // TODO: now is 2 reads 1 write
	TM_STORE(pos_write2, load1); // TODO: now is 2 reads 1 write
	// TM_STORE(&accounts[dst], load2);

#endif /* BANK_PART != 7 */
  TM_COMMIT;

  // TODO: remove this
//   volatile int j = 0;
// loop:
//   j++;
//   if (j < 100) goto loop;

  return amount;
}

int 
transfer_simple(
	account_t *accounts,
	volatile unsigned *positions,
	int count,
	int amount
) {
	uintptr_t load1, load2;
	void *src;
	void *dst;

	src = &accounts[positions[0]];
	dst = &accounts[positions[0]+1];

	/* Allow overdrafts */
	TM_START(/*txid*/0, RW);
	load1 = TM_LOAD(src);
	load2 = TM_LOAD(dst);
	TM_STORE(src, load1-amount);
	TM_STORE(dst, load2+amount);
	TM_COMMIT;

	return 0;
}


int readOnly(account_t *accounts, volatile unsigned *positions, int count, int amount)
{
  	int n;
	int loads[count];
	float res = 0;
	int resI = 0;

	for (n = 0; n < count; ++n) {
		__builtin_prefetch(&accounts[positions[n]], 0, 0);
	}

  	TM_START(/*txid*/0, RW);

	for (n = 0; n < count; ++n) {
		loads[n] = TM_LOAD(&accounts[positions[n]]);
			res += loads[n] * INTEREST_RATE;
			// res = __builtin_cos(res);
	}
	res += amount;
	resI = (int)res;
	// TM_STORE(&accounts[positions[0]], resI);
	// accounts[positions[0]] = resI;
  	TM_COMMIT;
 	return resI;
}


int total(bank_t *bank, int transactional)
{
  long i;
  long total;

  if (!transactional) {
    total = 0;
    for (i = 0; i < bank->size; i++) {
      total += bank->accounts[i];
    }
  } else {
    TM_START(1, RO);
    total = 0;
    for (i = 0; i < bank->size; i++) {
      total += TM_LOAD(&bank->accounts[i]);
    }
    TM_COMMIT;
  }
  return total;
}


void reset(bank_t *bank)
{
  long i;

  TM_START(/*txid*/2, RW);
  for (i = 0; i < bank->size; i++) {
    TM_STORE(&bank->accounts[i], 0);
  }
  TM_COMMIT;
}

int readIntensive(account_t *accounts, volatile unsigned *positions, int count, int amount)
{
  int n;
	int loads[count];
	float res = 0;
	int resI = 0;

	__builtin_prefetch(&accounts[positions[0]], 1, 2);
	for (n = 1; n < count; ++n) {
		__builtin_prefetch(&accounts[positions[n]], 0, 0);
	}

  /* Allow overdrafts */
  TM_START(/*txid*/0, RW);

  for (n = 0; n < count; ++n) {
    loads[n] = TM_LOAD(&accounts[positions[n]]);
		res += loads[n] * INTEREST_RATE;
  }
	res += amount;
	resI = (int)res;

	TM_STORE(&accounts[positions[0]], resI);
	accounts[positions[0]] = resI;

  TM_COMMIT;

// #if BANK_PART == 3
//   // TODO: BANK_PART 3 benifits VERS somehow
  // volatile int j = 0;
	// loop:
	//   j++;
	//   if (j < 150) goto loop;
// #endif /* BANK_PART == 3 */
  return amount;
}

int transferReadOnly(account_t *accounts, volatile unsigned *positions, int count, int amount)
{
	uintptr_t load1, load2;
  	int n, src, dst;

	for (n = 0; n < count; ++n) {
		__builtin_prefetch(&accounts[positions[n]], 0, 0);
	}

	/* Allow overdrafts */
	TM_START(/*txid*/0, RW);

	for (n = 0; n < count; n += 2) {
		src = positions[n];
		dst = positions[n+1];

		load1 = TM_LOAD(&accounts[src]);
		load1 -= COMPUTE_TRANSFER(amount);

		load2 = TM_LOAD(&accounts[dst]);
		load2 += COMPUTE_TRANSFER(amount);

			// TM_STORE(&accounts[src], load1);
		// TM_STORE(&accounts[dst], load2);
	}
	TM_COMMIT;
	return load1 + load2;
}
