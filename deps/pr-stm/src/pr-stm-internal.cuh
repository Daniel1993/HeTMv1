#ifndef PR_STM_I_H_GUARD
#define PR_STM_I_H_GUARD

#include "pr-stm.cuh"

#include <curand_kernel.h>
#include <cuda_profiler_api.h>
#include "helper_cuda.h"
#include "helper_timer.h"

// Global vars
PR_globalVars;

// ----------------------------------------
// Utility defines
#define PR_ADD_TO_ARRAY(array, size, value) ({ \
	(array)[(size)++] = (value); \
})

#define PR_ADD_TO_LOG(log, value) ({ \
	((void**)log.buf)[log.size++] = ((void*)value); \
})

// assume find ran first
#define PR_RWSET_GET_VAL(rwset, idx) \
	(rwset).values[idx] \
//

#define PR_RWSET_SET_VAL(rwset, idx, buf) \
	((rwset).values[idx] = buf) \
//

#define PR_RWSET_SET_VERSION(rwset, idx, version) \
	((rwset).versions[idx] = (int)version) \
//

#define PR_ADD_TO_RWSET(rwset, addr, version, value) \
	(rwset).addrs[(rwset).size] = (PR_GRANULE_T*)(addr); \
	PR_RWSET_SET_VAL(rwset, (rwset).size, value); \
	PR_RWSET_SET_VERSION(rwset, (rwset).size, version); \
	(rwset).size += 1; \
//

#define PR_FIND_IN_RWSET(rwset, addr) ({ \
	int i; \
	PR_GRANULE_T* a = (PR_GRANULE_T*)(addr); \
	long res = -1; \
	for (i = 0; i < (rwset).size; i++) { \
		if ((rwset).addrs[i] == a) { \
			res = i; \
			break; \
		} \
	} \
	res; \
})

// ----------------------------------------

//openRead Function reads data from global memory to local memory. args->rset_vals stores value and args->rset_versions stores version.
PR_DEVICE PR_GRANULE_T PR_i_openReadKernel(
	PR_args_s *args, PR_GRANULE_T *addr
) {
	int j, k;
	int temp, version;
	PR_GRANULE_T res;
	// volatile int *data = (volatile int*)args->data;
	// int target = args->rset_addrs[args->rset_size];

	temp = PR_GET_MTX(args->mtx, addr);
	// ------------------------------------------------------------------------
	// TODO: no repeated reads/writes
	k = PR_FIND_IN_RWSET(args->wset, addr);
	// ------------------------------------------------------------------------

	if (!args->is_abort && !PR_CHECK_PRELOCK(temp)) {
		// ------------------------------------------------------------------------
		// if (PR_THREAD_IDX == 405) printf("[405] did not abort yet!\n");
		// not locked
		if (k != -1) {
			// in wset
			res = PR_RWSET_GET_VAL(args->wset, k);
		} else {
			j = PR_FIND_IN_RWSET(args->rset, addr);
		// ------------------------------------------------------------------------
			version = PR_GET_VERSION(temp);
		// ------------------------------------------------------------------------
			if (j != -1) {
				// in rset
				// TODO: this seems a bit different in the paper
				res = PR_RWSET_GET_VAL(args->rset, j);
				PR_RWSET_SET_VERSION(args->rset, j, version);
			} else {
				// not found
		// ------------------------------------------------------------------------
				res = *addr;
				PR_ADD_TO_RWSET(args->rset, addr, version, res);
		// ------------------------------------------------------------------------
			}
		}
		// ------------------------------------------------------------------------
	} else {
		// if (PR_THREAD_IDX == 405) printf("[405] PR_CHECK_PRELOCK failed lock = 0x%x tid = %i version = %i addr = %p!\n",
		// 	temp, PR_GET_OWNER(temp), PR_GET_VERSION(temp), addr);
		// printf("   [%i] <ABORT> read addr %p, isprelocked = %i\n", args->tid, addr, PR_CHECK_PRELOCK(temp));
		res = *addr;
		args->is_abort = 1;
		__threadfence();
	}
	return res;
}

PR_DEVICE void PR_i_openWriteKernel(
	PR_args_s *args, PR_GRANULE_T *addr, PR_GRANULE_T wbuf
) {
	int j = -1, k = -1;
	int temp, version;

	temp = PR_GET_MTX(args->mtx, addr);

	// printf("write addr %p\n", addr);

	if (!args->is_abort && !PR_CHECK_PRELOCK(temp)) {
		// ------------------------------------------------------------------------
		// // not locked --> safe to access TODO: the rset seems redundant
		// TODO: non-repeated writes
		j = PR_FIND_IN_RWSET(args->rset, addr); // check in the read-set
		k = PR_FIND_IN_RWSET(args->wset, addr); // check in the write-set
		// ------------------------------------------------------------------------
		version = PR_GET_VERSION(temp);

		// ------------------------------------------------------------------------
		// TODO: assume read-before-write
		if (j == -1) {
			// not in the read-set
			PR_ADD_TO_RWSET(args->rset, addr, version, wbuf);
		} else {
			// update the read-set (not the version) TODO: is this needed?
			PR_RWSET_SET_VAL(args->rset, j, wbuf);
		}
		// ------------------------------------------------------------------------

		// ------------------------------------------------------------------------
		if (k == -1) {
		// ------------------------------------------------------------------------
			// does not exist in write-set
			PR_ADD_TO_RWSET(args->wset, addr, version, wbuf);
		// ------------------------------------------------------------------------
		} else {
			// update the write-set
			PR_RWSET_SET_VAL(args->wset, j, wbuf);
		}
		// ------------------------------------------------------------------------
	} else {
		// already locked
		// printf("   [%i] <ABORT> write addr %p, isprelocked = %i\n", args->tid, addr, PR_CHECK_PRELOCK(temp));
		args->is_abort = 1;
		__threadfence();
	}
}

PR_DEVICE static int
PR_i_tryPrelock(PR_args_s *args)
{
	int i, res = 0;
	PR_GRANULE_T *addr;
	int version, lock;
	bool validVersion, notLocked, ownerHasHigherPriority;
	int tid = args->tid;

	for (i = 0; i < args->rset.size; i++) {
		addr = args->rset.addrs[i];
		version = args->rset.versions[i];
		lock = PR_GET_MTX(args->mtx, addr);
		validVersion = PR_GET_VERSION(lock) == version;
		notLocked = !PR_CHECK_PRELOCK(lock);
		ownerHasHigherPriority = PR_GET_OWNER(lock) < tid;
		if (validVersion && (notLocked || !ownerHasHigherPriority)) {
			res++;
			continue;
		} else {
			res = -1; // did not validate
			break;
		}
	}
	return res;
}

PR_DEVICE static void
PR_i_unlockWset(PR_args_s *args)
{
	int i;
	int *lock, lval, nval;
	int tid = args->tid;

	for (i = 0; i < args->wset.size; i++) {
		lock = (int*) &(PR_GET_MTX(args->mtx, args->wset.addrs[i]));
		lval = *lock;
		nval = lval & PR_MASK_VERSION;
		if (PR_GET_OWNER(lval) == tid)
		{
			atomicCAS(lock, lval, nval);
			// if (atomicCAS(lock, lval, nval) != lval)
			// {
			// 	printf("   [%i] unlock %p failed\n", tid, lock);
			// }
		}
	}
}

PR_DEVICE static int
PR_i_prelock(PR_args_s *args)
{
	int tid = args->tid;
	int i, k;
	int *lock, *old_lock, new_lock, lval, oval, nval;
	int isLocked, isPreLocked, ownerIsSelf, ownerHasHigherPriority, newVersion;
	int retries = 0;
	const int MAX_RETRIES = 100;

	for (i = 0; i < args->wset.size; i++) {
		retries = MAX_RETRIES;
		while (retries--) {
			// spin until this thread can lock one account in write set

			lock = (int*)&(PR_GET_MTX(args->mtx, args->wset.addrs[i]));
			lval = *lock;

			//check if version changed or locked by higher priority thread
			isLocked = PR_CHECK_LOCK(lval);
			isPreLocked = PR_CHECK_PRELOCK(lval);
			ownerHasHigherPriority = PR_GET_OWNER(lval) < tid;
			newVersion = PR_GET_VERSION(lval) != args->wset.versions[i];

			// printf("PR_i_prelock lock %p retries = %i\n", lock, retries);

			if (isLocked || (isPreLocked && ownerHasHigherPriority) || newVersion) {
				// if one of accounts is locked by a higher priority thread
				// or version changed, unlock all accounts it already locked
				for (k = 0; k < i; k++)
				{
					old_lock = (int*)&(PR_GET_MTX(args->mtx, args->wset.addrs[k]));
					oval = *old_lock;

					// check if this account is still locked by itself
					isPreLocked = PR_CHECK_PRELOCK(oval);
					ownerIsSelf = PR_GET_OWNER(oval) == tid;
					if (isPreLocked && ownerIsSelf)
					{
						nval = oval & PR_MASK_VERSION;
						if (atomicCAS(old_lock, oval, nval) != oval)
						{
// printf("failed lock %p\n", old_lock);
							 // TODO: do not be so agressive here
							return -1;
						}
					}
				}
				// if one of accounts is locked by a higher priority thread or version changed, return false
				return -1;
			}
			new_lock = PR_PRELOCK_VAL(args->wset.versions[i], tid);

			// atomic lock that account
			if (atomicCAS(lock, lval, new_lock) == lval)
			{
				break;
			}
		}
		if (!retries)
		{
			return -1;
		}
	}
	return 0;
}

PR_DEVICE void
PR_i_validateKernel(PR_args_s *args)
{
	int vr = 0; // flag for how many values in read  set is still validate
	int vw = 0; // flag for how many values in write set is still validate
	int i;
	int *lock, new_lock, lval;
	int tid = args->tid;
	int ownerIsSelf;

	// if (args->is_abort) return;

	vr = PR_i_tryPrelock(args);
	if (vr == -1) {
// printf("[%i] could not tryprelock\n", tid);
		args->is_abort = 1;
		__threadfence();
		// return; // abort
	}

	// __threadfence(); // The paper does not mention this fence

	if (-1 == PR_i_prelock(args))
	{
// printf("[%i] could not prelock\n", tid);
		PR_i_unlockWset(args);
		args->is_abort = 1;
		__threadfence();
		// return;
	}

	// __threadfence(); // The paper does not mention this fence
	// if this thread can pre-lock all accounts it needs to, really lock them
	for (i = 0; i < args->wset.size; i++) {
		// get lock|owner|version from global memory
	 	lock = (int*) &(PR_GET_MTX(args->mtx, args->wset.addrs[i]));
		lval = *lock;

		// if it is still locked by itself (check lock flag and owner position)
		ownerIsSelf = PR_GET_OWNER(lval) == tid;
		if (!ownerIsSelf || args->is_abort) {
			// cannot final lock --> unlock all
// printf("[%i] could not finallock\n", tid);
			PR_i_unlockWset(args);
			args->is_abort = 1;
			// return;
		} else {
			new_lock = PR_LOCK_VAL(args->wset.versions[i], tid); // temp final lock
			if (atomicCAS(lock, lval, new_lock) != lval) {
				// failed to lock
// printf("[%i] could not lock\n", tid);
				PR_i_unlockWset(args);
				args->is_abort = 1;
				// return;
			} else {
				vw++;	// if succeed, vw++
			}
		}
	}
	if (!args->is_abort) {
		// assert(vw == args->wset.size && vr == args->rset.size);
		// all locks taken (should not need the extra if)
		PR_after_val_locks_EXT(args);
	}
}

PR_DEVICE void PR_i_commitKernel(PR_args_s *args)
{
	int i;
	int *lock, nval, lval;
	PR_GRANULE_T *addr, val;
	int tid = args->tid;

	__threadfence();
	// printf("[%i] PR_i_commitKernel args->is_abort = %i args->wset.size = %i\n", tid, args->is_abort, args->wset.size);

	// write all values in write set back to global memory and increase version
	if (!args->is_abort)
	{
		for (i = 0; i < args->wset.size; i++) {
			addr = args->wset.addrs[i];
			val = args->wset.values[i]; // TODO: variable size
			PR_SET(addr, val);
			PR_after_writeback_EXT(args, i, addr, val);
		}
	}
	
	for (i = 0; i < args->wset.size; i++) {
		lock = (int*) &(PR_GET_MTX(args->mtx, args->wset.addrs[i]));
		lval = *lock;
		// increase version(if version is going to overflow, change it to 0)
		nval = args->wset.versions[i] < PR_LOCK_VERSION_OVERFLOW ?
			(args->wset.versions[i] + 1) << PR_LOCK_VERSION_BITS :
			0;
		// printf("   [%i] >COMMIT< unlock %p owner = %i\n", args->tid, lock, PR_GET_OWNER(lval));
		if (PR_GET_OWNER(lval) == tid)
		{
			atomicCAS(lock, lval, nval);
		}
		// if (atomicCAS(lock, lval, nval) != lval)
		// {
		// 	printf("   [%i] >COMMIT< unlock FAILED!!!! %p\n", args->tid, lock);
		// }
		// *lock = nval;
	}
}

// Be sure to match the kernel config!
PR_ENTRY_POINT void PR_reduceCommitAborts(
	int doReset,
	int targetStream,
	PR_globalKernelArgs,
	uint64_t *nbCommits,
	uint64_t *nbAborts)
{
	const int      WARP_SIZE         = 32;
	const uint32_t FULL_MASK         = 0xffffffff;
	const int      MAX_SHARED_MEMORY = 32;
	__shared__ uint64_t sharedSumCommits[MAX_SHARED_MEMORY];
	__shared__ uint64_t sharedSumAborts [MAX_SHARED_MEMORY];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int idx = tid + blockDim.x*gridDim.x*targetStream;
	uint64_t tmpAborts  = args.nbAborts [args.devId][idx];
	uint64_t tmpCommits = args.nbCommits[args.devId][idx];
	uint32_t mask;
	int writerTid;
	int writerPos;

	if (doReset) {
		args.nbAborts [idx] = 0;
		args.nbCommits[idx] = 0;
	}

	mask = __ballot_sync(FULL_MASK, 1);

	#pragma unroll (4)
	for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
		tmpAborts  += __shfl_xor_sync(mask, tmpAborts,  offset);
		tmpCommits += __shfl_xor_sync(mask, tmpCommits, offset);
	}

	writerTid = threadIdx.x % WARP_SIZE;
	writerPos = threadIdx.x / WARP_SIZE;

	if (writerTid == 0) {
		sharedSumAborts [writerPos] = tmpAborts ;
		sharedSumCommits[writerPos] = tmpCommits;
	}
	// Shared memory must be synchronized via barriers
	__syncthreads();
	if (threadIdx.x == 0) {
		// first thread in the entire block sums the rest
		tmpAborts  = sharedSumAborts [0];
		tmpCommits = sharedSumCommits[0];
		int divWarpSize = blockDim.x / WARP_SIZE;
		int modWarpSize = blockDim.x % WARP_SIZE > 0 ? 1 : 0;
		divWarpSize += modWarpSize;
		for (int i = 1; i < divWarpSize; i++) {
			tmpAborts  += sharedSumAborts [i];
			tmpCommits += sharedSumCommits[i];
		}
		// Global memory must be synchronized via atomics
		atomicAdd((unsigned long long*)nbAborts,  (unsigned long long)tmpAborts);
		atomicAdd((unsigned long long*)nbCommits, (unsigned long long)tmpCommits);
	}
}

// PR_DEVICE void PR_beforeKernel_EXT(PR_txCallDefArgs) { PR_BEFORE_KERNEL_EXT(args, pr_args); }
// PR_DEVICE void PR_afterKernel_EXT (PR_txCallDefArgs) { PR_AFTER_KERNEL_EXT(args, pr_args);  }
// PR_DEVICE void PR_beforeBegin_EXT (PR_txCallDefArgs) { PR_BEFORE_BEGIN_EXT(args, pr_args);  }
// PR_DEVICE void PR_afterCommit_EXT (PR_txCallDefArgs) { PR_AFTER_COMMIT_EXT(args, pr_args);  }

#endif /* PR_STM_I_H_GUARD */
