/* Include this file in your pr-stm-implementation file,
 * then use the MACROs accordingly!
 * Do not forget to include pr-stm-internal.cuh after this
 */
#ifndef PR_STM_TRACK_RWSET_H_GUARD_
#define PR_STM_TRACK_RWSET_H_GUARD_

#include "pr-stm-track-rwset-structs.cuh"

PR_DEVICE void PR_after_val_locks_EXT (PR_args_s *args)
{
	int _id = PR_THREAD_IDX;
	int _loc = _id * (PR_MAX_RWSET_SIZE + 1);
	pr_args_ext_t *GPU_log = (pr_args_ext_t*)(args->pr_args_ext);
	GPU_log->write_addrs[_loc] = (PR_GRANULE_T*)args->wset.size;
	GPU_log->read_addrs[_loc] = (PR_GRANULE_T*)args->rset.size;
	// printf("TX%3i rSize=%i wSize=%i PR_MAX_RWSET_SIZE=%i loc=%i\n", _id, args->rset.size, args->wset.size, PR_MAX_RWSET_SIZE, _loc);
	for (int _i = 0; _i < max(args->rset.size, args->wset.size); _i++) {
		GPU_log->read_addrs[_loc + _i + 1] = args->rset.addrs[_i];
		GPU_log->write_addrs[_loc + _i + 1] = args->wset.addrs[_i];
	}
}

// Logs the GPU write-set after acquiring the locks
PR_DEVICE void PR_after_writeback_EXT (PR_args_s *args, int i, PR_GRANULE_T *addr, PR_GRANULE_T val) { }

#include "pr-stm.cuh"

#endif /* PR_STM_TRACK_RWSET_H_GUARD_ */
