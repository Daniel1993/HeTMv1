/* Include this file in your pr-stm-implementation file,
 * then use the MACROs accordingly!
 * Do not forget to include pr-stm-internal.cuh after this
 */
#ifndef PR_STM_TRACK_RWSET_STRUCTS_H_GUARD_
#define PR_STM_TRACK_RWSET_STRUCTS_H_GUARD_

#define PR_GRANULE_T        int // TODO
#define PR_LOCK_GRANULARITY 4 /* size in bytes */
#define PR_LOCK_GRAN_BITS   2 /* log2 of the size in bytes */
#define PR_LOCK_TABLE_SIZE  0x800000

#ifndef PRSTM_TRACK_RWSET_MAX_THREADS
#define PRSTM_TRACK_RWSET_MAX_THREADS (128*128)
#endif

typedef struct {
	PR_GRANULE_T **read_addrs;
	PR_GRANULE_T **write_addrs;
	unsigned char *confl_mat;
	int nbThreads;
	int max_rwset_size;
} pr_args_ext_t;

#ifndef PRSTM_TRACK_RWSET_MAX_SIZE /* note first entry is the RWSET_SIZE */
#define PRSTM_TRACK_RWSET_MAX_SIZE ((PR_MAX_RWSET_SIZE+1)*sizeof(PR_GRANULE_T*)) /* *NB_THREADS_IN_GPU */
#endif
extern int prstm_track_rwset_max_size;

#endif /* PR_STM_TRACK_RWSET_STRUCTS_H_GUARD_ */
