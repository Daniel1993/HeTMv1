#ifndef BANK_AUX_H_
#define BANK_AUX_H_

#include "bitmap.hpp"

// TODO: refactor this code else where
/* ################################################################### *
 * GPU ENABLE
 * ################################################################### */
// LOCK --> usa a cuda_barrier para não premitir avançar para o próximo batch (depois da cmp)

#ifdef USE_TSX_IMPL
  #define BANK_GET_STATS_TSX(d) \
    d->nb_aborts = TM_get_error(HTM_ABORT); \
    d->nb_aborts_1 = TM_get_error(HTM_CONFLICT); \
    d->nb_aborts_2 = TM_get_error(HTM_CAPACITY); \
//
#else
  #define BANK_GET_STATS_TSX(d) /* empty */
#endif /* USE_TSX_IMPL */

#if !defined(TM_COMPILER) && !defined(USE_TSX_IMPL)
  #define BANK_GET_STATS(d) \
    stm_get_stats("nb_aborts", &d->nb_aborts); \
    stm_get_stats("nb_aborts_1", &d->nb_aborts_1); \
    stm_get_stats("nb_aborts_2", &d->nb_aborts_2); \
    stm_get_stats("nb_aborts_locked_read", &d->nb_aborts_locked_read); \
    stm_get_stats("nb_aborts_locked_write", &d->nb_aborts_locked_write); \
    stm_get_stats("nb_aborts_validate_read", &d->nb_aborts_validate_read); \
    stm_get_stats("nb_aborts_validate_write", &d->nb_aborts_validate_write); \
    stm_get_stats("nb_aborts_validate_commit", &d->nb_aborts_validate_commit); \
    stm_get_stats("nb_aborts_invalid_memory", &d->nb_aborts_invalid_memory); \
    stm_get_stats("nb_aborts_killed", &d->nb_aborts_killed); \
    stm_get_stats("locked_reads_ok", &d->locked_reads_ok); \
    stm_get_stats("locked_reads_failed", &d->locked_reads_failed); \
    stm_get_stats("max_retries", &d->max_retries); \
//
#elif !defined(TM_COMPILER) && defined(USE_TSX_IMPL)
#define BANK_GET_STATS(d) \
	BANK_GET_STATS_TSX(d); \
//
#else
  #define BANK_GET_STATS(d) /* empty */
#endif /* ! TM_COMPILER */

//   Memory access layout
// +------------+--------------------+
// | NOT ACCESS |      CPU_PART      |
// +------------+--------------------+
// +------------------+--------------+
// |     GPU_PART     |  NOT ACCESS  |
// +------------------+--------------+
//

// BANK_PART == 1 does partitioned accesses
#if BANK_PART == 1

// distRnd is not used here
#define BANK_PREPARE_TRANSFER(id, seed, distRnd, HMult, total, account_vec, nb_accounts) ({ \
  int i, src, dst; \
  for (i = 0; i < d->trfs; i += 2) { \
    int is_intersect = isInterBatch; /* IS_INTERSECT_HIT( RAND_R_FNC(seed) ); */ \
		if (!is_intersect) { \
      long rnd = RAND_R_FNC(seed); \
			src = CPU_ACCESS(rnd, (nb_accounts - 2)) + 1; \
		} else { \
			src = INTERSECT_ACCESS_CPU(RAND_R_FNC(seed), (nb_accounts - 2)) + 1; \
		} \
    dst = (src + 1) % nb_accounts; \
    /*printf("CPU - addr(%i): %i  --  %i\n", i, src, dst);*/ \
    accounts_vec[i]   = src; \
    accounts_vec[i+1] = dst; \
  } \
}) \
//
#define BANK_PREPARE_READ_INTENSIVE(id, seed, distRnd, HMult, total, account_vec, nb_accounts) ({ \
  int i, pos; \
  int is_intersect = isInterBatch; /* IS_INTERSECT_HIT( RAND_R_FNC(seed) ); */ \
  if (!is_intersect) { \
    long rnd = RAND_R_FNC(seed); \
    pos = CPU_ACCESS(rnd, (nb_accounts - d->trfs * 10 - 2)) + 1; \
  } else { \
    pos = INTERSECT_ACCESS_CPU(RAND_R_FNC(seed), (nb_accounts - d->trfs * 10 - 2)) + 1; \
  } \
  accounts_vec[i] = pos; \
  for (i = 1; i < d->trfs * 20; ++i) { \
    accounts_vec[i] = accounts_vec[i-1] + 1; \
  } \
}) \
//
#elif BANK_PART == 2 /* using hotspots */

#define BANK_PREPARE_TRANSFER(id, seed, distRnd, HMult, total, account_vec, nb_accounts) ({ \
  int i; \
  for (i = 0; i < d->trfs; i += 2) { \
    int is_hot = IS_ACCESS_H( RAND_R_FNC(seed), distRnd ); \
    if (!is_hot) { \
      src = CPU_ACCESS_H(RAND_R_FNC(seed), HMult, (nb_accounts - 2)) + 1; \
    } else { \
      src = CPU_ACCESS_M(RAND_R_FNC(seed), HMult, (nb_accounts - 2)) + 1; \
    } \
    dst = (src + 1) % nb_accounts; \
    accounts_vec[i]   = src; \
    accounts_vec[i+1] = dst; \
  } \
}) \
//
#define BANK_PREPARE_READ_INTENSIVE(id, seed, distRnd, HMult, total, account_vec, nb_accounts) ({ \
  int i, pos; \
  int is_hot = IS_ACCESS_H( RAND_R_FNC(seed), distRnd ); \
  if (!is_hot) { \
    pos = CPU_ACCESS_H(RAND_R_FNC(seed), HMult, (nb_accounts - d->trfs * 10 - 1)) + 1; \
  } else { \
    pos = CPU_ACCESS_M(RAND_R_FNC(seed), HMult, (nb_accounts - d->trfs * 10 - 1)) + 1; \
  } \
  for (i = 0; i < d->trfs * 10; ++i) { \
    accounts_vec[i] = accounts_vec[i-1] + 1; \
  } \
}) \
//
#else /* BANK_PART != 1||2 */

// use distRnd
#define BANK_PREPARE_TRANSFER(id, seed, distRnd, HMult, total, account_vec, nb_accounts) ({ \
  int i; \
  for (i = 0; i < d->trfs; i += 2) { \
    distRnd = distRnd % (CPU_TOP_IDX(nb_accounts) - CPU_BOT_IDX(nb_accounts)); \
    distRnd = (CPU_TOP_IDX(nb_accounts) - 2) - distRnd; \
		src = distRnd; /*INTERSECT_ACCESS_CPU(RAND_R_FNC(seed)%(nb_accounts-1), nb_accounts-1);*/ \
    dst = (src + 1) % nb_accounts; \
    accounts_vec[i]   = src; \
    accounts_vec[i+1] = dst; \
  } \
}) \
// BANK_PREPARE_READ_INTENSIVE not defined
#endif /* BANK_PART == 1 */

#endif /* BANK_AUX_H_ */
