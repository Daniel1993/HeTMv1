#ifndef AUX_H_GUARD
#define AUX_H_GUARD

#ifndef NDEBUG
#define DEB_PRINTF(...) printf(__VA_ARGS__)
#else /* DEBUG */
#define DEB_PRINTF(...) /* empty */
#endif /* NDEBUG */

#define CLOCK_DIFF_MS(startClock, endClock) \
  ((double)(endClock.tv_sec-startClock.tv_sec) * 1000.0f + \
  (double)(endClock.tv_nsec-startClock.tv_nsec) / 1.0e6f)

#if defined(USE_ROLLBACK)
#define BATCH_FACTOR 4
#else 
#define BATCH_FACTOR 0
#endif

#endif /* AUX_H_GUARD */