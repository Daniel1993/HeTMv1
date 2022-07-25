#include "tsx_impl.h"

__thread uint64_t HeTM_htmRndSeed = 0x000012348231ac4e;

int errors[HETM_MAX_THREADS][HTM_NB_ERRORS];
