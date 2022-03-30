// TODO: it seems each file is compiled with different flags...
#ifndef SYNC_BALANCING
#define SYNC_BALANCING_VALUE 0
#else
#define SYNC_BALANCING_VALUE 1
#endif

#ifndef CPU_INV
#define CPU_INV_VALUE 0
#else
#define CPU_INV_VALUE 1
#endif

#ifndef NO_LOCK
#define NO_LOCK_VALUE 0
#else
#define NO_LOCK_VALUE 1
#endif

#ifndef USE_STREAM
#define USE_STREAM_VALUE 0
#else
#define USE_STREAM_VALUE 1
#endif

#ifndef USE_TSX_IMPL
#define USE_TSX_IMPL_VALUE 0
#else
#define USE_TSX_IMPL_VALUE 1
#endif

// BM_TRANSF --> so envia chunks de memória alterada (evita transferir o dataset inteiro)

// BM_TRANSF+CPU_INV é capaz de dar contas erradas

#ifndef PRINT_FLAGS
#define PRINT_FLAGS() \
  printf("Flag values    : "); \
  printf("BANK_PART=%i ", BANK_PART); \
  printf("HETM_CPU_EN=%i ", HETM_CPU_EN); \
  printf("HETM_GPU_EN=%i ", HETM_GPU_EN); \
  printf("CMP_APPLY=%i ", CMP_APPLY); \
  printf("SYNC_BALANCING=%i ", SYNC_BALANCING_VALUE); \
  printf("RUN_ONCE=%i ", RUN_ONCE); \
  printf("CPU_INV=%i ", CPU_INV_VALUE); \
  printf("NO_LOCK=%i ", NO_LOCK_VALUE); \
  printf("BM_TRANSF=%i ", BM_TRANSF); \
  printf("USE_STREAM=%i ", USE_STREAM_VALUE); \
  printf("USE_TSX_IMPL=%i ", USE_TSX_IMPL_VALUE); \
  printf("CPU_PART=%f ", CPU_PART); \
  printf("GPU_PART=%f ", GPU_PART); \
  printf("P_INTERSECT=%f ", P_INTERSECT); \
  printf("DEFAULT_blockNum=%i ", DEFAULT_blockNum); \
  printf("DEFAULT_threadNum=%i ", DEFAULT_threadNum); \
  printf("\n"); \
//
#endif /* PRINT_FLAGS */

// TODO: use #pragma message to print the flags
