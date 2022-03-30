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
  printf("Flag values: \n"); \
  printf("    HETM_CPU_EN    =%i\n", HETM_CPU_EN); \
  printf("    HETM_GPU_EN    =%i\n", HETM_GPU_EN); \
  printf("    CMP_APPLY      =%i\n", CMP_APPLY); \
  printf("    SYNC_BALANCING =%i\n", SYNC_BALANCING_VALUE); \
  printf("    RUN_ONCE       =%i\n", RUN_ONCE); \
  printf("    CPU_INV        =%i\n", CPU_INV_VALUE); \
  printf("    NO_LOCK        =%i\n", NO_LOCK_VALUE); \
  printf("    BM_TRANSF      =%i\n", BM_TRANSF); \
  printf("    USE_STREAM     =%i\n", USE_STREAM_VALUE); \
  printf("    USE_TSX_IMPL   =%i\n", USE_TSX_IMPL_VALUE); \
  printf("    CPU_PART       =%f\n", CPU_PART); \
  printf("    GPU_PART       =%f\n", GPU_PART); \
  printf("    P_INTERSECT    =%f\n", P_INTERSECT); \
  printf("    DEFAULT_blockNum  =%i\n", DEFAULT_blockNum); \
  printf("    DEFAULT_threadNum =%i\n", DEFAULT_threadNum); \
//
#endif /* PRINT_FLAGS */

// TODO: use #pragma message to print the flags
