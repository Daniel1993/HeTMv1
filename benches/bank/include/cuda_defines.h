#ifndef _CUDA_DEFS
#define _CUDA_DEFS

#define DEFAULT_hashNum           1       // how many accounts shared 1 lock
#define DEFAULT_arraySize         2621440 // total accounts number 10M = 2621440 integer
#define DEFAULT_blockNum          128     // block number
#define DEFAULT_threadNum         256     // threads number
#define resetNum                  0       // how many threads do all write
#define totalNum                  0       // how many threads do all read

#ifndef BANK_NB_TRANSFERS
#define BANK_NB_TRANSFERS         2       // transfer money between 1 accounts
#endif

#ifndef MEMCD_NB_TRANSFERS
#define MEMCD_NB_TRANSFERS        16       // TODO
#endif

#ifndef DEFAULT_TransEachThread
#ifdef BENCH_BANK
#define DEFAULT_TransEachThread   50      // how many transactions each thread do
#else /* BENCH_MEMCD */
#define DEFAULT_TransEachThread   1       // how many transactions each thread do
#endif /* BENCH */
#endif /* DEFAULT_TransEachThread */

// MEMCACHED ------------------
#define NUMBER_WAYS               32
#define NUMBER_SETS               512
#define WRITE_PERCENT             1     // 0-100 percentage of write requests (memcached GET)
#define QUEUE_SIZE                128*128//1024*1024
#define QUEUE_SHARED_VALUE        1
// ----------------------------

// Support for the compressed log implementation
#define readMask                  0b01   // Set entry to write
#define writeMask                 0b10   // Set entry to read

// For testing purposses
#define firstXEn                  1      // Set to 1 to force transactions to happen betwen the firstX accounts
#ifndef firstX
  #define firstX                  1      // Configure the limit(number of shifts)
#endif

#ifndef CMP_APPLY
  #define CMP_APPLY               1      // Set to 1 to enable applying data while comparing
#endif
#define BM_HASH                   14
#define BM_HASH_SIZE              16384

#define uint_64                   int /* TODO: unsigned long*/

// #define SET_AFFINITY

// MEMCACHED ------------------
typedef struct queue_element {
	unsigned int *values;
	unsigned int size;
	unsigned int current;
	pthread_mutex_t lock;
} queue_element_t;

typedef struct queue {
	queue_element_t cpu_queue;
	queue_element_t gpu_queue;
	queue_element_t shared_queue;
	int shared_rate;
} queue_t;

// no output for the set
typedef struct memcd_get_output_ {
	int isFound;
	int value;
	int val2;
	int val3;
	int val4;
	int val5;
	int val6;
	int val7;
	int val8;
} memcd_get_output_t;
// ----------------------------

#endif
