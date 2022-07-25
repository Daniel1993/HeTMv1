#include "TestCP.cuh"

CPPUNIT_TEST_SUITE_REGISTRATION(TestCP);

using namespace std;

static hetm_pc_s *pc;

typedef struct spam_ {
	int from;
	int counter;
} spam_s;

typedef struct spam_args_ {
	int nbThreads;
	int nbSpamObjs;
	int *matrix;
} spam_args_s;

static void spamPC(int id, void *argsPtr);
static void empty(int id, void *argsPtr);
static void emptyExit(int id, void *argsPtr);

#define PC_SIZE 0x80000

TestCP::TestCP()  { }
TestCP::~TestCP() { }

void TestCP::setUp()
{
	pc = hetm_pc_init(PC_SIZE);
}

void TestCP::tearDown()
{
	hetm_pc_destroy(pc);
}

static void testTemplate(int nbSpamObjs, int nbThreads)
{
	int i, j, matrixSize = nbSpamObjs*nbThreads;
	spam_args_s args;
	int *matrix;

	matrix = (int*)malloc(sizeof(int)*matrixSize);

	args.nbThreads = nbThreads;
	args.nbSpamObjs = nbSpamObjs;
	args.matrix = matrix;

	HeTM_init((HeTM_init_s){
    .policy       = HETM_GPU_INV,
    .nbCPUThreads = nbThreads,
    .nbGPUBlocks  = 1,
    .nbGPUThreads = 1,
    .isCPUEnabled = 1,
    .isGPUEnabled = 0
  });

	HeTM_start(spamPC, NULL, &args);
	HeTM_join_CPU_threads();
	// printf("after join threads\n");

	HeTM_destroy();

	for (i = 0; i < nbThreads; ++i) {
		for (j = 0; j < nbSpamObjs; ++j) {
			int res = matrix[i*nbSpamObjs + j];
			char msg[512];
			sprintf(msg, "Expected %i but got %i", j, res);

			CPPUNIT_ASSERT_MESSAGE(msg, res == j);
		}
	}

	free(matrix);
}

void TestCP::TestMultipleRuns()
{
	printf("%s: init\n", __func__);

	const int nbThreads = 4;
	HeTM_init((HeTM_init_s){
    .policy       = HETM_GPU_INV,
    .nbCPUThreads = nbThreads,
    .nbGPUBlocks  = 1,
    .nbGPUThreads = 1,
    .isCPUEnabled = 1,
    .isGPUEnabled = 0
  });

	HeTM_start(empty, NULL, NULL);
	HeTM_set_is_stop(1); // only one exit is allowed
	HeTM_join_CPU_threads();
	HeTM_destroy();

	CPPUNIT_ASSERT(true);
}

void TestCP::TestMultipleRuns2()
{
	printf("%s: init\n", __func__);

	const int nbThreads = 4;
	HeTM_init_s init = {
    .policy       = HETM_GPU_INV,
    .nbCPUThreads = nbThreads,
    .nbGPUBlocks  = 1,
    .nbGPUThreads = 1,
    .isCPUEnabled = 1,
    .isGPUEnabled = 0
  };

	HeTM_init(init);
	HeTM_start(emptyExit, NULL, NULL);
	HeTM_join_CPU_threads();
	HeTM_destroy();

	HeTM_init(init);
	HeTM_start(emptyExit, NULL, NULL);
	HeTM_join_CPU_threads();
	HeTM_destroy();

	HeTM_init(init);
	HeTM_start(emptyExit, NULL, NULL);
	HeTM_join_CPU_threads();
	HeTM_destroy();

	CPPUNIT_ASSERT(true);
}

void TestCP::TestLight()
{
	testTemplate(128, 4);
}

void TestCP::TestMedium()
{
	testTemplate(256, 16);
}

void TestCP::TestHeavy()
{
	testTemplate(1024, 32);
}

static void spamPC(int id, void *argsPtr)
{
	int i;
	spam_args_s *args = (spam_args_s*)argsPtr;
	int nbSpamObjs = args->nbSpamObjs;
	spam_s *array = (spam_s*)malloc(sizeof(spam_s)*nbSpamObjs);
	int counter = 0;

	for (i = 0; i < nbSpamObjs; ++i) {
		array[i].counter = counter;
		array[i].from = HeTM_thread_data[0]->id;
		// printf("[%i] sent %i\n", id, counter);
		counter++;
		hetm_pc_produce(pc, &array[i]);
	}

	for (i = 0; i < nbSpamObjs; ++i) {
		spam_s *res;
		hetm_pc_consume(pc, (void**)&res);
		// printf("matrix[%i]=%i\n", res->from*args->nbSpamObjs + res->counter, res->counter);
		args->matrix[res->from*args->nbSpamObjs + res->counter] = res->counter;
	}
	HeTM_sync_barrier(0);
	HeTM_set_is_stop(1);
	free(array);
}

static void empty(int id, void *argsPtr) { }
static void emptyExit(int id, void *argsPtr) { HeTM_set_is_stop(1); }
