#include "TestBarrier.h"

#include "hetm-utils.h"

#include <thread>

CPPUNIT_TEST_SUITE_REGISTRATION(TestBarrier);

using namespace std;

static ticket_barrier_t barrier;
static int counter[2];

static void testThread();

TestBarrier::TestBarrier()  { }
TestBarrier::~TestBarrier() { }

void TestBarrier::setUp() { }

void TestBarrier::tearDown() { }

void TestBarrier::Test4Threads()
{
  barrier_init(barrier, 2);
  thread t(testThread);
  counter[0]++;
  barrier_cross(barrier);
  CPPUNIT_ASSERT(counter[1] == 1);
  barrier_destroy(barrier);
}

static void testThread()
{
  counter[1]++;
  barrier_cross(barrier);
}
