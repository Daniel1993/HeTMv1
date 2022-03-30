#ifndef TEST_BANK_H
#define TEST_BANK_H

#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestCase.h>
#include <cppunit/TestFixture.h>
#include <cppunit/ui/text/TextTestRunner.h>
#include <cppunit/XmlOutputter.h>
#include <netinet/in.h>
#include <chrono>

#include "hetm.cuh"

class TestMultiGPU : public CPPUNIT_NS::TestFixture
{
	CPPUNIT_TEST_SUITE(TestMultiGPU);
	CPPUNIT_TEST(TestOnOff);
	CPPUNIT_TEST_SUITE_END();

public:
	TestMultiGPU();
	virtual ~TestMultiGPU();
	void setUp();
	void tearDown();

private:
	void TestOnOff();
} ;

#endif /* TEST_BANK_H */
