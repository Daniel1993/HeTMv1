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

class TestCP : public CPPUNIT_NS::TestFixture
{
	CPPUNIT_TEST_SUITE(TestCP);
	CPPUNIT_TEST(TestMultipleRuns);
	CPPUNIT_TEST(TestMultipleRuns2);
	CPPUNIT_TEST(TestLight);
	CPPUNIT_TEST(TestMedium);
	CPPUNIT_TEST(TestHeavy);
	CPPUNIT_TEST_SUITE_END();

public:
	TestCP();
	virtual ~TestCP();
	void setUp();
	void tearDown();

private:
	void TestMultipleRuns();
	void TestMultipleRuns2();
	void TestLight();
	void TestMedium();
	void TestHeavy();
} ;

#endif /* TEST_BANK_H */
