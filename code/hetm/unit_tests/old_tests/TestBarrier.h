#ifndef TEST_LOG_H
#define TEST_LOG_H

#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestCase.h>
#include <cppunit/TestFixture.h>
#include <cppunit/ui/text/TextTestRunner.h>
#include <cppunit/XmlOutputter.h>

class TestBarrier : public CPPUNIT_NS::TestFixture
{
	CPPUNIT_TEST_SUITE(TestBarrier);
	// CPPUNIT_TEST(Test4Threads);
	CPPUNIT_TEST_SUITE_END();

public:
	TestBarrier();
	virtual ~TestBarrier();
	void setUp();
	void tearDown();

private:
	void Test4Threads();

};

#endif /* TEST_LOG_H */
