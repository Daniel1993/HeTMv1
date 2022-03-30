#ifndef TEST_LOG_H
#define TEST_LOG_H

#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestCase.h>
#include <cppunit/TestFixture.h>
#include <cppunit/ui/text/TextTestRunner.h>
#include <cppunit/XmlOutputter.h>

class TestLog : public CPPUNIT_NS::TestFixture
{
	CPPUNIT_TEST_SUITE(TestLog);
	CPPUNIT_TEST(TestChunk);
	CPPUNIT_TEST(TestChunking);
	CPPUNIT_TEST(TestTruncate);
	CPPUNIT_TEST(TestModLog);
	CPPUNIT_TEST_SUITE_END();

public:
	TestLog();
	virtual ~TestLog();
	void setUp();
	void tearDown();

private:
	void TestChunk();
	void TestChunking();
	void TestTruncate();
	void TestModLog();

};

#endif /* TEST_LOG_H */
