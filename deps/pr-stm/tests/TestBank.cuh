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

class TestBank : public CPPUNIT_NS::TestFixture
{
	CPPUNIT_TEST_SUITE(TestBank);
	CPPUNIT_TEST(TestMacros);
	CPPUNIT_TEST(TestRead);
	CPPUNIT_TEST(TestWrite);
	CPPUNIT_TEST(TestReadWrite);
	CPPUNIT_TEST(TestRandom);
	CPPUNIT_TEST_SUITE_END();

public:
	TestBank();
	virtual ~TestBank();
	void setUp();
	void tearDown();

private:
	void TestMacros();
	void TestRead();
	void TestWrite();
	void TestReadWrite();
	void TestRandom();
} ;

#endif /* TEST_BANK_H */
