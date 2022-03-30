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

class TestBMAP : public CPPUNIT_NS::TestFixture
{
	CPPUNIT_TEST_SUITE(TestBMAP);
	CPPUNIT_TEST(TestAllConflict);
	CPPUNIT_TEST(TestGPUsConflict);
	CPPUNIT_TEST(TestRWDifferentPositions);
	CPPUNIT_TEST(TestWrtDifferentReadSamePositions);
	CPPUNIT_TEST(TestCache);
	CPPUNIT_TEST_SUITE_END();

public:
	TestBMAP();
	virtual ~TestBMAP();
	void setUp();
	void tearDown();

private:
	void TestAllConflict();
	void TestGPUsConflict();
	
	void TestRWDifferentPositions();
	void TestWrtDifferentReadSamePositions();
	void TestCache();
} ;

#endif /* TEST_BANK_H */
