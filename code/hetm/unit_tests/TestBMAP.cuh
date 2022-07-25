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
	CPPUNIT_TEST(TestWrtDifferentReadSamePositions);
	CPPUNIT_TEST(TestRWDifferentPositions2);
	CPPUNIT_TEST(TestRWDifferentPositions);
	CPPUNIT_TEST(TestGPUsConflict);
	CPPUNIT_TEST(TestGPUsConflictNotFirstChunk);
	CPPUNIT_TEST(TestCache);
	CPPUNIT_TEST(TestDisjointChunkWrites);
	CPPUNIT_TEST_SUITE_END();

public:
	TestBMAP();
	virtual ~TestBMAP();
	void setUp();
	void tearDown();

private:
	void TestAllConflict();
	void TestGPUsConflict();
	void TestGPUsConflictNotFirstChunk();
	void TestDisjointChunkWrites();
	
	void TestRWDifferentPositions();
	void TestRWDifferentPositions2();
	void TestWrtDifferentReadSamePositions();
	void TestCache();
} ;

#endif /* TEST_BANK_H */
