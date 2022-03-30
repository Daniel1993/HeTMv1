include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)

cmake_minimum_required(VERSION 2.8)

project(pr_stm)

include(${PROJECT_SOURCE_DIR}/cmake_utils.txt)
include(${PROJECT_SOURCE_DIR}/cmake_deb.txt)

file(GLOB_RECURSE pr_stm_SOURCES
  ABSOLUTE "${PROJECT_SOURCE_DIR}/src/"
  "${PROJECT_SOURCE_DIR}/src/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/*.c"
  "${PROJECT_SOURCE_DIR}/src/*.cu"
)

file(GLOB_RECURSE pr_stm_TESTS
  ABSOLUTE "${PROJECT_SOURCE_DIR}/tests/"
  "${PROJECT_SOURCE_DIR}/tests/*.cpp"
  "${PROJECT_SOURCE_DIR}/tests/*.c"
  "${PROJECT_SOURCE_DIR}/tests/*.cu"
)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

include_directories("${CPPUNIT_DIR}/include")
link_directories("${CPPUNIT_DIR}/src/cppunit/.libs")

include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/tests")

#cuda_add_library(pr_stm ${pr_stm_SOURCES})
cuda_add_executable(bank bank.cu)
cuda_add_executable(test_runner ${pr_stm_TESTS})
#target_link_libraries(bank pr_stm)
#target_link_libraries(test_runner pr_stm)
#TODO: this library is a mess to find
target_link_libraries(test_runner dl)
target_link_libraries(test_runner cppunit)

find_package(Threads)
target_link_libraries(test_runner ${CMAKE_THREAD_LIBS_INIT})
