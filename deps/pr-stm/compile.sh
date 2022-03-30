#!/bin/bash

# install cppunit from here https://github.com/blytkerchan/cppunit.git

rm -rf *._
rm -rf ./src/._*
rm -rf ./include/._*
rm -rf ./tests/._*

### if you prefer cmake to generate the Makefile
#rm -rf build
#mkdir build
#cd build

CPPUNIT_DIR=~/libs/cppunit
CUDA_UTIL_DIR=~/projs/cuda-utils
PATH=$PATH:/usr/local/cuda/bin

#cmake ../cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug \
#  -DCPPUNIT_DIR=$CPPUNIT_DIR -DCUDA_UTIL_DIR=$CUDA_UTIL_DIR

make clean
make
