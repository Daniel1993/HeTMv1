#!/bin/bash
if [ ! -d ./tbb-2019_U8 ]
then
  wget https://github.com/intel/tbb/archive/2019_U8.tar.gz
  tar -xzf 2019_U8.tar.gz
  cd tbb-2019_U8/
  make -j 4
  cd -

  ### TODO: check if it is available with -ltbb and -ltbbmalloc
fi
