#!/bin/bash

PROG=./bank
# THRS="1 2 4 8 16 32 64 96 128 192 256 512 1024"
THRS="16 32 64 128 256 512"
BLKS="1 2 4 8 16 32 64 96 128 192 256 512 1024"
# default is 2621440==10M
# TODO
ACCS="4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432"
DATASET=2621440
SAMPLES=5

rm -f stats.txt

make clean ; make -j4

STATS_FILE_NAME="PR-STM_16blks"
make clean ; make NB_ACCOUNTS=$DATASET -j4
b=16
for s in `seq $SAMPLES`
do
  for t in $THRS
  do
    $PROG $b $t
  done
  mv stats.txt "${STATS_FILE_NAME}_s${s}"
done

# STATS_FILE_NAME="Threads32"
# for s in `seq $SAMPLES`
# do
#   t=32
#   for b in $BLKS
#   do
#     $PROG $b $t
#   done
#   mv stats.txt "${STATS_FILE_NAME}_s${s}"
# done

# STATS_FILE_NAME="NbAccounts_128B_64T"
# for s in `seq $SAMPLES`
# do
#   t=64
#   b=128
#   for a in $ACCS
#   do
#     make clean ; make NB_ACCOUNTS=$a -j4
#     $PROG $b $t
#   done
#   mv stats.txt "${STATS_FILE_NAME}_s${s}"
# done
