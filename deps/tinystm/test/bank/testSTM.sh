#!/bin/bash

PROG=./bank
THRS="1 2 4 6 8 10 12 14 16 18 20 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56"
ACCS="256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144"
SAMPLES=30

rm -f stats.txt

# tiny
make clean ; make -j4

# STATS_FILE_NAME="TinyThreads_16384A"
# for s in `seq 6 $SAMPLES`
# do
#   a=16384
#   for t in $THRS
#   do
#     $PROG -a $a -n $t
#   done
#   mv BankStats.txt "${STATS_FILE_NAME}_s${s}"
# done

STATS_FILE_NAME="TinyNbAccounts_14T"
for s in `seq $SAMPLES`
do
  t=14
  for a in $ACCS
  do
    $PROG -a $a -n $t
  done
  mv BankStats.txt "${STATS_FILE_NAME}_s${s}"
done

# HTM
# make clean ; make USE_TSX_IMPL=1 -j4
#
# STATS_FILE_NAME="TSXThreads_16384A"
# for s in `seq $SAMPLES`
# do
#   a=16384
#   for t in $THRS
#   do
#     $PROG -a $a -n $t
#   done
#   mv BankStats.txt "${STATS_FILE_NAME}_s${s}"
# done
#
# STATS_FILE_NAME="TSXNbAccounts_14T"
# for s in `seq $SAMPLES`
# do
#   t=14
#   for a in $ACCS
#   do
#     $PROG -a $a -n $t
#   done
#   mv BankStats.txt "${STATS_FILE_NAME}_s${s}"
# done
