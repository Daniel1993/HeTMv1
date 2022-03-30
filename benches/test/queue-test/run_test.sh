#!/bin/bash

NB_SAMPLES=5

cd bank-wo-queues
rm bank_stats.csv

# make clean ; make CPU_FREQ=1800000 USE_HTM=1 USE_WORKLOAD=1
# for s in $(seq $NB_SAMPLES)
# do
#   for i in 10 20 30 40 50 60 70 80 90 ; do ./bank -n 8 -r $i -d 20000 -a 40000000 ; done
#   mv bank_stats.csv wo_TSX_W1_s$s.csv
# done
#
# make clean ; make CPU_FREQ=1800000 USE_HTM=0 USE_WORKLOAD=1
# for s in $(seq $NB_SAMPLES)
# do
#   for i in 10 20 30 40 50 60 70 80 90 ; do ./bank -n 8 -r $i -d 20000 -a 40000000 ; done
#   mv bank_stats.csv wo_TinySTM_W1_s$s.csv
# done
#
# make clean ; make CPU_FREQ=1800000 USE_HTM=1 USE_WORKLOAD=2
# for s in $(seq $NB_SAMPLES)
# do
#   for i in 10 20 30 40 50 60 70 80 90 ; do ./bank -n 8 -r $i -d 20000 -a 40000000 ; done
#   mv bank_stats.csv wo_TSX_W2_s$s.csv
# done
#
# make clean ; make CPU_FREQ=1800000 USE_HTM=0 USE_WORKLOAD=2
# for s in $(seq $NB_SAMPLES)
# do
#   for i in 10 20 30 40 50 60 70 80 90 ; do ./bank -n 8 -r $i -d 20000 -a 40000000 ; done
#   mv bank_stats.csv wo_TinySTM_W2_s$s.csv
# done

###

cd ../bank-wi-queues
rm bank_stats.csv

make clean ; make CPU_FREQ=1800000 USE_HTM=1 USE_WORKLOAD=1
for s in $(seq $NB_SAMPLES)
do
  for i in 10 20 30 40 50 60 70 80 90 ; do ./bank -n 8 -i 6 -r $i -d 10000 -a 40000000 ; done
  mv bank_stats.csv wi_TSX_W1_s$s.csv
done

make clean ; make CPU_FREQ=1800000 USE_HTM=0 USE_WORKLOAD=1
for s in $(seq $NB_SAMPLES)
do
  for i in 10 20 30 40 50 60 70 80 90 ; do ./bank -n 8 -i 6 -r $i -d 10000 -a 40000000 ; done
  mv bank_stats.csv wi_TinySTM_W1_s$s.csv
done

make clean ; make CPU_FREQ=1800000 USE_HTM=1 USE_WORKLOAD=2
for s in $(seq $NB_SAMPLES)
do
  for i in 10 20 30 40 50 60 70 80 90 ; do ./bank -n 8 -i 6 -r $i -d 10000 -a 40000000 ; done
  mv bank_stats.csv wi_TSX_W2_s$s.csv
done

make clean ; make CPU_FREQ=1800000 USE_HTM=0 USE_WORKLOAD=2
for s in $(seq $NB_SAMPLES)
do
  for i in 10 20 30 40 50 60 70 80 90 ; do ./bank -n 8 -i 6 -r $i -d 10000 -a 40000000 ; done
  mv bank_stats.csv wi_TinySTM_W2_s$s.csv
done
