#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1
filename_tsx="Bank_TSX"
filename_tiny="Bank_Tiny"

GPU_PART="0.5"
CPU_PART="0.5"
P_INTERSECT="0.0"

DATASET="10000000"
BLOCKS="512"
GPU_THREADS="512"
BATCH_SIZE="2"

# THREADS="1 2 4 6 8 10 12 14 16 22 30 42 56"
THREADS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 16 22 30 42 56"
DURATION=10000
SAMPLES=5
#./makeTM.sh

rm -f Bank.csv

for s in `seq $SAMPLES`
do
	# TSX
	make clean ; make CMP_TYPE=COMPRESSED GPU_PART=$GPU_PART CPU_PART=$CPU_PART \
		P_INTERSECT=$P_INTERSECT CPU_INV=0 USE_TSX_IMPL=1 BANK_PART=1 -j 14
	for t in $THREADS
	do
		timeout 20s ./bank -n $t -a $DATASET -d $DURATION -x $GPU_THREADS -b $BLOCKS -T $BATCH_SIZE
	done
	mv Bank.csv HeTM_TSX_s${s}.csv

	# Tiny
	make clean ; make CMP_TYPE=COMPRESSED GPU_PART=$GPU_PART CPU_PART=$CPU_PART \
		P_INTERSECT=$P_INTERSECT CPU_INV=0 USE_TSX_IMPL=0 BANK_PART=1 -j 14
	for t in $THREADS
	do
		timeout 20s ./bank -n $t -a $DATASET -d $DURATION -x $GPU_THREADS -b $BLOCKS -T $BATCH_SIZE
	done
	mv Bank.csv HeTM_Tiny_s${s}.csv
done
