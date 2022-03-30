#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1
filename_tsx="Bank_TSX"
filename_tiny="Bank_Tiny"

GPU_PART="0.8"
CPU_PART="0.8"
P_INTERSECT="0.001"
DATASET="1000000" # 90 000 000 is the max for my home machine
BLOCKS="256"
GPU_THREADS="128"
BATCH_SIZE="50 100 300"
THREADS="1 2 4 6 8 10 12 14 16 22 30 42 56"
DURATION=5000
SAMPLES=5
#./makeTM.sh

rm -f Bank.csv

for s in `seq $SAMPLES`
do
	for b in $BATCH_SIZE
	do
		# TSX
		make clean ; make HETM_CMP_TYPE=COMPRESSED GPU_PART=$GPU_PART CPU_PART=$CPU_PART P_INTERSECT=$P_INTERSECT USE_TSX_IMPL=1 -j 14
		for t in $THREADS
		do
			./bank -n $THREADS -a $DATASET -b $BLOCKS -x $GPU_THREADS -d $DURATION -T $b
		done
		mv Bank.csv HeTM_TSX_COMPRESSED_T${b}_s${s}.csv

		# TSX --> erased
		# make clean ; make HETM_CMP_TYPE=EXPLICIT GPU_PART=$GPU_PART CPU_PART=$CPU_PART P_INTERSECT=$P_INTERSECT USE_TSX_IMPL=1 -j 14
		# for t in $THREADS
		# do
		# 	./bank -n $THREADS -a $DATASET -b $BLOCKS -x $GPU_THREADS -d $DURATION -T $b
		# done
		# mv Bank.csv HeTM_TSX_EXPLICIT_T${b}_s${s}.csv

		# Tiny
		make clean ; make HETM_CMP_TYPE=COMPRESSED GPU_PART=$GPU_PART CPU_PART=$CPU_PART P_INTERSECT=$P_INTERSECT USE_TSX_IMPL=0 -j 14
		for t in $THREADS
		do
			./bank -n $t -a $DATASET -b $BLOCKS -x $GPU_THREADS -d $DURATION -T $b
		done
		mv Bank.csv HeTM_Tiny_COMPRESSED_T${b}_s${s}.csv

		# Tiny --> erased
		# make clean ; make HETM_CMP_TYPE=EXPLICIT GPU_PART=$GPU_PART CPU_PART=$CPU_PART P_INTERSECT=$P_INTERSECT USE_TSX_IMPL=0 -j 14
		# for t in $THREADS
		# do
		# 	./bank -n $t -a $DATASET -b $BLOCKS -x $GPU_THREADS -d $DURATION -T $b
		# done
		# mv Bank.csv HeTM_Tiny_EXPLICIT_T${b}_s${s}.csv
	done
done
