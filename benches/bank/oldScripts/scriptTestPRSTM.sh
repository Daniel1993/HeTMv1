#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1

GPU_PART="1.0"
CPU_PART="0.0"
P_INTERSECT="0.0"
DURATION=5000
DATASET="100000 500000 1000000 5000000 10000000 50000000" #2621440 # 90 000 000 is the max for my home machine
BLOCKS="2 4 8 16 32 64 128 256 512 1024" # 512
THREADS="2 4 8 16 32 64 128 256 512 1024"
BATCH_SIZE="1 2 4 8 16 32 64 128 256 512 1024"
DEFAULT_BATCH_SIZE="2"
DEFAULT_DATASET="10000000"
DEFAULT_BLOCKS="512"
DEFAULT_THREADS="512"
SAMPLES=1
#./makeTM.sh

rm -f Bank.csv

make clean ; make CMP_TYPE=DISABLED GPU_PART=$GPU_PART CPU_PART=$CPU_PART P_INTERSECT=$P_INTERSECT CPUEn=0 DISABLE_PRSTM=1 -j 14
for s in `seq $SAMPLES`
do
	a=$DEFAULT_DATASET
	b=$DEFAULT_BLOCKS
	T=$DEFAULT_BATCH_SIZE
	for t in $THREADS
	do
		timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -T $T
	done
	mv Bank.csv GPU_THRS_s${s}
	a=$DEFAULT_DATASET
	t=$DEFAULT_THREADS
	T=$DEFAULT_BATCH_SIZE
	for b in $BLOCKS
	do
		timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -T $T
	done
	mv Bank.csv GPU_BLKS_s${s}
	b=$DEFAULT_BLOCKS
	t=$DEFAULT_THREADS
	T=$DEFAULT_BATCH_SIZE
	for a in $DATASET
	do
		timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -T $T
	done
	mv Bank.csv GPU_DTST_s${s}
	a=$DEFAULT_DATASET
	b=$DEFAULT_BLOCKS
	t=$DEFAULT_THREADS
	for T in $BATCH_SIZE
	do
		timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -T $T
	done
	mv Bank.csv GPU_BTCH_s${s}
done

make clean ; make CMP_TYPE=DISABLED GPU_PART=$GPU_PART CPU_PART=$CPU_PART P_INTERSECT=$P_INTERSECT CPUEn=0 DISABLE_PRSTM=0 -j 14
for s in `seq $SAMPLES`
do
	a=$DEFAULT_DATASET
	b=$DEFAULT_BLOCKS
	T=$DEFAULT_BATCH_SIZE
	for t in $THREADS
	do
		timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -T $T
	done
	mv Bank.csv PRSTM_THRS_s${s}
	a=$DEFAULT_DATASET
	t=$DEFAULT_THREADS
	T=$DEFAULT_BATCH_SIZE
	for b in $BLOCKS
	do
		timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -T $T
	done
	mv Bank.csv PRSTM_BLKS_s${s}
	b=$DEFAULT_BLOCKS
	t=$DEFAULT_THREADS
	T=$DEFAULT_BATCH_SIZE
	for a in $DATASET
	do
		timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -T $T
	done
	mv Bank.csv PRSTM_DTST_s${s}
	a=$DEFAULT_DATASET
	b=$DEFAULT_BLOCKS
	t=$DEFAULT_THREADS
	for T in $BATCH_SIZE
	do
		timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -T $T
	done
	mv Bank.csv PRSTM_BTCH_s${s}
done
