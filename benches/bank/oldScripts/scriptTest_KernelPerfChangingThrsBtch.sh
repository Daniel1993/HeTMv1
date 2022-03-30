#!/bin/bash

GPU_PART="0.67"
CPU_PART="0.66"
P_INTERSECT="0.0" # 0.10 0.30 0.5 0.8 1.0"
DATASET="10000000"
GPU_THREADS="4 8 16 32 64 128 256 512 1024"
BLOCKS="16 20 32 64 128 256"
BATCH_SIZE="1 2 4 8 16 32 64 128 256"
DATASET_SIZE="10000000 100000000 200000000 400000000 600000000"
THR_GPU_THREADS="4 1024"
THR_BLOCKS="4 1024"
THR_BATCH_SIZE="1 256"

DEFAULT_BATCH="1"
DEFAULT_GPU_THREADS="512"
DEFAULT_BLOCKS="20"

#"1 2 4 6 8 10 12 14 16 22 30 42 56"
THREADS="14"
DURATION=10000
SAMPLES=1

rm -f Bank.csv

### TODO: comment this
# sleep 20m

for s in `seq $SAMPLES`
do
	##################################
	# GPU-only
	make clean ; make CMP_TYPE=DISABLED GPU_PART=1.0 CPU_PART=0.0 \
		P_INTERSECT=0.0 CPUEn=0 USE_TSX_IMPL=0 BANK_PART=1 -j 14
	for b in $BLOCKS
	do
		for a in $DATASET_SIZE
		do
			timeout 20s ./bank -a $a -d $DURATION -b $b -x 1024 -T 1
			timeout 20s ./bank -a $a -d $DURATION -b $b -x 512 -T 2
			timeout 20s ./bank -a $a -d $DURATION -b $b -x 256 -T 4
			timeout 20s ./bank -a $a -d $DURATION -b $b -x 128 -T 8
			timeout 20s ./bank -a $a -d $DURATION -b $b -x 64 -T 16
			timeout 20s ./bank -a $a -d $DURATION -b $b -x 32 -T 32
			timeout 20s ./bank -a $a -d $DURATION -b $b -x 16 -T 64
			mv Bank.csv GPU_NO_HACKS_${a}acc_${b}blks_s${s}
		done
	done

	##################################
done
