#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1
filename_tsx="Bank_TSX"
filename_tiny="Bank_Tiny"

GPU_PART="1.0"
CPU_PART="0.0"
P_INTERSECT="0.0"
DURATION=40000
BLOCKS="2 4 8 16 32 64 256 512 1024" # 512
THREADS="512" #"2 4 8 16 32 64 96 256 320 512 640 768 1024"
BATCH_SIZE="4"
SAMPLES=1
#./makeTM.sh

CPU_THREADS=4
# LOW_CPU_THREADS=2
LARGE_HIGH_CPU_THREADS=4
LARGE_VERY_HIGH_CPU_THREADS=8
SMALL_HIGH_CPU_THREADS=4
SMALL_VERY_HIGH_CPU_THREADS=8

rm -f Bank.csv

LARGE_DATASET=32768
SMALL_DATASET=4096

# memcached read only

### TODO: use shared 0, 30, 50

### TODO: find the best combination of parameters
function doRunSmallDTST {
	timeout 50s ./bank -n $CPU_THREADS -l 32 -b 1024 -x 512 -T 1 -a $SMALL_DATASET -d $DURATION -N 0     -S 0
	timeout 50s ./bank -n $CPU_THREADS -l 32 -b 1024 -x 512 -T 1 -a $SMALL_DATASET -d $DURATION -N 0.05  -S 0
	timeout 50s ./bank -n $CPU_THREADS -l 32 -b 1024 -x 512 -T 1 -a $SMALL_DATASET -d $DURATION -N 0.07  -S 0
	timeout 50s ./bank -n $CPU_THREADS -l 32 -b 1024 -x 512 -T 1 -a $SMALL_DATASET -d $DURATION -N 0.09  -S 0
	timeout 50s ./bank -n $CPU_THREADS -l 32 -b 1024 -x 512 -T 1 -a $SMALL_DATASET -d $DURATION -N 0.1   -S 0
	timeout 50s ./bank -n $CPU_THREADS -l 32 -b 1024 -x 512 -T 1 -a $SMALL_DATASET -d $DURATION -N 0.11  -S 0
	timeout 50s ./bank -n $CPU_THREADS -l 32 -b 1024 -x 512 -T 1 -a $SMALL_DATASET -d $DURATION -N 0.12  -S 0
	timeout 50s ./bank -n $CPU_THREADS -l 32 -b 1024 -x 512 -T 1 -a $SMALL_DATASET -d $DURATION -N 0.13  -S 0
	timeout 50s ./bank -n $CPU_THREADS -l 32 -b 1024 -x 512 -T 1 -a $SMALL_DATASET -d $DURATION -N 0.2   -S 0
}

CPU_THREADS=14

make clean ; make CMP_TYPE=DISABLED BANK_PART=1 CPUEn=0 PROFILE=1 PR_MAX_RWSET_SIZE=4 -j 14 BENCH=MEMCD
for s in `seq $SAMPLES`
do
	doRunSmallDTST
	mv Bank.csv memcd_GPUonly_s${s}
done

make clean ; make CMP_TYPE=DISABLED BANK_PART=1 USE_TSX_IMPL=1 INST_CPU=0 GPUEn=0 \
	PROFILE=1 PR_MAX_RWSET_SIZE=4 -j 14 BENCH=MEMCD
for s in `seq $SAMPLES`
do
	doRunSmallDTST
	mv Bank.csv memcd_CPUonly_s${s}
done

function RunExperiment {
	# solution is in $1
	# prob intersect is in $2
	make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 \
		PR_MAX_RWSET_SIZE=4 LOG_TYPE=$1 USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD
	for s in `seq $SAMPLES`
	do
		doRunSmallDTST
		mv Bank.csv memcd_${1}_${2}_s${s}
	done
}

RunExperiment VERS 0.0
RunExperiment BMAP 0.0
