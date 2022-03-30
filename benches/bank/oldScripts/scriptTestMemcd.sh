#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1
filename_tsx="Bank_TSX"
filename_tiny="Bank_Tiny"

GPU_PART="1.0"
CPU_PART="0.0"
P_INTERSECT="0.0"
DURATION=6000
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
	timeout 30s ./bank -n $CPU_THREADS -l 16 -b 64   -x 512 -T 16 -a $SMALL_DATASET -d $DURATION -N 1 -S 0
	timeout 30s ./bank -n $CPU_THREADS -l 16 -b 128  -x 512 -T 8  -a $SMALL_DATASET -d $DURATION -N 1 -S 0
	timeout 30s ./bank -n $CPU_THREADS -l 16 -b 256  -x 512 -T 4  -a $SMALL_DATASET -d $DURATION -N 1 -S 0
	timeout 30s ./bank -n $CPU_THREADS -l 16 -b 512  -x 512 -T 2  -a $SMALL_DATASET -d $DURATION -N 1 -S 0
	timeout 30s ./bank -n $CPU_THREADS -l 16 -b 640  -x 512 -T 2  -a $SMALL_DATASET -d $DURATION -N 1 -S 0
	timeout 30s ./bank -n $CPU_THREADS -l 16 -b 768  -x 512 -T 2  -a $SMALL_DATASET -d $DURATION -N 1 -S 0
	timeout 30s ./bank -n $CPU_THREADS -l 16 -b 896  -x 512 -T 1  -a $SMALL_DATASET -d $DURATION -N 1 -S 0
	timeout 30s ./bank -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1  -a $SMALL_DATASET -d $DURATION -N 1 -S 0
}

function doRunLargeDTST {
	timeout 30s ./bank -n $CPU_THREADS -l 16 -b 64   -T 16 -x 512 -a $LARGE_DATASET -d $DURATION -N 1 -S 0
	timeout 30s ./bank -n $CPU_THREADS -l 16 -b 128  -T 8  -x 512 -a $LARGE_DATASET -d $DURATION -N 1 -S 0
	timeout 30s ./bank -n $CPU_THREADS -l 16 -b 256  -T 4  -x 512 -a $LARGE_DATASET -d $DURATION -N 1 -S 0
	timeout 30s ./bank -n $CPU_THREADS -l 16 -b 512  -T 2  -x 512 -a $LARGE_DATASET -d $DURATION -N 1 -S 0
	timeout 30s ./bank -n $CPU_THREADS -l 16 -b 640  -T 2  -x 512 -a $LARGE_DATASET -d $DURATION -N 1 -S 0
	timeout 30s ./bank -n $CPU_THREADS -l 16 -b 768  -T 2  -x 512 -a $LARGE_DATASET -d $DURATION -N 1 -S 0
	timeout 30s ./bank -n $CPU_THREADS -l 16 -b 896  -T 1  -x 512 -a $LARGE_DATASET -d $DURATION -N 1 -S 0
	timeout 30s ./bank -n $CPU_THREADS -l 16 -b 1024 -T 1  -x 512 -a $LARGE_DATASET -d $DURATION -N 1 -S 0
}

make clean ; make CMP_TYPE=DISABLED BANK_PART=1 CPUEn=0 PROFILE=1 PR_MAX_RWSET_SIZE=4 -j 14 BENCH=MEMCD
for s in `seq $SAMPLES`
do
	doRunSmallDTST
	mv Bank.csv GPUonly_small_s${s}
	doRunLargeDTST
	mv Bank.csv GPUonly_large_s${s}
done

make clean ; make CMP_TYPE=DISABLED BANK_PART=1 USE_TSX_IMPL=1 INST_CPU=0 GPUEn=0 \
	PROFILE=1 PR_MAX_RWSET_SIZE=4 -j 14 BENCH=MEMCD
for s in `seq $SAMPLES`
do
	# CPU_THREADS=$LOW_CPU_THREADS
	# doRunSmallDTST
	# mv Bank.csv CPUonly_low_THRS_small_s${s}
	CPU_THREADS=$SMALL_HIGH_CPU_THREADS
	doRunSmallDTST
	mv Bank.csv CPUonly_high_THRS_small_s${s}
	CPU_THREADS=$SMALL_VERY_HIGH_CPU_THREADS
	doRunSmallDTST
	mv Bank.csv CPUonly_very_high_THRS_small_s${s}

	# CPU_THREADS=$LOW_CPU_THREADS
	# doRunLargeDTST
	# mv Bank.csv CPUonly_low_THRS_large_s${s}
	CPU_THREADS=$LARGE_HIGH_CPU_THREADS
	doRunLargeDTST
	mv Bank.csv CPUonly_high_THRS_large_s${s}
	CPU_THREADS=$LARGE_VERY_HIGH_CPU_THREADS
	doRunLargeDTST
	mv Bank.csv CPUonly_very_high_THRS_large_s${s}
done

function RunExperimentHighThreads {
	# solution is in $1
	# prob intersect is in $2
	make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 \
		PR_MAX_RWSET_SIZE=4 LOG_TYPE=$1 USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD
	for s in `seq $SAMPLES`
	do
		CPU_THREADS=$SMALL_HIGH_CPU_THREADS
		doRunSmallDTST
		mv Bank.csv ${1}_${2}_high_THRS_small_s${s}

		CPU_THREADS=$LARGE_HIGH_CPU_THREADS
		doRunLargeDTST
		mv Bank.csv ${1}_${2}_high_THRS_large_s${s}
	done
}

function RunExperimentVeryHighThreads {
	# solution is in $1
	make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 \
		PR_MAX_RWSET_SIZE=4 LOG_TYPE=$1 USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD
	for s in `seq $SAMPLES`
	do
		CPU_THREADS=$SMALL_VERY_HIGH_CPU_THREADS
		doRunSmallDTST
		mv Bank.csv ${1}_${2}_very_high_THRS_small_s${s}

		CPU_THREADS=$LARGE_VERY_HIGH_CPU_THREADS
		doRunLargeDTST
		mv Bank.csv ${1}_${2}_very_high_THRS_large_s${s}
	done
}

function RunExperimentOverlapHighThreads {
	# solution is in $1
	make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 \
		PR_MAX_RWSET_SIZE=4 LOG_TYPE=$1 USE_TSX_IMPL=1 OVERLAP_CPY_BACK=1 \
		PROFILE=1 -j 14 BENCH=MEMCD
	for s in `seq $SAMPLES`
	do
		CPU_THREADS=$SMALL_HIGH_CPU_THREADS
		doRunSmallDTST
		mv Bank.csv ${1}_${2}_overlap_high_THRS_small_s${s}

		CPU_THREADS=$LARGE_HIGH_CPU_THREADS
		doRunLargeDTST
		mv Bank.csv ${1}_${2}_overlap_high_THRS_large_s${s}
	done
}

function RunExperimentOverlapVeryHighThreads {
	# solution is in $1
	make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 \
		PR_MAX_RWSET_SIZE=4 LOG_TYPE=$1 USE_TSX_IMPL=1 OVERLAP_CPY_BACK=1 \
		PROFILE=1 -j 14 BENCH=MEMCD
	for s in `seq $SAMPLES`
	do
		CPU_THREADS=$SMALL_VERY_HIGH_CPU_THREADS
		doRunSmallDTST
		mv Bank.csv ${1}_${2}_overlap_very_high_THRS_small_s${s}

		CPU_THREADS=$LARGE_VERY_HIGH_CPU_THREADS
		doRunLargeDTST
		mv Bank.csv ${1}_${2}_overlap_very_high_THRS_large_s${s}
	done
}

RunExperimentHighThreads VERS 0.0
RunExperimentVeryHighThreads VERS 0.0
# RunExperimentOverlapHighThreads VERS 0.0
# RunExperimentOverlapVeryHighThreads VERS 0.0

# RunExperimentHighThreads ADDR 0.0
# RunExperimentVeryHighThreads ADDR 0.0
# RunExperimentOverlapHighThreads ADDR 0.0
# RunExperimentOverlapVeryHighThreads ADDR 0.0

RunExperimentHighThreads BMAP 0.0
RunExperimentVeryHighThreads BMAP 0.0
# RunExperimentOverlapHighThreads BMAP 0.0
# RunExperimentOverlapVeryHighThreads BMAP 0.0

###############################################################################
###############################################################################
###############################################################################
### with conflicts

# RunExperimentHighThreads VERS 0.1
# RunExperimentVeryHighThreads VERS 0.1
# RunExperimentOverlapHighThreads VERS 0.1
# RunExperimentOverlapVeryHighThreads VERS 0.1

# RunExperimentHighThreads ADDR 0.1
# RunExperimentVeryHighThreads ADDR 0.1
# RunExperimentOverlapHighThreads ADDR 0.1
# RunExperimentOverlapVeryHighThreads ADDR 0.1

# RunExperimentHighThreads BMAP 0.1
# RunExperimentVeryHighThreads BMAP 0.1
# RunExperimentOverlapHighThreads BMAP 0.1
# RunExperimentOverlapVeryHighThreads BMAP 0.1

###############################################################################

# RunExperimentHighThreads VERS 0.5
# RunExperimentVeryHighThreads VERS 0.5
# RunExperimentOverlapHighThreads VERS 0.5
# RunExperimentOverlapVeryHighThreads VERS 0.5

# RunExperimentHighThreads ADDR 0.5
# RunExperimentVeryHighThreads ADDR 0.5
# RunExperimentOverlapHighThreads ADDR 0.5
# RunExperimentOverlapVeryHighThreads ADDR 0.5

# RunExperimentHighThreads BMAP 0.5
# RunExperimentVeryHighThreads BMAP 0.5
# RunExperimentOverlapHighThreads BMAP 0.5
# RunExperimentOverlapVeryHighThreads BMAP 0.5

###############################################################################

# RunExperimentHighThreads VERS 0.9
# RunExperimentVeryHighThreads VERS 0.9
# RunExperimentOverlapHighThreads VERS 0.9
# RunExperimentOverlapVeryHighThreads VERS 0.9

# RunExperimentHighThreads ADDR 0.9
# RunExperimentVeryHighThreads ADDR 0.9
# RunExperimentOverlapHighThreads ADDR 0.9
# RunExperimentOverlapVeryHighThreads ADDR 0.9

# RunExperimentHighThreads BMAP 0.9
# RunExperimentVeryHighThreads BMAP 0.9
# RunExperimentOverlapHighThreads BMAP 0.9
# RunExperimentOverlapVeryHighThreads BMAP 0.9

###############################################################################

# RunExperimentHighThreads VERS 1.0
# RunExperimentVeryHighThreads VERS 1.0
# RunExperimentOverlapHighThreads VERS 1.0
# RunExperimentOverlapVeryHighThreads VERS 1.0

# RunExperimentHighThreads ADDR 1.0
# RunExperimentVeryHighThreads ADDR 1.0
# RunExperimentOverlapHighThreads ADDR 1.0
# RunExperimentOverlapVeryHighThreads ADDR 1.0

# RunExperimentHighThreads BMAP 1.0
# RunExperimentVeryHighThreads BMAP 1.0
# RunExperimentOverlapHighThreads BMAP 1.0
# RunExperimentOverlapVeryHighThreads BMAP 1.0
