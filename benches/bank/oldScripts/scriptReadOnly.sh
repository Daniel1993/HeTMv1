#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1
filename_tsx="Bank_TSX"
filename_tiny="Bank_Tiny"

GPU_PART="1.0"
CPU_PART="0.0"
P_INTERSECT="0.0"
DURATION=8000
BLOCKS="2 4 8 16 32 64 256 512 1024" # 512
THREADS="512" #"2 4 8 16 32 64 96 256 320 512 640 768 1024"
BATCH_SIZE="4"
SAMPLES=3
#./makeTM.sh

CPU_THREADS=4
# LOW_CPU_THREADS=2
LARGE_HIGH_CPU_THREADS=30
LARGE_VERY_HIGH_CPU_THREADS=50
SMALL_HIGH_CPU_THREADS=30
SMALL_VERY_HIGH_CPU_THREADS=50

rm -f Bank.csv

LARGE_DATASET=100000000 #2621440 # 90 000 000 is the max for my home machine
LARGE_DATASET_P21=12100000 #2621440 # 90 000 000 is the max for my home machine
LARGE_DATASET_P300=40000000 #2621440 # 90 000 000 is the max for my home machine
SMALL_DATASET=1000000 #2621440 # 90 000 000 is the max for my home machine
SMALL_DATASET_P21=1210000 #2621440 # 90 000 000 is the max for my home machine
SMALL_DATASET_P300=4000000 #2621440 # 90 000 000 is the max for my home machine

### TODO: find the best combination of parameters
function doRunSmallDTST {
	timeout 12s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $SMALL_DATASET -d $DURATION -T 1 -R 100 -l 0   -S 4
	timeout 12s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $SMALL_DATASET -d $DURATION -T 1 -R 100 -l 10  -S 4
	timeout 12s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $SMALL_DATASET -d $DURATION -T 1 -R 100 -l 25  -S 4
	timeout 12s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $SMALL_DATASET -d $DURATION -T 1 -R 100 -l 50  -S 4
	timeout 12s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $SMALL_DATASET -d $DURATION -T 1 -R 100 -l 75  -S 4
	timeout 12s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $SMALL_DATASET -d $DURATION -T 1 -R 100 -l 90  -S 4
	timeout 12s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $SMALL_DATASET -d $DURATION -T 1 -R 100 -l 100 -S 4
}

function doRunLargeDTST {
	timeout 12s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -T 1 -R 100 -l 0   -S 4
	timeout 12s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -T 1 -R 100 -l 10  -S 4
	timeout 12s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -T 1 -R 100 -l 25  -S 4
	timeout 12s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -T 1 -R 100 -l 50  -S 4
	timeout 12s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -T 1 -R 100 -l 75  -S 4
	timeout 12s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -T 1 -R 100 -l 90  -S 4
	timeout 12s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -T 1 -R 100 -l 100 -S 4
}

make clean ; make CMP_TYPE=DISABLED BANK_PART=1 GPU_PART=1.0 \
	CPU_PART=0.0 P_INTERSECT=$P_INTERSECT CPUEn=0 PROFILE=1 PR_MAX_RWSET_SIZE=20 -j 14
for s in `seq $SAMPLES`
do
	doRunSmallDTST
	mv Bank.csv GPUonly_small_s${s}
	doRunLargeDTST
	mv Bank.csv GPUonly_large_s${s}
done

make clean ; make CMP_TYPE=DISABLED BANK_PART=1 GPU_PART=0.0 \
	CPU_PART=1.0 USE_TSX_IMPL=1 P_INTERSECT=$P_INTERSECT INST_CPU=0 GPUEn=0 \
	PROFILE=1 PR_MAX_RWSET_SIZE=20  -j 14
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
	make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 GPU_PART=0.9 CPU_PART=0.9 \
		PR_MAX_RWSET_SIZE=20 LOG_TYPE=$1 P_INTERSECT=1.0 USE_TSX_IMPL=1 PROFILE=1 -j 14
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
	make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 GPU_PART=0.9 CPU_PART=0.9 \
		PR_MAX_RWSET_SIZE=20 LOG_TYPE=$1 P_INTERSECT=1.0 USE_TSX_IMPL=1 PROFILE=1 -j 14
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
	make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 GPU_PART=0.9 CPU_PART=0.9 \
		PR_MAX_RWSET_SIZE=20 LOG_TYPE=$1 P_INTERSECT=1.0 USE_TSX_IMPL=1 OVERLAP_CPY_BACK=1 \
		PROFILE=1 -j 14
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
	make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 GPU_PART=0.9 CPU_PART=0.9 \
		PR_MAX_RWSET_SIZE=20 LOG_TYPE=$1 P_INTERSECT=1.0 USE_TSX_IMPL=1 OVERLAP_CPY_BACK=1 \
		PROFILE=1 -j 14
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
