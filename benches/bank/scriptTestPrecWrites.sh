#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1
filename_tsx="Bank_TSX"
filename_tiny="Bank_Tiny"

GPU_PART="1.0"
CPU_PART="0.0"
P_INTERSECT="0.0"
DURATION=18000
BLOCKS="2 4 8 16 32 64 256 512 1024" # 512
THREADS="512" #"2 4 8 16 32 64 96 256 320 512 640 768 1024"
BATCH_SIZE="4"
SAMPLES=1
#./makeTM.sh

CPU_THREADS=4
LOW_CPU_THREADS=10
HIGH_CPU_THREADS=20

rm -f Bank.csv

LARGE_DATASET=100000000 #2621440 # 90 000 000 is the max for my home machine
# LARGE_DATASET=20000000 #2621440 # 90 000 000 is the max for my home machine
LARGE_DATASET_P20=61000000 #2621440 # 90 000 000 is the max for my home machine
SMALL_DATASET=1000000 #2621440 # 90 000 000 is the max for my home machine
SMALL_DATASET_P20=122000 #2621440 # 90 000 000 is the max for my home machine

function doRunLargeDTST {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 18 -l 5  -N 1 -T 8 -G $GPU_INPUT -C $CPU_INPUT
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 18 -l 10 -N 1 -T 8 -G $GPU_INPUT -C $CPU_INPUT
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 18 -l 25 -N 1 -T 8 -G $GPU_INPUT -C $CPU_INPUT
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 18 -l 50 -N 1 -T 8 -G $GPU_INPUT -C $CPU_INPUT
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 18 -l 75 -N 1 -T 8 -G $GPU_INPUT -C $CPU_INPUT
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 18 -l 90 -N 1 -T 8 -G $GPU_INPUT -C $CPU_INPUT
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 18 -l 95 -N 1 -T 8 -G $GPU_INPUT -C $CPU_INPUT
		mv Bank.csv ${1}${s}
	done
}

function doRunLargeDTSTZipf {
	for s in `seq 1 $SAMPLES`
	do
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 1   -N 1 -T 8 -G $GPU_INPUT -C $CPU_INPUT
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 4   -N 1 -T 8 -G $GPU_INPUT -C $CPU_INPUT
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 8   -N 1 -T 8 -G $GPU_INPUT -C $CPU_INPUT
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 10  -N 1 -T 8 -G $GPU_INPUT -C $CPU_INPUT
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 25  -N 1 -T 8 -G $GPU_INPUT -C $CPU_INPUT
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 50  -N 1 -T 8 -G $GPU_INPUT -C $CPU_INPUT
		# timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 75  -N 1 -T 8 -G $GPU_INPUT -C $CPU_INPUT
		timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 90  -N 1 -T 8 -G $GPU_INPUT -C $CPU_INPUT
		# timeout 30s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 95  -N 1 -T 8 -G $GPU_INPUT -C $CPU_INPUT
		mv Bank.csv ${1}${s}
	done
}

### Fixed the amount of CPU threads
CPU_THREADS=8

cd java_zipf
GPU_INPUT="GPU_input_${LARGE_DATASET}_099_4194304.txt"
CPU_INPUT="CPU_input_${LARGE_DATASET}_099_16384.txt"

if [ ! -f ../$GPU_INPUT ]
then
	java Main $LARGE_DATASET 0.99 4194304 > ../$GPU_INPUT
fi
if [ ! -f ../$CPU_INPUT ]
then
	java Main $LARGE_DATASET 0.99 16384 > ../$CPU_INPUT
fi
cd ..

############### GPU-only
make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 CPUEn=0 BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTSTZipf GPUonly_zipf_sep_BMAP_LARGE_s

make clean ; make CMP_TYPE=DISABLED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTSTZipf GPUonly_zipf_sep_DISABLED_LARGE_s

# ############## CPU-only (HTM)
make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTSTZipf CPUonly_zipf_sep_DISABLED_LARGE_s

make clean ; make LOG_TYPE=VERS INST_CPU=1 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTSTZipf CPUonly_zipf_sep_VERS_LARGE_s

make clean ; make LOG_TYPE=BMAP INST_CPU=1 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTSTZipf CPUonly_zipf_sep_BMAP_LARGE_s

# ############## CPU-only (TinySTM)
make clean ; make INST_CPU=0 GPUEn=0 PR_MAX_RWSET_SIZE=20 BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTSTZipf CPUonly_zipf_sep_DISABLED_STM_LARGE_s

make clean ; make LOG_TYPE=VERS INST_CPU=1 GPUEn=0 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTSTZipf CPUonly_zipf_sep_VERS_SMALL_STM_LARGE_s

make clean ; make LOG_TYPE=BMAP INST_CPU=1 GPUEn=0 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTSTZipf CPUonly_zipf_sep_BMAP_STM_LARGE_s

############## VERS (HTM)
make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTSTZipf VERS_zipf_sep_LARGE_s

make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTSTZipf VERS_zipf_sep_STM_LARGE_s

############## BMAP (HTM)
make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTST BMAP_zipf_sep_LARGE_s

make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTST BMAP_zipf_sep_STM_LARGE_s

######################

LARGE_DATASET=10000000

######################

CPU_THREADS=8 ### TODO: same nb of threads for tiny STM

cd java_zipf
GPU_INPUT="GPU_input_${LARGE_DATASET}_099_67108864.txt"
CPU_INPUT="CPU_input_${LARGE_DATASET}_099_2097152.txt"

if [ ! -f ../$GPU_INPUT ]
then
	java Main $LARGE_DATASET 0.99 67108864 > ../$GPU_INPUT
fi
if [ ! -f ../$CPU_INPUT ]
then
	java Main $LARGE_DATASET 0.99 2097152 > ../$CPU_INPUT
fi
cd ..

############### GPU-only
make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 CPUEn=0 BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTSTZipf GPUonly_zipf_sep_BMAP_SMALL_s

make clean ; make CMP_TYPE=DISABLED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTSTZipf GPUonly_zipf_sep_DISABLED_SMALL_s

# ############## CPU-only (HTM)
make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTSTZipf CPUonly_zipf_sep_DISABLED_SMALL_s

make clean ; make LOG_TYPE=VERS INST_CPU=1 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTSTZipf CPUonly_zipf_sep_VERS_SMALL_s

make clean ; make LOG_TYPE=BMAP INST_CPU=1 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTSTZipf CPUonly_zipf_sep_BMAP_SMALL_s

# ############## CPU-only (TinySTM)
make clean ; make INST_CPU=0 GPUEn=0 PR_MAX_RWSET_SIZE=20 BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTSTZipf CPUonly_zipf_sep_DISABLED_STM_SMALL_s

make clean ; make LOG_TYPE=VERS INST_CPU=1 GPUEn=0 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTSTZipf CPUonly_zipf_sep_VERS_SMALL_STM_SMALL_s

make clean ; make LOG_TYPE=BMAP INST_CPU=1 GPUEn=0 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTSTZipf CPUonly_zipf_sep_BMAP_STM_SMALL_s

############## VERS (HTM)
make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTSTZipf VERS_zipf_sep_SMALL_s

make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTSTZipf VERS_zipf_sep_STM_SMALL_s

############## BMAP (HTM)
make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTST BMAP_zipf_sep_SMALL_s

make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP PR_MAX_RWSET_SIZE=20 \
	BANK_PART=4 PROFILE=1 -j 14
doRunLargeDTST BMAP_zipf_sep_STM_SMALL_s
