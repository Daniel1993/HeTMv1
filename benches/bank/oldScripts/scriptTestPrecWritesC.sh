#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1
filename_tsx="Bank_TSX"
filename_tiny="Bank_Tiny"

GPU_PART="1.0"
CPU_PART="0.0"
P_INTERSECT="0.0"
DURATION=12000
BLOCKS="2 4 8 16 32 64 256 512 1024" # 512
THREADS="512" #"2 4 8 16 32 64 96 256 320 512 640 768 1024"
BATCH_SIZE="4"
SAMPLES=3
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
	for s in `seq $SAMPLES`
	do
		timeout 20s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 18 -l 5  -N 1 -T 8
		timeout 20s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 18 -l 10 -N 1 -T 8
		timeout 20s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 18 -l 25 -N 1 -T 8
		timeout 20s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 18 -l 50 -N 1 -T 8
		timeout 20s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 18 -l 75 -N 1 -T 8
		timeout 20s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 18 -l 90 -N 1 -T 8
		timeout 20s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 18 -l 95 -N 1 -T 8
		mv Bank.csv ${1}${s}
	done
}

function doRunLargeDTSTZipf {
	for s in `seq $SAMPLES`
	do
		timeout 20s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 1 -l 5   -N 1 -T 8
		timeout 20s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 1 -l 10  -N 1 -T 8
		timeout 20s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 1 -l 25  -N 1 -T 8
		timeout 20s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 1 -l 50  -N 1 -T 8
		timeout 20s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 1 -l 75  -N 1 -T 8
		timeout 20s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 1 -l 90  -N 1 -T 8
		timeout 20s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 1 -l 95  -N 1 -T 8
		mv Bank.csv ${1}${s}
	done
}

### Fixed the amount of CPU threads
CPU_THREADS=8

cd java_zipf
# GPU_THREADS * GPU_BLOCKS * T
java Main $LARGE_DATASET 0.99 4194304 > ../GPU_input.txt
# NB_THREADS * 2048
java Main $LARGE_DATASET 0.99 16384 > ../CPU_input.txt
cd ..

############### GPU-only
make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 GPU_PART=0.55 \
	CPU_PART=0.55 P_INTERSECT=0.00 CPUEn=0 BANK_PART=1 PROFILE=1 -j 14
doRunLargeDTST GPUonly_rand_sep_BMAP_s

make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 CPUEn=0 BANK_PART=2 PROFILE=1 -j 14
doRunLargeDTST GPUonly_rand_interleaved_BMAP_s

make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 CPUEn=0 BANK_PART=3 PROFILE=1 -j 14
doRunLargeDTSTZipf GPUonly_zipf_interleaved_BMAP_s

make clean ; make CMP_TYPE=DISABLED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 GPU_PART=0.55 \
	CPU_PART=0.55 P_INTERSECT=0.00 BANK_PART=1 PROFILE=1 -j 14
doRunLargeDTST GPUonly_rand_sep_DISABLED_s

make clean ; make CMP_TYPE=DISABLED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 BANK_PART=2 PROFILE=1 -j 14
doRunLargeDTST GPUonly_rand_interleaved_DISABLED_s

make clean ; make CMP_TYPE=DISABLED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 BANK_PART=3 PROFILE=1 -j 14
doRunLargeDTSTZipf GPUonly_zipf_interleaved_DISABLED_s

############### CPU-only
make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 BANK_PART=1 GPU_PART=0.55 \
	CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_rand_sep_DISABLED_s

make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 BANK_PART=2 PROFILE=1 -j 14
doRunLargeDTST CPUonly_rand_interleaved_DISABLED_s

make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 BANK_PART=3 PROFILE=1 -j 14
doRunLargeDTSTZipf CPUonly_zipf_interleaved_DISABLED_s

make clean ; make LOG_TYPE=VERS INST_CPU=1 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_rand_sep_VERS_s

make clean ; make LOG_TYPE=VERS INST_CPU=1 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=2 PROFILE=1 -j 14
doRunLargeDTST CPUonly_rand_interleaved_VERS_s

make clean ; make LOG_TYPE=VERS INST_CPU=1 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=3 PROFILE=1 -j 14
doRunLargeDTSTZipf CPUonly_zipf_interleaved_VERS_s

make clean ; make LOG_TYPE=BMAP INST_CPU=1 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST CPUonly_rand_sep_BMAP_s

make clean ; make LOG_TYPE=BMAP INST_CPU=1 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=2 PROFILE=1 -j 14
doRunLargeDTST CPUonly_rand_interleaved_BMAP_s

make clean ; make LOG_TYPE=BMAP INST_CPU=1 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=3 PROFILE=1 -j 14
doRunLargeDTSTZipf CPUonly_zipf_interleaved_BMAP_s

############### VERS
make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST VERS_rand_sep_s

make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=2 PROFILE=1 -j 14
doRunLargeDTST VERS_rand_interleaved_s

make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=3 PROFILE=1 -j 14
doRunLargeDTSTZipf VERS_zipf_interleaved_s

############### BMAP
make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14
doRunLargeDTST BMAP_rand_sep_s

make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=2 PROFILE=1 -j 14
doRunLargeDTST BMAP_rand_interleaved_s

make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=3 PROFILE=1 -j 14
doRunLargeDTSTZipf BMAP_zipf_interleaved_s
