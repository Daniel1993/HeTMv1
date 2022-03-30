#!/bin/bash

SAMPLES=10
DURATION=12000
#./makeTM.sh
DURATION_ORG=12000
DURATION_GPU=6000
DURATION=$DURATION_ORG

DATA_FOLDER=$(pwd)/data
mkdir -p $DATA_FOLDER

cd ../../bank

rm -f Bank.csv

L_DATASET=150000000
S_DATASET=15000000
# CPU_BACKOFF=250

CPU_THREADS=8
# CPU_THREADS=2
# GPU_BLOCKS=80
GPU_BLOCKS=20

TRANSACTION_SIZE=4
CPU_BACKOFF=0
PROB_WRITE=100

function doRunLargeDTST {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		# 100M 500M 1G 1.5G
		timeout 30s ./bank -n $CPU_THREADS -b $GPU_BLOCKS -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.01
		timeout 30s ./bank -n $CPU_THREADS -b $GPU_BLOCKS -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.02
		timeout 30s ./bank -n $CPU_THREADS -b $GPU_BLOCKS -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.03
		timeout 30s ./bank -n $CPU_THREADS -b $GPU_BLOCKS -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.04
		timeout 30s ./bank -n $CPU_THREADS -b $GPU_BLOCKS -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.06
		timeout 30s ./bank -n $CPU_THREADS -b $GPU_BLOCKS -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.08
		timeout 30s ./bank -n $CPU_THREADS -b $GPU_BLOCKS -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.10
		### TODO: larger batches
		###
		mv Bank.csv $DATA_FOLDER/${1}_w${PROB_WRITE}_s${s}
	done
}

function doRunLargeDTST_CPU {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		# 100M 500M 1G 1.5G
		timeout 60s ./bank -n $CPU_THREADS -b 20 -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.60
		tail -n 1 Bank.csv > /tmp/BankLastLine.csv
		# for i in `seq 1 7`
		for i in `seq 1 6`
		do
			cat /tmp/BankLastLine.csv >> Bank.csv
		done
		mv Bank.csv $DATA_FOLDER/${1}_w${PROB_WRITE}_s${s}
	done
}

# DATASET=$L_DATASET
DATASET=$S_DATASET
DURATION=$DURATION_GPU


# GPU_BACKOFF=200000
GPU_BACKOFF=0

###########################################################################
############### GPU-only
make clean ; make CMP_TYPE=COMPRESSED DISABLE_RS=1 USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=50 \
	BANK_PART=6 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	BMAP_GRAN_BITS=13 BANK_PART_SCALE=1 >/dev/null
### 90% writes
doRunLargeDTST GPUonly_rand_sep_DISABLED_large

# make clean ; make CMP_TYPE=COMPRESSED DISABLE_RS=1 USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=50 \
# 	BANK_PART=6 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=13 OVERLAP_CPY_BACK=1 BANK_PART_SCALE=1 >/dev/null
# ### 90% writes
# doRunLargeDTST GPUonly_rand_sep_DISABLED_OVERLAP_large

############## CPU-only
make clean ; make INST_CPU=0 GPUEn=0 LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=50 \
	BANK_PART=6 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	BMAP_GRAN_BITS=13 BANK_PART_SCALE=1 >/dev/null
### 90% writes
doRunLargeDTST_CPU CPUonly_rand_sep_DISABLED_large

DURATION=$DURATION_ORG


############## VERS
make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=50 \
	BANK_PART=6 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	BMAP_GRAN_BITS=13 DISABLE_NON_BLOCKING=1 OVERLAP_CPY_BACK=0 \
	LOG_SIZE=4096 DISABLE_EARLY_VALIDATION=1 STM_LOG_BUFFER_SIZE=256 BANK_PART_SCALE=1 >/dev/null
### 90% writes
doRunLargeDTST VERS_BLOC_rand_sep_large
#
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=50 \
# 	BANK_PART=6 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=13 DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=0 \
# 	LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 BANK_PART_SCALE=1 >/dev/null
# ### 90% writes
# doRunLargeDTST VERS_NON_BLOC_rand_sep_large

# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=50 \
# 	BANK_PART=6 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=13 DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 \
# 	LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 BANK_PART_SCALE=1 >/dev/null
# ### 90% writes
# doRunLargeDTST VERS_NON_BLOC_OVER_rand_sep_large

# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=50 \
# 	BANK_PART=6 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=13 DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 \
# 	LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 BANK_PART_SCALE=1 DISABLE_EARLY_VALIDATION=1 >/dev/null
# ### 90% writes
# doRunLargeDTST VERS_NON_BLOC_OVER_NE_rand_sep_large

############## BMAP
make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=50 \
	BANK_PART=6 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	BMAP_GRAN_BITS=13 DISABLE_NON_BLOCKING=1 OVERLAP_CPY_BACK=0 \
	LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 BANK_PART_SCALE=1  DISABLE_EARLY_VALIDATION=1 >/dev/null
### 90% writes
doRunLargeDTST BMAP_rand_sep_large
