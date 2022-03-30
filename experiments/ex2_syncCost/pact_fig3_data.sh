#!/bin/bash

DATA_FOLDER=$(pwd)/data
mkdir -p $DATA_FOLDER

cd ../../benches/bank

SAMPLES=3
DURATION_ORG=12000
DURATION_GPU=8000
#./makeTM.sh
DURATION=$DURATION_ORG

rm -f Bank.csv

# L_DATASET=150000000
L_DATASET=1000000
S_DATASET=15000000
# CPU_BACKOFF=250

CPU_THREADS=8
TRANSACTION_SIZE=4
CPU_BACKOFF=0
# GPU_BACKOFF=800000
GPU_BACKOFF=0
PROB_WRITE=100

function doRunLargeDTST {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		# 100M 500M 1G 1.5G
		timeout 60s ./bank -n $CPU_THREADS -b 80 -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.02
		timeout 60s ./bank -n $CPU_THREADS -b 80 -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.04
		timeout 60s ./bank -n $CPU_THREADS -b 80 -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.08
		timeout 60s ./bank -n $CPU_THREADS -b 80 -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.12
		timeout 60s ./bank -n $CPU_THREADS -b 80 -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.20
		timeout 60s ./bank -n $CPU_THREADS -b 80 -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.30
		timeout 60s ./bank -n $CPU_THREADS -b 80 -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.40
		timeout 60s ./bank -n $CPU_THREADS -b 80 -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.50
		timeout 60s ./bank -n $CPU_THREADS -b 80 -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.60
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
		timeout 60s ./bank -n $CPU_THREADS -b 80 -x 256 -a $DATASET -d $DURATION -R 0 \
			-S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1 CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.60
		tail -n 1 Bank.csv > /tmp/BankLastLine.csv
		# for i in `seq 1 7`
		for i in `seq 1 8`
		do
			cat /tmp/BankLastLine.csv >> Bank.csv
		done
		mv Bank.csv $DATA_FOLDER/${1}_w${PROB_WRITE}_s${s}
	done
}

DURATION=$DURATION_GPU
DATASET=$L_DATASET
# GPU_BACKOFF=800000

###########################################################################
############### GPU-only
# make clean ; make CMP_TYPE=COMPRESSED DISABLE_RS=1 USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=13 >/dev/null
./compile.sh opt                       \
	CMP_TYPE=COMPRESSED                \
	HETM_CPU_EN=0                      \
	HETM_GPU_EN=1                      \
	LOG_TYPE=BMAP                      \
	USE_TSX_IMPL=0                     \
	PR_MAX_RWSET_SIZE=200              \
	BANK_PART=9                        \
	GPU_PART=0.55                      \
	CPU_PART=0.55                      \
	P_INTERSECT=0.00                   \
	PROFILE=1                          \
	BMAP_GRAN_BITS=13 \
	HETM_NB_DEVICES=1
#
doRunLargeDTST GPUonly_rand_sep_DISABLED_large

# make clean ; make CMP_TYPE=COMPRESSED DISABLE_RS=1 USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=13 OVERLAP_CPY_BACK=1 >/dev/null
# doRunLargeDTST GPUonly_rand_sep_DISABLED_OVERLAP_large

############## CPU-only
# make clean ; make INST_CPU=0 GPUEn=0 LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=13 >/dev/null
./compile.sh opt                       \
	CMP_TYPE=COMPRESSED                \
	HETM_CPU_EN=1                      \
	HETM_GPU_EN=0                      \
	LOG_TYPE=BMAP                      \
	USE_TSX_IMPL=1                     \
	PR_MAX_RWSET_SIZE=200              \
	BANK_PART=9                        \
	GPU_PART=0.55                      \
	CPU_PART=0.55                      \
	P_INTERSECT=0.00                   \
	PROFILE=1                          \
	BMAP_GRAN_BITS=13 \
	HETM_NB_DEVICES=1
#
doRunLargeDTST_CPU CPUonly_rand_sep_DISABLED_large

DURATION=$DURATION_ORG

############## VERS
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=13 DISABLE_NON_BLOCKING=1 OVERLAP_CPY_BACK=0 \
# 	LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 >/dev/null
# doRunLargeDTST VERS_BLOC_rand_sep_large

# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=13 DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=0 \
# 	LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 >/dev/null
# ### 90% writes
# DATASET=$L_DATASET
# GPU_BACKOFF=800000
# doRunLargeDTST VERS_NON_BLOC_rand_sep_large
# # DATASET=$S_DATASET
# # GPU_BACKOFF=600000
# # doRunLargeDTST VERS_NON_BLOC_rand_sep_small

# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=13 DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 \
# 	LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 >/dev/null
# doRunLargeDTST VERS_NON_BLOC_OVER_rand_sep_large

# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=13 DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 \
# 	LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 DISABLE_EARLY_VALIDATION=1 >/dev/null
# doRunLargeDTST VERS_NON_BLOC_OVER_NE_rand_sep_large

############## BMAP
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=13 DISABLE_NON_BLOCKING=1 OVERLAP_CPY_BACK=0 \
# 	LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 >/dev/null
# doRunLargeDTST BMAPC_rand_sep_large

# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=13 DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=0 \
# 	LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 >/dev/null
# ### 90% writes
# DATASET=$L_DATASET
# GPU_BACKOFF=800000
# doRunLargeDTST VERS_NON_BLOC_rand_sep_large
# # DATASET=$S_DATASET
# # GPU_BACKOFF=600000
# # doRunLargeDTST VERS_NON_BLOC_rand_sep_small

# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=13 DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 \
# 	LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 >/dev/null
./compile.sh opt                       \
	CMP_TYPE=COMPRESSED                \
	HETM_CPU_EN=1                      \
	HETM_GPU_EN=1                      \
	LOG_TYPE=BMAP                      \
	USE_TSX_IMPL=1                     \
	PR_MAX_RWSET_SIZE=200              \
	BANK_PART=9                        \
	GPU_PART=0.55                      \
	CPU_PART=0.55                      \
	P_INTERSECT=0.00                   \
	PROFILE=1                          \
	BMAP_GRAN_BITS=13 \
	HETM_NB_DEVICES=1
#
doRunLargeDTST BMAP_rand_sep_1GPU

./compile.sh opt                       \
	CMP_TYPE=COMPRESSED                \
	HETM_CPU_EN=1                      \
	HETM_GPU_EN=1                      \
	LOG_TYPE=BMAP                      \
	USE_TSX_IMPL=1                     \
	PR_MAX_RWSET_SIZE=200              \
	BANK_PART=9                        \
	GPU_PART=0.55                      \
	CPU_PART=0.55                      \
	P_INTERSECT=0.00                   \
	PROFILE=1                          \
	BMAP_GRAN_BITS=13 \
	HETM_NB_DEVICES=2
#
doRunLargeDTST BMAP_rand_sep_2GPU

# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=13 DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 \
# 	LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 DISABLE_EARLY_VALIDATION=1 >/dev/null
# doRunLargeDTST BMAP_rand_sep_large

# mkdir -p $DATA_FOLDER/array_batch_duration_TSX
# mv $DATA_FOLDER/*_s* $DATA_FOLDER/array_batch_duration_TSX/
#
# ###############################################################################
# ###############################################################################
#
# #### TinySTM (Just the large data set for discussion)
#
# CPU_THREADS=8
#
# cp array_batch_duration_TSX/GPUonly_* array_batch_duration_STM/
#
# ############## CPU-only
# make clean ; make INST_CPU=0 GPUEn=0 LOG_TYPE=VERS USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=13 >/dev/null
# DATASET=$L_DATASET
# doRunLargeDTST_CPU CPUonly_rand_sep_DISABLED_large
# # DATASET=$S_DATASET
# # doRunLargeDTST_CPU CPUonly_rand_sep_DISABLED_small
#
# ############## VERS
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=13 DISABLE_NON_BLOCKING=1 OVERLAP_CPY_BACK=0 \
# 	LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 >/dev/null
# DATASET=$L_DATASET
# doRunLargeDTST_CPU VERS_BLOC_rand_sep_large
# # DATASET=$S_DATASET
# # doRunLargeDTST_CPU VERS_BLOC_rand_sep_small
#
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=13 DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=0 \
# 	LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 >/dev/null
# DATASET=$L_DATASET
# doRunLargeDTST_CPU VERS_NON_BLOC_rand_sep_large
# # DATASET=$S_DATASET
# # doRunLargeDTST_CPU VERS_NON_BLOC_rand_sep_small
#
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=9 BANK_INTRA_CONFL=0.0 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=13 DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 \
# 	LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 >/dev/null
# DATASET=$L_DATASET
# doRunLargeDTST_CPU VERS_NON_BLOC_OVER_rand_sep_large
# # DATASET=$S_DATASET
# # doRunLargeDTST_CPU VERS_NON_BLOC_OVER_rand_sep_small
#
# mkdir -p array_batch_duration_STM
# mv *_s* array_batch_duration_STM/
# ###############################################################################
