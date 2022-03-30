#!/bin/bash

DURATION=8000
SAMPLES=4

DATA_FOLDER=$(pwd)/data
mkdir -p $DATA_FOLDER

cd ../../benches/bank

rm -f Bank.csv

# sleep 3h

LARGE_DATASET=1000000 #2621440 # 90 000 000 is the max for my home machine
# LARGE_DATASET=20000000 #2621440 # 90 000 000 is the max for my home machine
LARGE_DATASET_P20=61000000 #2621440 # 90 000 000 is the max for my home machine
SMALL_DATASET=1000000 #2621440 # 90 000 000 is the max for my home machine
SMALL_DATASET_P20=122000 #2621440 # 90 000 000 is the max for my home machine

function doRunLargeDTST {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		# -n $CPU_THREADS -b 1024 -x 512 ---> changed to single threaded
		timeout 40s ./bank -n 8 -b 15 -x 128 -a $LARGE_DATASET -d $DURATION -R 0 -S 4 -l  10 -N 1 -T 1 CPU_BACKOFF=0 GPU_BACKOFF=0 -X 0.01
		timeout 40s ./bank -n 8 -b 15 -x 128 -a $LARGE_DATASET -d $DURATION -R 0 -S 4 -l  25 -N 1 -T 1 CPU_BACKOFF=0 GPU_BACKOFF=0 -X 0.01
		timeout 40s ./bank -n 8 -b 15 -x 128 -a $LARGE_DATASET -d $DURATION -R 0 -S 4 -l  50 -N 1 -T 1 CPU_BACKOFF=0 GPU_BACKOFF=0 -X 0.01
		timeout 40s ./bank -n 8 -b 15 -x 128 -a $LARGE_DATASET -d $DURATION -R 0 -S 4 -l  75 -N 1 -T 1 CPU_BACKOFF=0 GPU_BACKOFF=0 -X 0.01
		timeout 40s ./bank -n 8 -b 15 -x 128 -a $LARGE_DATASET -d $DURATION -R 0 -S 4 -l  90 -N 1 -T 1 CPU_BACKOFF=0 GPU_BACKOFF=0 -X 0.01

		mv Bank.csv $DATA_FOLDER/${1}${s}
	done
}

# LARGE_DATASET=1000000

############### LARGE
###########################################################################
# LARGE_DATASET=150000000

############### GPU-only
# make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=200 GPU_PART=0.9 \
# 	CPU_PART=0.1 P_INTERSECT=0.00 CPUEn=0 BANK_PART=1 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=18 >/dev/null
# doRunLargeDTST GPUonly_rand_sep_BMAP_s

# make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=200 GPU_PART=1.0 \
# 	CPU_PART=0.0 P_INTERSECT=0.00 CPUEn=0 BANK_PART=1 PROFILE=1 -j 14 DISABLE_WS=1 \
# 	BMAP_GRAN_BITS=18 >/dev/null
# doRunLargeDTST GPUonly_rand_sep_BMAP_NO_WS_s

# make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=200 GPU_PART=1.0 \
# 	CPU_PART=0.0 P_INTERSECT=0.00 CPUEn=0 BANK_PART=1 PROFILE=1 -j 14 DISABLE_RS=1 \
# 	BMAP_GRAN_BITS=18 >/dev/null
# doRunLargeDTST GPUonly_rand_sep_BMAP_NO_RS_s

# TODO add INST_GPU
./compile.sh opt                     \
	CMP_TYPE=COMPRESSED                \
	HETM_CPU_EN=0                      \
	HETM_GPU_EN=1                      \
	LOG_TYPE=BMAP                      \
	USE_TSX_IMPL=0                     \
	PR_MAX_RWSET_SIZE=20               \
	BANK_PART=1                        \
	GPU_PART=0.90                      \
	CPU_PART=0.10                      \
	P_INTERSECT=0.00                   \
	PROFILE=0                          \
	BMAP_GRAN_BITS=15                  \
	HETM_NB_DEVICES=1
#
doRunLargeDTST GPUonly_1_s

# # TODO add INST_GPU
# ./compile.sh opt                     \
# 	CMP_TYPE=COMPRESSED                \
# 	HETM_CPU_EN=0                      \
# 	HETM_GPU_EN=1                      \
# 	LOG_TYPE=BMAP                      \
# 	USE_TSX_IMPL=0                     \
# 	PR_MAX_RWSET_SIZE=200              \
# 	BANK_PART=1                        \
# 	GPU_PART=0.00                      \
# 	CPU_PART=1.00                      \
# 	P_INTERSECT=0.00                   \
# 	PROFILE=0                          \
# 	BMAP_GRAN_BITS=18 \
# 	HETM_NB_DEVICES=2
# #
# doRunLargeDTST GPUonly_2_s

############### GPU-only (small RS)
# make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 GPU_PART=1.0 \
# 	CPU_PART=0.0 P_INTERSECT=0.00 CPUEn=0 BANK_PART=1 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=18 REDUCED_RS=10 >/dev/null
# doRunLargeDTST GPUonly_rand_sep_BMAP_10b_s

# make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 GPU_PART=1.0 \
# 	CPU_PART=0.0 P_INTERSECT=0.00 CPUEn=0 BANK_PART=1 PROFILE=1 -j 14 DISABLE_WS=1 \
# 	BMAP_GRAN_BITS=18 REDUCED_RS=10 >/dev/null
# doRunLargeDTST GPUonly_rand_sep_BMAP_NO_WS_10b_s

# make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 GPU_PART=1.0 \
# 	CPU_PART=0.0 P_INTERSECT=0.00 CPUEn=0 BANK_PART=1 PROFILE=1 -j 14 DISABLE_RS=1 \
# 	BMAP_GRAN_BITS=18 REDUCED_RS=10 >/dev/null
# doRunLargeDTST GPUonly_rand_sep_BMAP_NO_RS_10b_s

# make clean ; make CMP_TYPE=DISABLED USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 GPU_PART=1.0 \
# 	CPU_PART=0.0 P_INTERSECT=0.00 CPUEn=0 BANK_PART=1 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=18 REDUCED_RS=10 >/dev/null
# doRunLargeDTST GPUonly_rand_sep_DISABLED_10b_s

# ############## CPU-only
# make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 		BANK_PART=1 GPU_PART=0.0 CPU_PART=1.0 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 		BMAP_GRAN_BITS=18 >/dev/null
# doRunLargeDTST CPUonly_rand_sep_DISABLED_s

# TODO add INST_CPU back
./compile.sh opt                       \
	CMP_TYPE=COMPRESSED                \
	HETM_CPU_EN=1                      \
	HETM_GPU_EN=0                      \
	LOG_TYPE=BMAP                      \
	USE_TSX_IMPL=0                     \
	PR_MAX_RWSET_SIZE=20              \
	BANK_PART=1                        \
	GPU_PART=0.10                      \
	CPU_PART=0.90                      \
	P_INTERSECT=0.00                   \
	PROFILE=0                          \
	BMAP_GRAN_BITS=15 \
	HETM_NB_DEVICES=1
#
doRunLargeDTST CPUonly_STM_s

# ############## BMAP
# make clean ; make CMP_TYPE=COMPRESSED GPUEn=0 LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=200 \
# 	BANK_PART=1 GPU_PART=0.0 CPU_PART=1.0 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=18 >/dev/null
# doRunLargeDTST BMAP_rand_sep_s

./compile.sh opt                       \
	CMP_TYPE=COMPRESSED                \
	HETM_CPU_EN=1                      \
	HETM_GPU_EN=1                      \
	LOG_TYPE=BMAP                      \
	USE_TSX_IMPL=0                     \
	PR_MAX_RWSET_SIZE=20              \
	BANK_PART=1                        \
	GPU_PART=0.5                       \
	CPU_PART=0.5                       \
	P_INTERSECT=0.00                   \
	PROFILE=0                          \
	BMAP_GRAN_BITS=15 \
	HETM_NB_DEVICES=1
#
doRunLargeDTST BMAP_1GPU_STM_s

./compile.sh opt                       \
	CMP_TYPE=COMPRESSED                \
	HETM_CPU_EN=1                      \
	HETM_GPU_EN=1                      \
	LOG_TYPE=BMAP                      \
	USE_TSX_IMPL=0                     \
	PR_MAX_RWSET_SIZE=20              \
	BANK_PART=1                        \
	GPU_PART=0.5                       \
	CPU_PART=0.5                       \
	P_INTERSECT=0.00                   \
	PROFILE=0                          \
	BMAP_GRAN_BITS=15 \
	HETM_NB_DEVICES=2
#
doRunLargeDTST BMAP_2GPU_STM_s
###########################################################################

# mkdir -p v2_multiGPU_prec_write_multiGPU
# mv *_s* v2_multiGPU_prec_write_multiGPU/
