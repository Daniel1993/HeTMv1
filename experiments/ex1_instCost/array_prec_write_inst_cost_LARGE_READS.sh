#!/bin/bash

DURATION=5000
SAMPLES=1

rm -f Bank.csv

# sleep 3h

LARGE_DATASET=100000000 #2621440 # 90 000 000 is the max for my home machine
# LARGE_DATASET=20000000 #2621440 # 90 000 000 is the max for my home machine
LARGE_DATASET_P20=61000000 #2621440 # 90 000 000 is the max for my home machine
SMALL_DATASET=1000000 #2621440 # 90 000 000 is the max for my home machine
SMALL_DATASET_P20=122000 #2621440 # 90 000 000 is the max for my home machine

function doRunLargeDTST {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		# -n $CPU_THREADS -b 1024 -x 512 ---> changed to single threaded
		timeout 40s ./bank -n 8 -b 80 -x 256 -a $LARGE_DATASET -d $DURATION -R 0 -S 4 -l  10 -N 1 -T 1 CPU_BACKOFF=0 GPU_BACKOFF=0
		timeout 40s ./bank -n 8 -b 80 -x 256 -a $LARGE_DATASET -d $DURATION -R 0 -S 4 -l  25 -N 1 -T 1 CPU_BACKOFF=0 GPU_BACKOFF=0
		timeout 40s ./bank -n 8 -b 80 -x 256 -a $LARGE_DATASET -d $DURATION -R 0 -S 4 -l  50 -N 1 -T 1 CPU_BACKOFF=0 GPU_BACKOFF=0
		timeout 40s ./bank -n 8 -b 80 -x 256 -a $LARGE_DATASET -d $DURATION -R 0 -S 4 -l  75 -N 1 -T 1 CPU_BACKOFF=0 GPU_BACKOFF=0
		timeout 40s ./bank -n 8 -b 80 -x 256 -a $LARGE_DATASET -d $DURATION -R 0 -S 4 -l  90 -N 1 -T 1 CPU_BACKOFF=0 GPU_BACKOFF=0
		mv Bank.csv ${1}${s}
	done
}

############### LARGE
###########################################################################
LARGE_DATASET=150000000

############### GPU-only
make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=50 GPU_PART=1.0 \
	CPU_PART=0.0 P_INTERSECT=0.00 CPUEn=0 BANK_PART=10 PROFILE=1 -j 14 \
	BMAP_GRAN_BITS=13 >/dev/null
doRunLargeDTST GPUonly_rand_sep_BMAP_s

make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=50 GPU_PART=1.0 \
	CPU_PART=0.0 P_INTERSECT=0.00 CPUEn=0 BANK_PART=10 PROFILE=1 -j 14 DISABLE_WS=1 \
	BMAP_GRAN_BITS=13 >/dev/null
doRunLargeDTST GPUonly_rand_sep_BMAP_NO_WS_s

make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=50 GPU_PART=1.0 \
	CPU_PART=0.0 P_INTERSECT=0.00 CPUEn=0 BANK_PART=10 PROFILE=1 -j 14 DISABLE_RS=1 \
	BMAP_GRAN_BITS=13 >/dev/null
doRunLargeDTST GPUonly_rand_sep_BMAP_NO_RS_s

make clean ; make CMP_TYPE=DISABLED USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=50 GPU_PART=1.0 \
	CPU_PART=0.0 P_INTERSECT=0.00 CPUEn=0 BANK_PART=10 PROFILE=1 -j 14 \
	BMAP_GRAN_BITS=13 >/dev/null
doRunLargeDTST GPUonly_rand_sep_DISABLED_s

############### GPU-only (small RS)
make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=50 GPU_PART=1.0 \
	CPU_PART=0.0 P_INTERSECT=0.00 CPUEn=0 BANK_PART=10 PROFILE=1 -j 14 \
	BMAP_GRAN_BITS=13  REDUCED_RS=10 >/dev/null
doRunLargeDTST GPUonly_rand_sep_BMAP_10b_s

make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=50 GPU_PART=1.0 \
	CPU_PART=0.0 P_INTERSECT=0.00 CPUEn=0 BANK_PART=10 PROFILE=1 -j 14 DISABLE_WS=1 \
	BMAP_GRAN_BITS=13  REDUCED_RS=10 >/dev/null
doRunLargeDTST GPUonly_rand_sep_BMAP_NO_WS_10b_s

make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=50 GPU_PART=1.0 \
	CPU_PART=0.0 P_INTERSECT=0.00 CPUEn=0 BANK_PART=10 PROFILE=1 -j 14 DISABLE_RS=1 \
	BMAP_GRAN_BITS=13  REDUCED_RS=10 >/dev/null
doRunLargeDTST GPUonly_rand_sep_BMAP_NO_RS_10b_s

make clean ; make CMP_TYPE=DISABLED USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=50 GPU_PART=1.0 \
	CPU_PART=0.0 P_INTERSECT=0.00 CPUEn=0 BANK_PART=10 PROFILE=1 -j 14 \
	BMAP_GRAN_BITS=13  REDUCED_RS=10 >/dev/null
doRunLargeDTST GPUonly_rand_sep_DISABLED_10b_s

############## CPU-only
make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=50 \
		BANK_PART=10 GPU_PART=0.0 CPU_PART=1.0 P_INTERSECT=0.00 PROFILE=1 -j 14 \
		BMAP_GRAN_BITS=13 >/dev/null
doRunLargeDTST CPUonly_rand_sep_DISABLED_s

make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=50 \
		BANK_PART=10 GPU_PART=0.0 CPU_PART=1.0 P_INTERSECT=0.00 PROFILE=1 -j 14 \
		BMAP_GRAN_BITS=13 >/dev/null
doRunLargeDTST CPUonly_rand_sep_DISABLED_STM_s

############## VERS
make clean ; make CMP_TYPE=COMPRESSED GPUEn=0 LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=50 \
	BANK_PART=10 GPU_PART=0.0 CPU_PART=1.0 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	BMAP_GRAN_BITS=13 >/dev/null
doRunLargeDTST VERS_rand_sep_s

make clean ; make CMP_TYPE=COMPRESSED GPUEn=0 LOG_TYPE=VERS USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=50 \
	BANK_PART=10 GPU_PART=0.0 CPU_PART=1.0 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	BMAP_GRAN_BITS=13 >/dev/null
doRunLargeDTST VERS_rand_sep_STM_s
###########################################################################

mkdir -p array_prec_inst_cost_LARGE_READS
mv *_s* array_prec_inst_cost_LARGE_READS/
