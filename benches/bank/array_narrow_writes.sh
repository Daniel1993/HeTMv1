#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1
filename_tsx="Bank_TSX"
filename_tiny="Bank_Tiny"

GPU_PART="1.0"
CPU_PART="0.0"
P_INTERSECT="0.0"
DURATION=20000
BLOCKS="2 4 8 16 32 64 256 512 1024" # 512
THREADS="512" #"2 4 8 16 32 64 96 256 320 512 640 768 1024"
BATCH_SIZE="4"
SAMPLES=5
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
		# 100M 500M 1G 1.5G
		timeout 40s ./bank -n $CPU_THREADS -b 1024 -x 128 -a 25000000 -d $DURATION -R 0 -S 4 -l 100 -N 1      -T 32 CPU_BACKOFF=400
		timeout 40s ./bank -n $CPU_THREADS -b 1024 -x 128 -a 25000000 -d $DURATION -R 0 -S 4 -l 100 -N 25     -T 32 CPU_BACKOFF=400
		timeout 40s ./bank -n $CPU_THREADS -b 1024 -x 128 -a 25000000 -d $DURATION -R 0 -S 4 -l 100 -N 250    -T 32 CPU_BACKOFF=400
		timeout 40s ./bank -n $CPU_THREADS -b 1024 -x 128 -a 25000000 -d $DURATION -R 0 -S 4 -l 100 -N 2500   -T 32 CPU_BACKOFF=400
		timeout 40s ./bank -n $CPU_THREADS -b 1024 -x 128 -a 25000000 -d $DURATION -R 0 -S 4 -l 100 -N 25000  -T 32 CPU_BACKOFF=400
		# timeout 40s ./bank -n $CPU_THREADS -b 1024 -x 128 -a 25000000 -d $DURATION -R 0 -S 8 -l 100 -N 250000 -T 32 CPU_BACKOFF=100
		# timeout 40s ./bank -n $CPU_THREADS -b 1024 -x 128 -a 250000000 -d $DURATION -R 0 -S 8 -l 90 -N 1      -T 32 CPU_BACKOFF=100
		# timeout 40s ./bank -n $CPU_THREADS -b 1024 -x 128 -a 250000000 -d $DURATION -R 0 -S 8 -l 90 -N 250    -T 32 CPU_BACKOFF=100
		# timeout 40s ./bank -n $CPU_THREADS -b 1024 -x 128 -a 250000000 -d $DURATION -R 0 -S 8 -l 90 -N 250000 -T 32 CPU_BACKOFF=100
		mv Bank.csv ${1}${s}
	done
}

### Fixed the amount of CPU threads
CPU_THREADS=14

###########################################################################

############### GPU-only
make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20  \
	BANK_PART=6 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	BMAP_GRAN_BITS=12 >/dev/null
doRunLargeDTST GPUonly_rand_sep_DISABLED_s

############## CPU-only
make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
		BANK_PART=6 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
		BMAP_GRAN_BITS=12 >/dev/null
doRunLargeDTST CPUonly_rand_sep_DISABLED_s

############## VERS
make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=6 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	BMAP_GRAN_BITS=12 >/dev/null
doRunLargeDTST VERS_rand_sep_s

############## BMAP
make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=6 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	BMAP_GRAN_BITS=12 >/dev/null
doRunLargeDTST BMAP_rand_sep_s
###########################################################################

#
#
# ################ Contiguous
#
#
# ############### LARGE
# ###########################################################################
# CPU_THREADS=14
# LARGE_DATASET=250000000
#
# ############### GPU-only
# make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 GPU_PART=0.55 \
# 	CPU_PART=0.55 P_INTERSECT=0.00 BANK_PART=5 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
# doRunLargeDTST GPUonly_cont_sep_BMAP_s
#
# make clean ; make CMP_TYPE=DISABLED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 GPU_PART=0.55 \
# 	CPU_PART=0.55 P_INTERSECT=0.00 BANK_PART=5 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
# doRunLargeDTST GPUonly_cont_sep_DISABLED_s
#
# ############## CPU-only
# make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 		BANK_PART=5 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
# doRunLargeDTST CPUonly_cont_sep_DISABLED_s
#
# # make clean ; make LOG_TYPE=VERS INST_CPU=1 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# # 	BANK_PART=5 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
# # doRunLargeDTST CPUonly_cont_sep_VERS_s
# #
# # make clean ; make LOG_TYPE=BMAP INST_CPU=1 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# # 	BANK_PART=5 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
# # doRunLargeDTST CPUonly_cont_sep_BMAP_s
# #
# # make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
# # 	BANK_PART=5 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
# # doRunLargeDTST CPUonly_cont_sep_DISABLED_STM_s
# #
# # make clean ; make LOG_TYPE=VERS INST_CPU=1 GPUEn=0 USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
# # 	BANK_PART=5 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
# # doRunLargeDTST CPUonly_cont_sep_VERS_STM_s
# #
# # make clean ; make LOG_TYPE=BMAP INST_CPU=1 GPUEn=0 USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
# # 	BANK_PART=5 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
# # doRunLargeDTST CPUonly_cont_sep_BMAP_STM_s
#
# ############## VERS
# ### STM
# # make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
# # 	BANK_PART=5 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
# # doRunLargeDTST VERS_cont_sep_STM_s
#
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=5 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
# doRunLargeDTST VERS_cont_sep_s
#
# ############## BMAP
# ### STM
# # make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
# # 	BANK_PART=5 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
# # doRunLargeDTST BMAP_cont_sep_STM_s
#
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=5 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
# doRunLargeDTST BMAP_cont_sep_s
# ###########################################################################
#
# ############### SMALL
# ###########################################################################
# LARGE_DATASET=25000000
#
# ############### GPU-only
# make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 GPU_PART=0.55 \
# 	CPU_PART=0.55 P_INTERSECT=0.00 CPUEn=0 BANK_PART=5 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
# doRunLargeDTST GPUonly_cont_sep_BMAP_SMALL_s
#
# make clean ; make CMP_TYPE=DISABLED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 GPU_PART=0.55 \
# 	CPU_PART=0.55 P_INTERSECT=0.00 BANK_PART=5 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
# doRunLargeDTST GPUonly_cont_sep_DISABLED_SMALL_s
#
# ############## CPU-only
# make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 		BANK_PART=5 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
# doRunLargeDTST CPUonly_cont_sep_DISABLED_SMALL_s
#
# # make clean ; make LOG_TYPE=VERS INST_CPU=1 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# # 	BANK_PART=5 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
# # doRunLargeDTST CPUonly_cont_sep_VERS_SMALL_s
# #
# # make clean ; make LOG_TYPE=BMAP INST_CPU=1 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# # 	BANK_PART=5 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
# # doRunLargeDTST CPUonly_cont_sep_BMAP_SMALL_s
# #
# # make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
# # 	BANK_PART=5 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
# # doRunLargeDTST CPUonly_cont_sep_DISABLED_STM_SMALL_s
# #
# # make clean ; make LOG_TYPE=VERS INST_CPU=1 GPUEn=0 USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
# # 	BANK_PART=5 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
# # doRunLargeDTST CPUonly_cont_sep_VERS_STM_SMALL_s
# #
# # make clean ; make LOG_TYPE=BMAP INST_CPU=1 GPUEn=0 USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
# # 	BANK_PART=5 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
# # doRunLargeDTST CPUonly_cont_sep_BMAP_STM_SMALL_s
#
# ############## VERS
# ### STM
# # make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
# # 	BANK_PART=5 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
# # doRunLargeDTST VERS_cont_sep_STM_SMALL_s
#
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=5 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
# doRunLargeDTST VERS_cont_sep_SMALL_s
#
# ############## BMAP
# ### STM
# # make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
# # 	BANK_PART=5 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
# # doRunLargeDTST BMAP_cont_sep_STM_SMALL_s
#
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=5 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.0
# doRunLargeDTST BMAP_cont_sep_SMALL_s
# ###########################################################################

mkdir -p array_narrow
mv *_s* array_narrow/
