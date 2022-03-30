#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1
filename_tsx="Bank_TSX"
filename_tiny="Bank_Tiny"

GPU_PART="1.0"
CPU_PART="0.0"
P_INTERSECT="0.0"
BLOCKS="2 4 8 16 32 64 256 512 1024" # 512
THREADS="512" #"2 4 8 16 32 64 96 256 320 512 640 768 1024"
BATCH_SIZE="4"
SAMPLES=4
#./makeTM.sh

CPU_THREADS=4
LOW_CPU_THREADS=10
HIGH_CPU_THREADS=20

rm -f Bank.csv

DURATION=10000
DATASET=250000000
PROB_WRITE=50 # test also with 50 and 10
CPU_BACKOFF=50

function doRunLargeDTST_1 {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		# 100M 500M 1G 1.5G
		timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l $PROB_WRITE -N 1 -T 32 CPU_BACKOFF=$CPU_BACKOFF BATCH_DURATION=1
		timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l $PROB_WRITE -N 1 -T 32 CPU_BACKOFF=$CPU_BACKOFF BATCH_DURATION=5
		timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l $PROB_WRITE -N 1 -T 32 CPU_BACKOFF=$CPU_BACKOFF BATCH_DURATION=10
		timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l $PROB_WRITE -N 1 -T 32 CPU_BACKOFF=$CPU_BACKOFF BATCH_DURATION=25
		timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l $PROB_WRITE -N 1 -T 32 CPU_BACKOFF=$CPU_BACKOFF BATCH_DURATION=50
		timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l $PROB_WRITE -N 1 -T 32 CPU_BACKOFF=$CPU_BACKOFF BATCH_DURATION=100
		timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l $PROB_WRITE -N 1 -T 32 CPU_BACKOFF=$CPU_BACKOFF BATCH_DURATION=150
		timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l $PROB_WRITE -N 1 -T 32 CPU_BACKOFF=$CPU_BACKOFF BATCH_DURATION=200
		# timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l 100 -N 1 -T 32 CPU_BACKOFF=50 BATCH_DURATION=250
		# timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l 100 -N 1 -T 32 CPU_BACKOFF=50 BATCH_DURATION=300
		# timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l 100 -N 1 -T 32 CPU_BACKOFF=50 BATCH_DURATION=400
		# timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l 100 -N 1 -T 32 CPU_BACKOFF=50 BATCH_DURATION=500
		mv Bank.csv ${1}${s}
	done
}

function doRunLargeDTST_100 {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		# 100M 500M 1G 1.5G
		timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l $PROB_WRITE -N 100 -T 32 CPU_BACKOFF=$CPU_BACKOFF BATCH_DURATION=1
		timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l $PROB_WRITE -N 100 -T 32 CPU_BACKOFF=$CPU_BACKOFF BATCH_DURATION=5
		timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l $PROB_WRITE -N 100 -T 32 CPU_BACKOFF=$CPU_BACKOFF BATCH_DURATION=10
		timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l $PROB_WRITE -N 100 -T 32 CPU_BACKOFF=$CPU_BACKOFF BATCH_DURATION=25
		timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l $PROB_WRITE -N 100 -T 32 CPU_BACKOFF=$CPU_BACKOFF BATCH_DURATION=50
		timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l $PROB_WRITE -N 100 -T 32 CPU_BACKOFF=$CPU_BACKOFF BATCH_DURATION=100
		timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l $PROB_WRITE -N 100 -T 32 CPU_BACKOFF=$CPU_BACKOFF BATCH_DURATION=150
		timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l $PROB_WRITE -N 100 -T 32 CPU_BACKOFF=$CPU_BACKOFF BATCH_DURATION=200
		# timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l 100 -N 100 -T 32 CPU_BACKOFF=50 BATCH_DURATION=250
		# timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l 100 -N 100 -T 32 CPU_BACKOFF=50 BATCH_DURATION=300
		# timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l 100 -N 100 -T 32 CPU_BACKOFF=50 BATCH_DURATION=400
		# timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l 100 -N 100 -T 32 CPU_BACKOFF=50 BATCH_DURATION=500
		mv Bank.csv ${1}${s}
	done
}

function doRunLargeDTST_GPU_1 {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		# 100M 500M 1G 1.5G
		timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l $PROB_WRITE \
			-N 1 -T 32 CPU_BACKOFF=$CPU_BACKOFF BATCH_DURATION=150
		tail -n 1 Bank.csv > /tmp/BankLastLine.csv
		for i in `seq 1 7`
		do
			cat /tmp/BankLastLine.csv >> Bank.csv
		done
		mv Bank.csv ${1}${s}
	done
}

function doRunLargeDTST_GPU_100 {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		# 100M 500M 1G 1.5G
		timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 2 -l $PROB_WRITE \
			-N 100 -T 32 CPU_BACKOFF=$CPU_BACKOFF BATCH_DURATION=150
		tail -n 1 Bank.csv > /tmp/BankLastLine.csv
		for i in `seq 1 7`
		do
			cat /tmp/BankLastLine.csv >> Bank.csv
		done
		mv Bank.csv ${1}${s}
	done
}

### Fixed the amount of CPU threads
CPU_THREADS=14

###########################################################################

############### GPU-only
make clean ; make CMP_TYPE=DISABLED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=8 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	BMAP_GRAN_BITS=12 >/dev/null
doRunLargeDTST_GPU_1 GPUonly_rand_sep_DISABLED_1_s
doRunLargeDTST_GPU_100 GPUonly_rand_sep_DISABLED_100_s

############## CPU-only
make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=8 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	BMAP_GRAN_BITS=12 >/dev/null
doRunLargeDTST_GPU_1 CPUonly_rand_sep_DISABLED_1_s
doRunLargeDTST_GPU_100 CPUonly_rand_sep_DISABLED_100_s

############## VERS
make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=8 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	BMAP_GRAN_BITS=12 DISABLE_NON_BLOCKING=1 >/dev/null
doRunLargeDTST_1 VERS_BLOC_rand_sep_1_s
doRunLargeDTST_100 VERS_BLOC_rand_sep_100_s

make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=8 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	BMAP_GRAN_BITS=12 DISABLE_NON_BLOCKING=0 >/dev/null
doRunLargeDTST_1 VERS_NON_BLOC_rand_sep_1_s
doRunLargeDTST_100 VERS_NON_BLOC_rand_sep_100_s

############## BMAP
make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=8 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	BMAP_GRAN_BITS=12 >/dev/null
doRunLargeDTST_1 BMAP_rand_sep_1_s
doRunLargeDTST_100 BMAP_rand_sep_100_s
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

mkdir -p array_batch_duration_2
mv *_s* array_batch_duration_2/
