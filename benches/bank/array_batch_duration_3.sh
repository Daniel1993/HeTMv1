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
SAMPLES=3
#./makeTM.sh

CPU_THREADS=4
LOW_CPU_THREADS=10
HIGH_CPU_THREADS=20

rm -f Bank.csv

DURATION=10000
DATASET=150000000 #250000000
PROB_WRITE=100 # test also with 50 and 10
# CPU_BACKOFF=250
CPU_BACKOFF=1

# TRANSACTION_SIZE=16 # default
TRANSACTION_SIZE=2

function doRunLargeDTST_1 {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		# 100M 500M 1G 1.5G

		timeout 60s ./bank -n 14 -b 640 -x 256 -a $DATASET -d $DURATION -R 0 -S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 1  CPU_BACKOFF=$CPU_BACKOFF
		# timeout 60s ./bank -n 14 -b 640 -x 256 -a $DATASET -d $DURATION -R 0 -S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 2  CPU_BACKOFF=$CPU_BACKOFF
		timeout 60s ./bank -n 14 -b 640 -x 256 -a $DATASET -d $DURATION -R 0 -S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 4  CPU_BACKOFF=$CPU_BACKOFF
		# timeout 60s ./bank -n 14 -b 640 -x 256 -a $DATASET -d $DURATION -R 0 -S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 8  CPU_BACKOFF=$CPU_BACKOFF
		timeout 60s ./bank -n 14 -b 640 -x 384 -a $DATASET -d $DURATION -R 0 -S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 8  CPU_BACKOFF=$CPU_BACKOFF
		timeout 60s ./bank -n 14 -b 640 -x 256 -a $DATASET -d $DURATION -R 0 -S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 16 CPU_BACKOFF=$CPU_BACKOFF
		timeout 60s ./bank -n 14 -b 640 -x 384 -a $DATASET -d $DURATION -R 0 -S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 16 CPU_BACKOFF=$CPU_BACKOFF
		timeout 60s ./bank -n 14 -b 640 -x 256 -a $DATASET -d $DURATION -R 0 -S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 32 CPU_BACKOFF=$CPU_BACKOFF
		timeout 60s ./bank -n 14 -b 768 -x 256 -a $DATASET -d $DURATION -R 0 -S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 32 CPU_BACKOFF=$CPU_BACKOFF
		timeout 60s ./bank -n 14 -b 640 -x 384 -a $DATASET -d $DURATION -R 0 -S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 32 CPU_BACKOFF=$CPU_BACKOFF
		timeout 60s ./bank -n 14 -b 768 -x 384 -a $DATASET -d $DURATION -R 0 -S $TRANSACTION_SIZE -l $PROB_WRITE -N 1 -T 32 CPU_BACKOFF=$CPU_BACKOFF
		### TODO: larger batches
		###
		mv Bank.csv ${1}${s}
	done
}

function doRunLargeDTST_100 {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		# 100M 500M 1G 1.5G
		timeout 60s ./bank -n 14 -b 640 -x 256 -a $DATASET -d $DURATION -R 0 -S $TRANSACTION_SIZE -l $PROB_WRITE -N 10 -T 1  CPU_BACKOFF=$CPU_BACKOFF
		# timeout 60s ./bank -n 14 -b 640 -x 256 -a $DATASET -d $DURATION -R 0 -S $TRANSACTION_SIZE -l $PROB_WRITE -N 10 -T 2  CPU_BACKOFF=$CPU_BACKOFF
		timeout 60s ./bank -n 14 -b 640 -x 256 -a $DATASET -d $DURATION -R 0 -S $TRANSACTION_SIZE -l $PROB_WRITE -N 10 -T 4  CPU_BACKOFF=$CPU_BACKOFF
		# timeout 60s ./bank -n 14 -b 640 -x 256 -a $DATASET -d $DURATION -R 0 -S $TRANSACTION_SIZE -l $PROB_WRITE -N 10 -T 8  CPU_BACKOFF=$CPU_BACKOFF
		timeout 60s ./bank -n 14 -b 640 -x 384 -a $DATASET -d $DURATION -R 0 -S $TRANSACTION_SIZE -l $PROB_WRITE -N 10 -T 8  CPU_BACKOFF=$CPU_BACKOFF
		timeout 60s ./bank -n 14 -b 640 -x 256 -a $DATASET -d $DURATION -R 0 -S $TRANSACTION_SIZE -l $PROB_WRITE -N 10 -T 16 CPU_BACKOFF=$CPU_BACKOFF
		timeout 60s ./bank -n 14 -b 640 -x 384 -a $DATASET -d $DURATION -R 0 -S $TRANSACTION_SIZE -l $PROB_WRITE -N 10 -T 16 CPU_BACKOFF=$CPU_BACKOFF
		timeout 60s ./bank -n 14 -b 640 -x 256 -a $DATASET -d $DURATION -R 0 -S $TRANSACTION_SIZE -l $PROB_WRITE -N 10 -T 32 CPU_BACKOFF=$CPU_BACKOFF
		timeout 60s ./bank -n 14 -b 768 -x 256 -a $DATASET -d $DURATION -R 0 -S $TRANSACTION_SIZE -l $PROB_WRITE -N 10 -T 32 CPU_BACKOFF=$CPU_BACKOFF
		timeout 60s ./bank -n 14 -b 640 -x 384 -a $DATASET -d $DURATION -R 0 -S $TRANSACTION_SIZE -l $PROB_WRITE -N 10 -T 32 CPU_BACKOFF=$CPU_BACKOFF
		timeout 60s ./bank -n 14 -b 768 -x 384 -a $DATASET -d $DURATION -R 0 -S $TRANSACTION_SIZE -l $PROB_WRITE -N 10 -T 32 CPU_BACKOFF=$CPU_BACKOFF
		### TODO: larger batches
		###
		mv Bank.csv ${1}${s}
	done
}

function doRunLargeDTST_CPU_1 {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		# 100M 500M 1G 1.5G
		timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 16 -l $PROB_WRITE \
			-N 1 -T 32 CPU_BACKOFF=$CPU_BACKOFF
		tail -n 1 Bank.csv > /tmp/BankLastLine.csv
		for i in `seq 1 7`
		do
			cat /tmp/BankLastLine.csv >> Bank.csv
		done
		mv Bank.csv ${1}${s}
	done
}

function doRunLargeDTST_CPU_100 {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		# 100M 500M 1G 1.5G
		timeout 60s ./bank -n 14 -b 1 -x 1 -a $DATASET -d $DURATION -R 0 -S 16 -l $PROB_WRITE \
			-N 100 -T 32 CPU_BACKOFF=$CPU_BACKOFF
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
# make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=13 >/dev/null
# doRunLargeDTST_1 GPUonly_rand_sep_DISABLED_1_s
# doRunLargeDTST_100 GPUonly_rand_sep_DISABLED_100_s
#
# make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=13 OVERLAP_CPY_BACK=1 >/dev/null
# doRunLargeDTST_1 GPUonly_rand_sep_DISABLED_OVERLAP_1_s
# doRunLargeDTST_100 GPUonly_rand_sep_DISABLED_OVERLAP_100_s

############## CPU-only
make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	BMAP_GRAN_BITS=13 >/dev/null
doRunLargeDTST_CPU_1 CPUonly_rand_sep_DISABLED_1_s
doRunLargeDTST_CPU_100 CPUonly_rand_sep_DISABLED_100_s

# LOG_SIZE=32768 STM_LOG_BUFFER_SIZE=128

############## VERS
make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	BMAP_GRAN_BITS=13 DISABLE_NON_BLOCKING=1 \
	LOG_SIZE=32768 STM_LOG_BUFFER_SIZE=128 >/dev/null
doRunLargeDTST_1 VERS_BLOC_rand_sep_1_s
doRunLargeDTST_100 VERS_BLOC_rand_sep_100_s

make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	BMAP_GRAN_BITS=13 DISABLE_NON_BLOCKING=0 \
	LOG_SIZE=32768 STM_LOG_BUFFER_SIZE=128 >/dev/null
doRunLargeDTST_1 VERS_NON_BLOC_rand_sep_1_s
doRunLargeDTST_100 VERS_NON_BLOC_rand_sep_100_s

make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	BMAP_GRAN_BITS=13 DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 \
	LOG_SIZE=32768 STM_LOG_BUFFER_SIZE=128 >/dev/null
doRunLargeDTST_1 VERS_NON_BLOC_OVER_rand_sep_1_s
doRunLargeDTST_100 VERS_NON_BLOC_OVER_rand_sep_100_s

############## BMAP
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=1 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
# 	BMAP_GRAN_BITS=13 >/dev/null
# doRunLargeDTST_1 BMAP_rand_sep_1_s
# doRunLargeDTST_100 BMAP_rand_sep_100_s
###########################################################################


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

mkdir -p array_batch_duration_3
mv *_s* array_batch_duration_3/
