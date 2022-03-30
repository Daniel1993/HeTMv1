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
BLOCKS="10" # 512
THREADS="128" #"2 4 8 16 32 64 96 256 320 512 640 768 1024"
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
	LARGE_DATASET=250000000
	for s in `seq $SAMPLES`
	do
		for blocks in 20 40 80 160 320 640
		do
			#for threads in 32 64 128 256 512 640 768 896 1024
			for threads in 128 192 256 384 512 768 1024
			do
				for txs in 1
				do
					timeout 30s ./bank -n $CPU_THREADS -b $blocks -x $threads \
						-a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 1 -T $txs CPU_BACKOFF=1
				done
			done
		done
		mv Bank.csv ${1}_LARGE_s${s}
	done
	# LARGE_DATASET=25000000
	# for s in `seq $SAMPLES`
	# do
	# 	for blocks in 20 40 80 160 320 640 1280 2560
	# 	do
	# 		for threads in 32 64 128 256 512 640 768 896 1024
	# 		do
	# 			for txs in 1 2 4 8 16 32
	# 			do
	# 				timeout 30s ./bank -n $CPU_THREADS -b $blocks -x $threads \
	# 				-a $LARGE_DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T $txs CPU_BACKOFF=50
	# 			done
	# 		done
	# 	done
	# 	mv Bank.csv ${1}_SMALL_s${s}
	# done
}
# function doRunSmallDTST {
# 	# Seq. access, 18 items, prob. write {5..95}, writes 1%
# 	for s in `seq $SAMPLES`
# 	do
# 		timeout 40s ./bank -n $CPU_THREADS -b 8     -x 128   -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 8     -x 192   -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 16    -x 128   -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 8     -x 384   -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 8     -x 512   -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 8     -x 768   -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 16    -x 512   -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 16    -x 640   -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 16    -x 640   -T 2  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 64    -x 640   -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 32    -x 1024  -T 4  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 64    -x 896   -T 4  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 640   -x 896   -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 1024  -x 896   -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 1536  -x 896   -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 2048  -x 896   -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 3072  -x 896   -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 2048  -x 896   -T 2  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 768   -x 896   -T 8  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 4096  -x 896   -T 2  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 3072  -x 896   -T 4  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 1024  -x 896   -T 16 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 3072  -x 896   -T 8  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 2048  -x 896   -T 16 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 4096  -x 1024  -T 8  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 3072  -x 1024  -T 16 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 4096  -x 1024  -T 16 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		mv Bank.csv ${1}${s}
# 	done
# }
# function doRunLargeDTST {
# 	# Seq. access, 18 items, prob. write {5..95}, writes 1%
# 	for s in `seq $SAMPLES`
# 	do
# 		timeout 40s ./bank -n $CPU_THREADS -b 8    -x 128  -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 8    -x 192  -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 16   -x 128  -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 8    -x 384  -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 16   -x 256  -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 8    -x 768  -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 8    -x 640  -T 2  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 8    -x 640  -T 4  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 32   -x 1024 -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 64   -x 896  -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 64   -x 896  -T 2  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 64   -x 896  -T 4  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 32   -x 1024 -T 16 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 64   -x 896  -T 16 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 2048 -x 896  -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 3072 -x 896  -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 640  -x 896  -T 8  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 4096 -x 896  -T 2  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 3072 -x 896  -T 4  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 4096 -x 896  -T 4  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 3072 -x 896  -T 8  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 4096 -x 896  -T 8  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 3072 -x 1024 -T 16 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		timeout 40s ./bank -n $CPU_THREADS -b 4096 -x 1024 -T 16 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		mv Bank.csv ${1}${s}
# 	done
# }
# # GPU_BLOCKS(3)	GPU_THREADS_PER_BLOCK(4)	TXs_PER_GPU_THREAD(5)
#
# function doRunSmallDTST_CPUonly {
# 	# Seq. access, 18 items, prob. write {5..95}, writes 1%
# 	for s in `seq $SAMPLES`
# 	do
# 		timeout 40s ./bank -n $CPU_THREADS -b 512 -x 128 -T 1 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		tail -n 1 Bank.csv > /tmp/Bank_last_line.csv
# 		for i in $(seq 26)
# 		do
# 			cat /tmp/Bank_last_line.csv >> Bank.csv
# 		done
# 		mv Bank.csv ${1}${s}
# 	done
# }
#
# function doRunLargeDTST_CPUonly {
# 	# Seq. access, 18 items, prob. write {5..95}, writes 1%
# 	for s in `seq $SAMPLES`
# 	do
# 		timeout 40s ./bank -n $CPU_THREADS -b 512 -x 128 -T 1 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
# 		tail -n 1 Bank.csv > /tmp/Bank_last_line.csv
# 		for i in $(seq 23)
# 		do
# 			cat /tmp/Bank_last_line.csv >> Bank.csv
# 		done
# 		mv Bank.csv ${1}${s}
# 	done
# }

################ Contiguos

############### LARGE
###########################################################################
# LARGE_DATASET=49999999 # half the size
# CPU_THREADS=4
#
# cd java_zipf
# GPU_INPUT="GPU_input_${LARGE_DATASET}_099_4194304.txt"
# CPU_INPUT="CPU_input_${LARGE_DATASET}_099_57344.txt"
#
# if [ ! -f ../$GPU_INPUT ]
# then
# 	java Main $LARGE_DATASET 0.99 4194304 > ../$GPU_INPUT
# fi
# if [ ! -f ../$CPU_INPUT ]
# then
# 	java Main $LARGE_DATASET 0.99 57344 > ../$CPU_INPUT
# fi
# cd ..
#
# LARGE_DATASET=100000000 # half the size

# 200MB
CPU_THREADS=14
# LARGE_DATASET=400000000
LARGE_DATASET=250000000
SMALL_DATASET=25000000

############### GPU-only
make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 GPU_PART=0.9 \
	CPU_PART=0.1 P_INTERSECT=0.00 BANK_PART=5 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.00
DATASET=$LARGE_DATASET
doRunLargeDTST GPUonly_cont_sep_BMAP_large_s
# DATASET=$SMALL_DATASET
# doRunSmallDTST GPUonly_cont_sep_BMAP_small_s
#
# make clean ; make CMP_TYPE=DISABLED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 GPU_PART=0.9 \
# 	CPU_PART=0.1 P_INTERSECT=0.00 BANK_PART=5 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.00 USE_UNIF_MEM=1
# DATASET=$LARGE_DATASET
# doRunLargeDTST GPUonly_cont_sep_DISABLED_UNIF_MEM_large_s
# DATASET=$SMALL_DATASET
# doRunSmallDTST GPUonly_cont_sep_DISABLED_UNIF_MEM_small_s
#
# make clean ; make CMP_TYPE=DISABLED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 GPU_PART=0.9 \
# 	CPU_PART=0.1 P_INTERSECT=0.00 BANK_PART=5 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.00
# DATASET=$LARGE_DATASET
# doRunLargeDTST GPUonly_cont_sep_DISABLED_large_s
# DATASET=$SMALL_DATASET
# doRunSmallDTST GPUonly_cont_sep_DISABLED_small_s
#
# ############### CPU-only
# make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=0 GPUEn=0 INST_CPU=0 \
# 	PR_MAX_RWSET_SIZE=20 GPU_PART=0.9 CPU_PART=0.1 P_INTERSECT=0.00 BANK_PART=5 \
# 	PROFILE=1 -j 14 BANK_INTRA_CONFL=0.00
# DATASET=$LARGE_DATASET
# doRunLargeDTST_CPUonly CPUonly_cont_sep_DISABLED_STM_large_s
# DATASET=$SMALL_DATASET
# doRunSmallDTST_CPUonly CPUonly_cont_sep_DISABLED_STM_small_s
#
# make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 GPUEn=0 INST_CPU=0 \
# 	PR_MAX_RWSET_SIZE=20 GPU_PART=0.9 CPU_PART=0.1 P_INTERSECT=0.00 BANK_PART=5 \
# 	PROFILE=1 -j 14 BANK_INTRA_CONFL=0.00
# DATASET=$LARGE_DATASET
# doRunLargeDTST_CPUonly CPUonly_cont_sep_DISABLED_large_s
# DATASET=$SMALL_DATASET
# doRunSmallDTST_CPUonly CPUonly_cont_sep_DISABLED_small_s
#
# ############## VERS
# ### STM
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=5 GPU_PART=0.9 CPU_PART=0.1 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.00
# DATASET=$LARGE_DATASET
# doRunLargeDTST VERS_cont_sep_STM_large_s
# DATASET=$SMALL_DATASET
# doRunSmallDTST VERS_cont_sep_STM_small_s
#
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=5 GPU_PART=0.9 CPU_PART=0.1 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.00
# DATASET=$LARGE_DATASET
# doRunLargeDTST VERS_cont_sep_large_s
# DATASET=$SMALL_DATASET
# doRunSmallDTST VERS_cont_sep_small_s
#
# ############## BMAP
# ### STM
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=5 GPU_PART=0.9 CPU_PART=0.1 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.00
# DATASET=$LARGE_DATASET
# doRunLargeDTST BMAP_cont_sep_STM_large_s
# DATASET=$SMALL_DATASET
# doRunSmallDTST BMAP_cont_sep_STM_small_s
#
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=5 GPU_PART=0.9 CPU_PART=0.1 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.00
# DATASET=$LARGE_DATASET
# doRunLargeDTST BMAP_cont_sep_large_s
# DATASET=$SMALL_DATASET
# doRunSmallDTST BMAP_cont_sep_small_s
# ###########################################################################

mkdir -p array_batch_size
mv *_s* array_batch_size
