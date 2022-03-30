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
SAMPLES=1
#./makeTM.sh

CPU_THREADS=4
LOW_CPU_THREADS=10
HIGH_CPU_THREADS=20

rm -f Bank.csv

# LARGE_DATASET=100000000 #2621440 # 90 000 000 is the max for my home machine
# LARGE_DATASET=20000000 #2621440 # 90 000 000 is the max for my home machine
# LARGE_DATASET_P20=61000000 #2621440 # 90 000 000 is the max for my home machine
# SMALL_DATASET=1000000 #2621440 # 90 000 000 is the max for my home machine
# SMALL_DATASET_P20=122000 #2621440 # 90 000 000 is the max for my home machine

LARGE_DATASET=250000000
SMALL_DATASET=25000000

# function doRunLargeDTST {
# 	# Seq. access, 18 items, prob. write {5..95}, writes 1%
# 	# 1GB
# 	LARGE_DATASET=250000000
# 	for s in `seq $SAMPLES`
# 	do
# 		for blocks in 8 16 32 64 128 192 256 512 640 768 896 1024 1536 2048 3072 4096
# 		do
# 			for threads in 128 192 256 384 512 640 768 896
# 			do
# 				for txs in 1 2 4 8 16
# 				do
# 					timeout 30s ./bank -n $CPU_THREADS -b $blocks -x $threads \
# 						-a $LARGE_DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T $txs CPU_BACKOFF=50
# 				done
# 			done
# 		done
# 		mv Bank.csv ${1}_LARGE_s${s}
# 	done
# 	# 100MB
# 	LARGE_DATASET=25000000
# 	for s in `seq $SAMPLES`
# 	do
# 		for blocks in 8 16 32 64 128 192 256 512 640 768 896 1024 1536 2048 3072 4096
# 		do
# 			for threads in 128 192 256 384 512 640 768 896
# 			do
# 				for txs in 1 2 4 8 16
# 				do
# 					timeout 30s ./bank -n $CPU_THREADS -b $blocks -x $threads \
# 					-a $LARGE_DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 -T $txs CPU_BACKOFF=50
# 				done
# 			done
# 		done
# 		mv Bank.csv ${1}_SMALL_s${s}
# 	done
# }
function doRunSmallDTST {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	DATASET=$SMALL_DATASET
	for s in `seq $SAMPLES`
	do
		timeout 40s ./bank -n $CPU_THREADS -b 8     -x 192  -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 8     -x 256  -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 8     -x 384  -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 8     -x 512  -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 16    -x 384  -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 32    -x 128  -T 2  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 16    -x 384  -T 2  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 32    -x 128  -T 4  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 8     -x 768  -T 4  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 64    -x 128  -T 4  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 8     -x 896  -T 8  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 16    -x 768  -T 8  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 32    -x 768  -T 8  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 256   -x 640  -T 2  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 640   -x 640  -T 2  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 1024  -x 640  -T 2  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 1536  -x 640  -T 2  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 512   -x 768  -T 8  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 768   -x 768  -T 8  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 1024  -x 768  -T 8  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 1536  -x 768  -T 8  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 2048  -x 768  -T 8  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 3072  -x 768  -T 8  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 2048  -x 896  -T 16 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 4096  -x 512  -T 16 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		mv Bank.csv ${1}_small_s${s}
	done
}

function doRunSmallDTST_CPUonly {
	DATASET=$SMALL_DATASET
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq $SAMPLES`
	do
		timeout 40s ./bank -n $CPU_THREADS -b 512 -x 128 -T 1 -a $DATASET -d $DURATION -R 0 -S 4 -l 10 -N 1 CPU_BACKOFF=100
		tail -n 1 Bank.csv > /tmp/Bank_last_line.csv
		for i in $(seq 24)
		do
			cat /tmp/Bank_last_line.csv >> Bank.csv
		done
		mv Bank.csv ${1}_small_s${s}
	done
}

function doRunLargeDTST {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	# LARGE_DATASET=250000000
	# LARGE_DATASET=25000000
	DATASET=$LARGE_DATASET
	for s in `seq $SAMPLES`
	do
		timeout 40s ./bank -n $CPU_THREADS -b 8     -x 128  -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 8     -x 192  -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 16    -x 128  -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 16    -x 192  -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 8     -x 512  -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 16    -x 384  -T 1  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 32    -x 128  -T 2  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 16    -x 256  -T 4  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 64    -x 128  -T 4  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 16    -x 768  -T 8  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 32    -x 768  -T 8  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 256   -x 640  -T 2  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 512   -x 640  -T 2  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 768   -x 640  -T 2  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 1024  -x 640  -T 2  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 1536  -x 640  -T 2  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 2048  -x 640  -T 2  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 3072  -x 640  -T 2  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 4096  -x 640  -T 2  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 1024  -x 768  -T 8  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 1536  -x 768  -T 8  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 2048  -x 768  -T 8  -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 1536  -x 896  -T 16 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 2048  -x 896  -T 16 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		timeout 40s ./bank -n $CPU_THREADS -b 4096  -x 512  -T 16 -a $DATASET -d $DURATION -R 0 -S 16 -l 10 -N 1 CPU_BACKOFF=50
		mv Bank.csv ${1}_large_s${s}
	done
}

function doRunLargeDTST_CPUonly {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	DATASET=$LARGE_DATASET
	for s in `seq $SAMPLES`
	do
		timeout 40s ./bank -n $CPU_THREADS -b 512 -x 128 -T 1 -a $DATASET -d $DURATION -R 0 -S 4 -l 10 -N 1 CPU_BACKOFF=100
		tail -n 1 Bank.csv > /tmp/Bank_last_line.csv
		for i in $(seq 24)
		do
			cat /tmp/Bank_last_line.csv >> Bank.csv
		done
		mv Bank.csv ${1}_large_s${s}
	done
}

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

# small     large
# 1536       1024
# 2048       1536
# 3072       2048
# 4096       3072
# 6144       4096
# 8192       6144
# 12288      8192
# 16384      16384
# 24576      32768
# 32768      98304
# 57344      196608
# 98304      327680
# 196608     655360
# 327680     983040
# 819200     1310720
# 1310720    1966080
# 1966080    2621440
# 3145728    3932160
# 4718592    5242880
# 6291456    6291456
# 9437184    9437184
# 12582912   12582912
# 18874368   22020096
# 29360128   29360128
# 33554432   33554432




























# 200MB
CPU_THREADS=14
# LARGE_DATASET=400000000

# ############### GPU-only
# make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 GPU_PART=0.9 \
# 	CPU_PART=0.1 P_INTERSECT=0.00 BANK_PART=1 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.00
# doRunLargeDTST GPUonly_cont_sep_BMAP
# doRunSmallDTST GPUonly_cont_sep_BMAP
#
# make clean ; make CMP_TYPE=DISABLED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 GPU_PART=0.9 \
# 	CPU_PART=0.1 P_INTERSECT=0.00 BANK_PART=1 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.00 USE_UNIF_MEM=1
# doRunLargeDTST GPUonly_cont_sep_DISABLED_UNIF_MEM
# doRunSmallDTST GPUonly_cont_sep_DISABLED_UNIF_MEM
#
# make clean ; make CMP_TYPE=DISABLED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 GPU_PART=0.9 \
# 	CPU_PART=0.1 P_INTERSECT=0.00 BANK_PART=1 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.00
# doRunLargeDTST GPUonly_cont_sep_DISABLED
# doRunSmallDTST GPUonly_cont_sep_DISABLED

# ############### CPU-only
make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=0 GPUEn=0 INST_CPU=0 \
	PR_MAX_RWSET_SIZE=20 GPU_PART=0.9 CPU_PART=0.1 P_INTERSECT=0.00 BANK_PART=1 \
	PROFILE=1 -j 14 BANK_INTRA_CONFL=0.00
doRunLargeDTST_CPUonly CPUonly_cont_sep_DISABLED_STM
doRunSmallDTST_CPUonly CPUonly_cont_sep_DISABLED_STM

make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 GPUEn=0 INST_CPU=0 \
	PR_MAX_RWSET_SIZE=20 GPU_PART=0.9 CPU_PART=0.1 P_INTERSECT=0.00 BANK_PART=1 \
	PROFILE=1 -j 14 BANK_INTRA_CONFL=0.00
doRunLargeDTST_CPUonly CPUonly_cont_sep_DISABLED
doRunSmallDTST_CPUonly CPUonly_cont_sep_DISABLED
#
# ############## VERS
# ### STM
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=1 GPU_PART=0.9 CPU_PART=0.1 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.00
# doRunLargeDTST VERS_cont_sep_STM_s
#
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=1 GPU_PART=0.9 CPU_PART=0.1 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.00
# doRunLargeDTST VERS_cont_sep_s
#
# ############## BMAP
# ### STM
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=0 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=1 GPU_PART=0.9 CPU_PART=0.1 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.00
# doRunLargeDTST BMAP_cont_sep_STM_s
#
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
# 	BANK_PART=1 GPU_PART=0.9 CPU_PART=0.1 P_INTERSECT=0.00 PROFILE=1 -j 14 BANK_INTRA_CONFL=0.00
# doRunLargeDTST BMAP_cont_sep_s
# ###########################################################################
