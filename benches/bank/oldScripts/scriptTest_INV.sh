#!/bin/bash

GPU_PART="0.6"
CPU_PART="0.4"
# P_INTERSECT="0.0 0.1 0.5 0.9 1.0"
P_INTERSECT="0.0"
# GPU_THREADS="4 8 16 32 64 128 256 512 1024"
# BATCH_SIZE="1 2 4 8 16 32 64 128 256"
# DATASET="100000 500000 1000000 5000000 10000000 50000000 100000000"
DATASET="1000000 10000000 100000000 400000000"
BLOCKS="64 512 1024"
# THREADS="1 2 4 6 8 10 12 14 16 22 28 42 56"
THREADS="4 8 14"
# THR_GPU_THREADS="4 1024"
# THR_BATCH_SIZE="1 256"
THR_BLOCKS="4 1024"
GPU_THREADS="4 14"

DEFAULT_DATASET="10000000"
DEFAULT_BATCH="40"
DEFAULT_GPU_THREADS="512"
DEFAULT_BLOCKS="1024"
DEFAULT_THREADS="42"

#"1 2 4 6 8 10 12 14 16 22 30 42 56"
DURATION=5000
SAMPLES=5

rm -f Bank.csv

### TODO: comment this
# sleep 2h

t=$DEFAULT_GPU_THREADS
T=$DEFAULT_BATCH

function doExperiment {
	# a=$DEFAULT_DATASET
	# n=$DEFAULT_THREADS
	# TARGET_B=1
	# eval "TARGET_B=\$$2"
	# for b in $TARGET_B
	# do
	# 	timeout 20s ./bank -n $n -a $a -d $DURATION -b $b -x $t -T $T
	# done
	# mv Bank.csv ${1}_BLKS_s${s}
	b=$DEFAULT_BLOCKS
	n=$DEFAULT_THREADS
	TARGET_A=1
	eval "TARGET_A=\$$3"
	for a in $TARGET_A
	do
		timeout 20s ./bank -n $n -a $a -d $DURATION -b $b -x $t -T $T
	done
	mv Bank.csv ${1}_DTST_s${s}
	# a=$DEFAULT_DATASET
	# b=$DEFAULT_BLOCKS
	# TARGET_N=1
	# eval "TARGET_N=\$$4"
	# for n in $TARGET_N
	# do
	# 	timeout 20s ./bank -n $n -a $a -d $DURATION -b $b -x $t -T $T
	# done
	# mv Bank.csv ${1}_THRS_s${s}
}

for s in `seq $SAMPLES`
do
	##################################
	# GPU-only
	make clean ; make CMP_TYPE=DISABLED GPU_PART=1 CPU_PART=0 \
		P_INTERSECT=0.0 CPUEn=0 USE_TSX_IMPL=0 BANK_PART=1 PROFILE=1 BENCH=BANK -j 14
	doExperiment GPU_NO_HACKS BLOCKS DATASET GPU_THREADS
	##################################

	##################################
	# TSX-only
	make clean ; make CMP_TYPE=DISABLED GPU_PART=0.0 CPU_PART=1.0 \
		P_INTERSECT=0.0 GPUEn=0 USE_TSX_IMPL=1 INST_CPU=0 BANK_PART=1 PROFILE=1 BENCH=BANK -j 14
	doExperiment TSX_only THR_BLOCKS DATASET THREADS
	##################################

	##################################
	# Tiny-only
	# make clean ; make CMP_TYPE=DISABLED GPU_PART=$GPU_PART CPU_PART=$CPU_PART \
	# 	P_INTERSECT=0.5 GPUEn=0 USE_TSX_IMPL=0 INST_CPU=0 BANK_PART=1 PROFILE=1 BENCH=BANK -j 14
	# doExperiment Tiny_only THR_BLOCKS DATASET THREADS
	##################################

	for i in $P_INTERSECT
	do
		##################################
		# TSX --- CPU_INV
		# make clean ; make CMP_TYPE=COMPRESSED GPU_PART=$GPU_PART CPU_PART=$CPU_PART \
		# 	P_INTERSECT=$i CPU_INV=1 USE_TSX_IMPL=1 BANK_PART=1 -j 14
		# doExperiment Tiny_only BLOCKS DATASET THREADS
		##################################

		##################################
		# TSX --- GPU_INV
		make clean ; make CMP_TYPE=COMPRESSED GPU_PART=$GPU_PART CPU_PART=$CPU_PART \
			P_INTERSECT=$i CPU_INV=0 USE_TSX_IMPL=1 BANK_PART=1 LOG_TYPE=VERS PROFILE=1 BENCH=BANK -j 14
		doExperiment HeTM_TSX_VERS_$i BLOCKS DATASET THREADS

		make clean ; make CMP_TYPE=COMPRESSED GPU_PART=$GPU_PART CPU_PART=$CPU_PART \
			P_INTERSECT=$i CPU_INV=0 USE_TSX_IMPL=1 BANK_PART=1 LOG_TYPE=VERS2 PROFILE=1 BENCH=BANK -j 14
		doExperiment HeTM_TSX_VERS2_$i BLOCKS DATASET THREADS
		##################################

		##################################
		# TSX --- GPU_INV
		make clean ; make CMP_TYPE=COMPRESSED GPU_PART=$GPU_PART CPU_PART=$CPU_PART \
			P_INTERSECT=$i CPU_INV=0 USE_TSX_IMPL=1 BANK_PART=1 LOG_TYPE=ADDR PROFILE=1 BENCH=BANK -j 14
		doExperiment HeTM_TSX_ADDR_$i BLOCKS DATASET THREADS
		##################################

		##################################
		# TSX --- GPU_INV
		make clean ; make CMP_TYPE=COMPRESSED GPU_PART=$GPU_PART CPU_PART=$CPU_PART \
			P_INTERSECT=$i CPU_INV=0 USE_TSX_IMPL=1 BANK_PART=1 LOG_TYPE=BMAP PROFILE=1 BENCH=BANK -j 14
		doExperiment HeTM_TSX_BMAP_$i BLOCKS DATASET THREADS
		##################################

		##################################
		# Tiny --- CPU_INV
		# make clean ; make CMP_TYPE=COMPRESSED GPU_PART=$GPU_PART CPU_PART=$CPU_PART \
		# 	P_INTERSECT=$i CPU_INV=1 USE_TSX_IMPL=0 BANK_PART=1 -j 14
		# doExperiment Tiny_only BLOCKS DATASET THREADS
		##################################

		##################################
		# # Tiny --- GPU_INV
		# make clean ; make CMP_TYPE=COMPRESSED GPU_PART=$GPU_PART CPU_PART=$CPU_PART \
		# 	P_INTERSECT=$i CPU_INV=0 USE_TSX_IMPL=0 BANK_PART=1 LOG_TYPE=VERS PROFILE=1 BENCH=BANK -j 14
		# doExperiment HeTM_Tiny_VERS_$i BLOCKS DATASET THREADS
		##################################

		##################################
		# # Tiny --- GPU_INV
		# make clean ; make CMP_TYPE=COMPRESSED GPU_PART=$GPU_PART CPU_PART=$CPU_PART \
		# 	P_INTERSECT=$i CPU_INV=0 USE_TSX_IMPL=0 BANK_PART=1 LOG_TYPE=ADDR PROFILE=1 BENCH=BANK -j 14
		# doExperiment HeTM_Tiny_ADDR_$i BLOCKS DATASET THREADS
		##################################

		##################################
		# # Tiny --- GPU_INV
		# make clean ; make CMP_TYPE=COMPRESSED GPU_PART=$GPU_PART CPU_PART=$CPU_PART \
		# 	P_INTERSECT=$i CPU_INV=0 USE_TSX_IMPL=0 BANK_PART=1 LOG_TYPE=BMAP PROFILE=1 BENCH=BANK -j 14
		# doExperiment HeTM_Tiny_BMAP_$i BLOCKS DATASET THREADS
		##################################
	done
done
