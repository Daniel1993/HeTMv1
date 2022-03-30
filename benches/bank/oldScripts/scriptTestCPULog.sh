#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1
filename_tsx="Bank_TSX"
filename_tiny="Bank_Tiny"

GPU_PART="0.0"
CPU_PART="1.0"
P_INTERSECT="0.0"
DURATION=5000
DATASET="4096 8192 32768 16384 65536 131072 262144" #2621440 # 90 000 000 is the max for my home machine
# CPU_THREADS="1 2 3 4"
CPU_THREADS="1 2 4 6 8 10 12 14 16 22 28 42 56"
BATCH_SIZE="1 2 4 8 16 32 64 128 256 512 1024"
DEFAULT_BATCH_SIZE="4"
# SMALL_DATASET="8192"
# LARGE_DATASET="131072"
SMALL_DATASET="1000000"
LARGE_DATASET="20000000"
DEFAULT_BLOCKS="512"
DEFAULT_THREADS="512"
DEFAULT_CPU_THREADS="56"
SAMPLES=5
#./makeTM.sh

b=$DEFAULT_BLOCKS
T=$DEFAULT_BATCH_SIZE
t=$DEFAULT_THREADS
rm -f Bank.csv

function doRunBench {
	for s in `seq 2 $SAMPLES`
	do
		a=$SMALL_DATASET
		for n in $CPU_THREADS
		do
			timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -l 100 -S 2 -N 1 -T $T -n $n
		done
		mv Bank.csv ${1}_small_LL_THRS_s${s}
		for n in $CPU_THREADS
		do
			timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -l 100 -S 2 -N 100 -T $T -n $n
		done
		mv Bank.csv ${1}_small_HL_THRS_s${s}
		a=$LARGE_DATASET
		for n in $CPU_THREADS
		do
			timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -l 100 -S 2 -N 1 -T $T -n $n
		done
		mv Bank.csv ${1}_large_LL_THRS_s${s}
		for n in $CPU_THREADS
		do
			timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -l 100 -S 2 -N 100 -T $T -n $n
		done
		mv Bank.csv ${1}_large_HL_THRS_s${s}
		# n=$DEFAULT_CPU_THREADS
		# for a in $DATASET
		# do
		# 	timeout 20s ./bank -b $b -x $t -a $a -d $DURATION -T $T -n $n
		# done
		# mv Bank.csv ${1}_DTST_s${s}
	done
}

###############################################################################
### TinySTM

### VERS W LOG
make clean ; make LOG_TYPE=VERS BANK_PART=1 GPUEn=0 INST_CPU=1 -j 14
doRunBench CPU_w_LOG_Tiny_VERS

make clean ; make LOG_TYPE=VERS BANK_PART=2 GPUEn=0 INST_CPU=1 -j 14
doRunBench CPU_w_LOG_Tiny_VERS_UR

# make clean ; make LOG_TYPE=VERS2 GPU_PART=$GPU_PART CPU_PART=$CPU_PART \
# 	P_INTERSECT=$P_INTERSECT BANK_PART=2 GPUEn=0 INST_CPU=1 -j 14
# doRunBench CPU_w_CHUNKED_LOG_Tiny_VERS

### ADDR W LOG
# make clean ; make LOG_TYPE=ADDR BANK_PART=2 GPUEn=0 INST_CPU=1 -j 14
# doRunBench CPU_w_LOG_Tiny_ADDR

### ADDR W CHUNKED LOG
# make clean ; make LOG_TYPE=ADDR GPU_PART=$GPU_PART CPU_PART=$CPU_PART \
# 	P_INTERSECT=$P_INTERSECT BANK_PART=2 GPUEn=0 INST_CPU=1 -j 14
# doRunBench CPU_w_CHUNKED_LOG_Tiny_ADDR

### BMAP W LOG
make clean ; make LOG_TYPE=BMAP BANK_PART=1 GPUEn=0 INST_CPU=1 -j 14
doRunBench CPU_w_LOG_Tiny_BMAP

make clean ; make LOG_TYPE=BMAP BANK_PART=2 GPUEn=0 INST_CPU=1 -j 14
doRunBench CPU_w_LOG_Tiny_BMAP_UR

### BMAP W CHUNKED LOG
# make clean ; make LOG_TYPE=BMAP GPU_PART=$GPU_PART CPU_PART=$CPU_PART \
# 	P_INTERSECT=$P_INTERSECT BANK_PART=2 GPUEn=0 INST_CPU=1 -j 14
# doRunBench CPU_w_CHUNKED_LOG_Tiny_BMAP

### NO LOG
make clean ; make LOG_TYPE=VERS BANK_PART=1 GPUEn=0 INST_CPU=0 -j 14
doRunBench CPU_w_o_LOG_Tiny

make clean ; make LOG_TYPE=VERS BANK_PART=2 GPUEn=0 INST_CPU=0 -j 14
doRunBench CPU_w_o_LOG_Tiny_UR
###############################################################################

###############################################################################
### TSX

### VERS W LOG
make clean ; make LOG_TYPE=VERS BANK_PART=1 GPUEn=0 INST_CPU=1 USE_TSX_IMPL=1 -j 14
doRunBench CPU_w_LOG_TSX_VERS

make clean ; make LOG_TYPE=VERS BANK_PART=2 GPUEn=0 INST_CPU=1 USE_TSX_IMPL=1 -j 14
doRunBench CPU_w_LOG_TSX_VERS_UR

# make clean ; make LOG_TYPE=VERS2 GPU_PART=$GPU_PART CPU_PART=$CPU_PART \
# 	P_INTERSECT=$P_INTERSECT BANK_PART=2 GPUEn=0 INST_CPU=1 USE_TSX_IMPL=1 -j 14


### ADDR W LOG
# make clean ; make LOG_TYPE=ADDR GPU_PART=$GPU_PART CPU_PART=$CPU_PART \
# 	P_INTERSECT=$P_INTERSECT BANK_PART=2 GPUEn=0 INST_CPU=1 USE_TSX_IMPL=1 -j 14
# doRunBench CPU_w_LOG_TSX_ADDR

### ADDR W CHUNKED LOG
# make clean ; make LOG_TYPE=ADDR GPU_PART=$GPU_PART CPU_PART=$CPU_PART \
# 	P_INTERSECT=$P_INTERSECT BANK_PART=2 GPUEn=0 INST_CPU=1 USE_TSX_IMPL=1 -j 14


### BMAP W LOG
make clean ; make LOG_TYPE=BMAP BANK_PART=1 GPUEn=0 INST_CPU=1 USE_TSX_IMPL=1 -j 14
doRunBench CPU_w_LOG_TSX_BMAP

make clean ; make LOG_TYPE=BMAP BANK_PART=2 GPUEn=0 INST_CPU=1 USE_TSX_IMPL=1 -j 14
doRunBench CPU_w_LOG_TSX_BMAP_UR

### BMAP W CHUNKED LOG
# make clean ; make LOG_TYPE=BMAP GPU_PART=$GPU_PART CPU_PART=$CPU_PART \
# 	P_INTERSECT=$P_INTERSECT BANK_PART=2 GPUEn=0 INST_CPU=1 USE_TSX_IMPL=1 -j 14


### NO LOG
make clean ; make LOG_TYPE=VERS BANK_PART=1 GPUEn=0 INST_CPU=0 USE_TSX_IMPL=1 -j 14
doRunBench CPU_w_o_LOG_TSX

make clean ; make LOG_TYPE=VERS BANK_PART=2 GPUEn=0 INST_CPU=0 USE_TSX_IMPL=1 -j 14
doRunBench CPU_w_o_LOG_TSX_UR
###############################################################################
