#!/bin/bash

HMULT="0.01 0.05 0.10 0.15"
HMULT2="0.02 0.10 0.20 0.30"
THR_BLOCKS="4 1024"
GPU_THREADS="4 14"
# THR_GPU_THREADS="4 1024"
# THR_BATCH_SIZE="1 256"

DEFAULT_DATASET="50000000"
DEFAULT_HPROB="90"
DEFAULT_BATCH="4"
DEFAULT_GPU_THREADS="512"
DEFAULT_BLOCKS="1024"
DEFAULT_THREADS="14"

#"1 2 4 6 8 10 12 14 16 22 30 42 56"
DURATION=5000
SAMPLES=3

rm -f Bank.csv

### TODO: comment this
# sleep 2h

t=$DEFAULT_GPU_THREADS
T=$DEFAULT_BATCH

function doExperiment {
	for s in `seq $SAMPLES`
	do
		a=$DEFAULT_DATASET
		n=$DEFAULT_THREADS
		b=$DEFAULT_BLOCKS
		TARGET_HMULT=1
		eval "TARGET_HMULT=\$$2"
		for h in $TARGET_HMULT
		do
			timeout 20s ./bank -n $n -a $a -d $DURATION -b $b -x $t -T $T -s $h -m $DEFAULT_HPROB
		done
		mv Bank.csv ${1}_HMULT_s${s}
	done
}

##################################
# GPU-only
make clean ; make CMP_TYPE=DISABLED CPUEn=0 USE_TSX_IMPL=1 GPU_PART=0.45 CPU_PART=0.45 P_INTERSECT=0.0 \
	BANK_PART=2 PROFILE=1 BENCH=BANK -j 14 >/dev/null
doExperiment GPU_NO_HACKS HMULT2
##################################

##################################
# GPU-only (OVERLAP)
make clean ; make CMP_TYPE=DISABLED CPUEn=0 USE_TSX_IMPL=1 GPU_PART=0.45 CPU_PART=0.45 P_INTERSECT=0.0 \
	BANK_PART=2 PROFILE=1 BENCH=BANK OVERLAP_CPY_BACK=1 -j 14 >/dev/null
doExperiment GPU_NO_HACKS_OVERLAP HMULT2
##################################

##################################
# TSX-only
make clean ; make CMP_TYPE=DISABLED GPUEn=0 USE_TSX_IMPL=1 GPU_PART=0.45 CPU_PART=0.45 P_INTERSECT=0.0 INST_CPU=0 \
	BANK_PART=2 PROFILE=1 BENCH=BANK -j 14 >/dev/null
doExperiment TSX_only HMULT2
##################################

# #################################
# Tiny-only
# make clean ; make CMP_TYPE=DISABLED GPU_PART=$GPU_PART CPU_PART=$CPU_PART \
# 	P_INTERSECT=0.5 GPUEn=0 USE_TSX_IMPL=0 INST_CPU=0 BANK_PART=1 PROFILE=1 BENCH=BANK -j 14
# doExperiment Tiny_only THR_BLOCKS DATASET THREADS
# #################################
#
# #################################
# TSX --- CPU_INV
# make clean ; make CMP_TYPE=COMPRESSED GPU_PART=$GPU_PART CPU_PART=$CPU_PART \
# 	P_INTERSECT=$i CPU_INV=1 USE_TSX_IMPL=1 BANK_PART=1 -j 14
# doExperiment Tiny_only BLOCKS DATASET THREADS
# #################################

##################################
# TSX --- GPU_INV
make clean ; make CMP_TYPE=COMPRESSED CPU_INV=0 USE_TSX_IMPL=1 GPU_PART=0.45 CPU_PART=0.45 P_INTERSECT=0.0 BANK_PART=2 \
	LOG_TYPE=VERS PROFILE=1 BENCH=BANK -j 14 >/dev/null
doExperiment HeTM_TSX_VERS HMULT

make clean ; make CMP_TYPE=COMPRESSED CPU_INV=0 USE_TSX_IMPL=1 GPU_PART=0.45 CPU_PART=0.45 P_INTERSECT=0.0 BANK_PART=2 \
	LOG_TYPE=VERS2 PROFILE=1 BENCH=BANK -j 14 >/dev/null
doExperiment HeTM_TSX_VERS2 HMULT
##################################

##################################
# TSX --- GPU_INV (OVERLAP)
make clean ; make CMP_TYPE=COMPRESSED CPU_INV=0 USE_TSX_IMPL=1 GPU_PART=0.45 CPU_PART=0.45 P_INTERSECT=0.0 BANK_PART=2 \
	LOG_TYPE=VERS PROFILE=1 BENCH=BANK OVERLAP_CPY_BACK=1 -j 14 >/dev/null
doExperiment HeTM_TSX_VERS_OVERLAP HMULT

make clean ; make CMP_TYPE=COMPRESSED CPU_INV=0 USE_TSX_IMPL=1 GPU_PART=0.45 CPU_PART=0.45 P_INTERSECT=0.0 BANK_PART=2 \
	LOG_TYPE=VERS2 PROFILE=1 BENCH=BANK OVERLAP_CPY_BACK=1 -j 14 >/dev/null
doExperiment HeTM_TSX_VERS2_OVERLAP HMULT
##################################

##################################
# TSX --- GPU_INV
make clean ; make CMP_TYPE=COMPRESSED CPU_INV=0 USE_TSX_IMPL=1 GPU_PART=0.45 CPU_PART=0.45 P_INTERSECT=0.0 BANK_PART=2 \
	LOG_TYPE=ADDR PROFILE=1 BENCH=BANK -j 14 >/dev/null
doExperiment HeTM_TSX_ADDR HMULT
##################################

##################################
# TSX --- GPU_INV (OVERLAP)
make clean ; make CMP_TYPE=COMPRESSED CPU_INV=0 USE_TSX_IMPL=1 GPU_PART=0.45 CPU_PART=0.45 P_INTERSECT=0.0 BANK_PART=2 \
	LOG_TYPE=ADDR PROFILE=1 BENCH=BANK OVERLAP_CPY_BACK=1 -j 14 >/dev/null
doExperiment HeTM_TSX_ADDR_OVERLAP HMULT
##################################

##################################
# TSX --- GPU_INV
make clean ; make CMP_TYPE=COMPRESSED CPU_INV=0 USE_TSX_IMPL=1 GPU_PART=0.45 CPU_PART=0.45 P_INTERSECT=0.0 BANK_PART=2 \
	LOG_TYPE=BMAP PROFILE=1 BENCH=BANK -j 14 >/dev/null
doExperiment HeTM_TSX_BMAP HMULT
##################################

##################################
# TSX --- GPU_INV (OVERLAP)
make clean ; make CMP_TYPE=COMPRESSED CPU_INV=0 USE_TSX_IMPL=1 GPU_PART=0.45 CPU_PART=0.45 P_INTERSECT=0.0 BANK_PART=2 \
	LOG_TYPE=BMAP PROFILE=1 BENCH=BANK OVERLAP_CPY_BACK=1 -j 14 >/dev/null
doExperiment HeTM_TSX_BMAP_OVERLAP HMULT
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
