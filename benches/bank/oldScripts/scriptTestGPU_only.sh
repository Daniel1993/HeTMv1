#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1
filename_tsx="Bank_TSX"
filename_tiny="Bank_Tiny"

GPU_PART="1.0"
CPU_PART="0.0"
P_INTERSECT="0.0"
DURATION=8000
BLOCKS="2 4 8 16 32 64 256 512 1024" # 512
THREADS="512" #"2 4 8 16 32 64 96 256 320 512 640 768 1024"
BATCH_SIZE="4"
SAMPLES=1
#./makeTM.sh

CPU_THREADS=4
LOW_CPU_THREADS=14
HIGH_CPU_THREADS=28

rm -f Bank.csv

LARGE_DATASET=20000000 #2621440 # 90 000 000 is the max for my home machine
LARGE_DATASET_P20=61000000 #2621440 # 90 000 000 is the max for my home machine
SMALL_DATASET=1000000 #2621440 # 90 000 000 is the max for my home machine
SMALL_DATASET_P20=122000 #2621440 # 90 000 000 is the max for my home machine

### TODO: find the best combination of parameters
function doRunSmallDTSTLowLocality {
	timeout 12s ./bank -n $CPU_THREADS -l 100 -b 8    -x 512 -a $SMALL_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 1 -T 1024
	timeout 12s ./bank -n $CPU_THREADS -l 100 -b 16   -x 512 -a $SMALL_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 1 -T 512
	timeout 12s ./bank -n $CPU_THREADS -l 100 -b 32   -x 512 -a $SMALL_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 1 -T 256
	timeout 12s ./bank -n $CPU_THREADS -l 100 -b 64   -x 512 -a $SMALL_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 1 -T 128
	timeout 12s ./bank -n $CPU_THREADS -l 100 -b 128  -x 512 -a $SMALL_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 1 -T 64
	timeout 12s ./bank -n $CPU_THREADS -l 100 -b 256  -x 512 -a $SMALL_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 1 -T 32
	timeout 12s ./bank -n $CPU_THREADS -l 100 -b 512  -x 512 -a $SMALL_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 1 -T 16
	timeout 12s ./bank -n $CPU_THREADS -l 100 -b 1024 -x 512 -a $SMALL_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 1 -T 8
}

function doCPURunSmallDTSTLowLocality {
	timeout 12s ./bank -n $CPU_THREADS -l 100 -b 8    -x 512 -a $SMALL_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 1 -T 1024
	timeout 12s ./bank -n $CPU_THREADS -l 100 -b 1024 -x 512 -a $SMALL_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 1 -T 8
}

function doRunSmallDTSTHighLocality {
	timeout 12s ./bank -n $CPU_THREADS -l 100 -b 8    -x 512 -a $SMALL_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 100 -T 1024
	timeout 12s ./bank -n $CPU_THREADS -l 100 -b 16   -x 512 -a $SMALL_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 100 -T 512
	timeout 12s ./bank -n $CPU_THREADS -l 100 -b 32   -x 512 -a $SMALL_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 100 -T 256
	timeout 12s ./bank -n $CPU_THREADS -l 100 -b 64   -x 512 -a $SMALL_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 100 -T 128
	timeout 12s ./bank -n $CPU_THREADS -l 100 -b 128  -x 512 -a $SMALL_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 100 -T 64
	timeout 12s ./bank -n $CPU_THREADS -l 100 -b 256  -x 512 -a $SMALL_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 100 -T 32
	timeout 12s ./bank -n $CPU_THREADS -l 100 -b 512  -x 512 -a $SMALL_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 100 -T 16
	timeout 12s ./bank -n $CPU_THREADS -l 100 -b 1024 -x 512 -a $SMALL_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 100 -T 8
}

function doCPURunSmallDTSTHighLocality {
	timeout 12s ./bank -n $CPU_THREADS -l 100 -b 8    -x 512 -a $SMALL_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 100 -T 1024
	timeout 12s ./bank -n $CPU_THREADS -l 100 -b 1024 -x 512 -a $SMALL_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 100 -T 8
}

function doRunLargeDTSTLowLocality {
	timeout 12s ./bank -n $CPU_THREADS -b 8    -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 1 -T 1024
	timeout 12s ./bank -n $CPU_THREADS -b 16   -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 1 -T 512
	timeout 12s ./bank -n $CPU_THREADS -b 32   -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 1 -T 256
	timeout 12s ./bank -n $CPU_THREADS -b 64   -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 1 -T 128
	timeout 12s ./bank -n $CPU_THREADS -b 128  -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 1 -T 64
	timeout 12s ./bank -n $CPU_THREADS -b 256  -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 1 -T 32
	timeout 12s ./bank -n $CPU_THREADS -b 512  -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 1 -T 16
	timeout 12s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 1 -T 8
}

function doCPURunLargeDTSTLowLocality {
	timeout 12s ./bank -n $CPU_THREADS -b 8    -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 1 -T 1024
	timeout 12s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 1 -T 8
}

function doRunLargeDTSTHighLocality {
	timeout 12s ./bank -n $CPU_THREADS -b 8    -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 100 -T 1024
	timeout 12s ./bank -n $CPU_THREADS -b 16   -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 100 -T 512
	timeout 12s ./bank -n $CPU_THREADS -b 32   -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 100 -T 256
	timeout 12s ./bank -n $CPU_THREADS -b 64   -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 100 -T 128
	timeout 12s ./bank -n $CPU_THREADS -b 128  -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 100 -T 64
	timeout 12s ./bank -n $CPU_THREADS -b 256  -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 100 -T 32
	timeout 12s ./bank -n $CPU_THREADS -b 512  -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 100 -T 16
	timeout 12s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 100 -T 8
}

function doCPURunLargeDTSTHighLocality {
	timeout 12s ./bank -n $CPU_THREADS -b 8    -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 100 -T 1024
	timeout 12s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -R 0 -S 2 -l 100 -N 100 -T 8
}

# LARGE_DATASET=$LARGE_DATASET_P20
# SMALL_DATASET=$SMALL_DATASET_P20
#
# make clean ; make CMP_TYPE=COMPRESSED CPUEn=0 BANK_PART=1 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTSTLowLocality
# 	mv Bank.csv GPUonly_small_BMAP_LL_s${s}
# 	doRunSmallDTSTHighLocality
# 	mv Bank.csv GPUonly_small_BMAP_HL_s${s}
# 	doRunLargeDTSTLowLocality
# 	mv Bank.csv GPUonly_large_BMAP_LL_s${s}
# 	doRunLargeDTSTHighLocality
# 	mv Bank.csv GPUonly_large_BMAP_HL_s${s}
# done
# #
make clean ; make CMP_TYPE=COMPRESSED CPUEn=0 BANK_PART=2 PROFILE=1 -j 14
for s in `seq $SAMPLES`
do
	doRunSmallDTSTLowLocality
	mv Bank.csv GPUonly_small_BMAP_URLL_s${s}
	doRunLargeDTSTLowLocality
	mv Bank.csv GPUonly_large_BMAP_URLL_s${s}
done

# #######################
# ##########################
# #############################
# make clean ; make CMP_TYPE=EXPLICIT BANK_PART=3 GPU_PART=0.52 CPU_PART=0.50 \
# 	P_INTERSECT=0.0 CPUEn=0 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTSTLowLocality
# 	mv Bank.csv GPUonly_small_EXPLICIT_s${s}
# 	doRunSmallDTSTHighLocality
# 	mv Bank.csv GPUonly_small_EXPLICIT_s${s}
# 	doRunLargeDTSTLowLocality
# 	mv Bank.csv GPUonly_large_EXPLICIT_s${s}
# 	doRunLargeDTSTHighLocality
# 	mv Bank.csv GPUonly_large_EXPLICIT_s${s}
# done
# #############################
# ##########################
# #######################

# make clean ; make CMP_TYPE=DISABLED CPUEn=0 BANK_PART=1 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTSTLowLocality
# 	mv Bank.csv GPUonly_low_thrs_small_DISABLED_LL_s${s}
# 	doRunSmallDTSTHighLocality
# 	mv Bank.csv GPUonly_low_thrs_small_DISABLED_HL_s${s}
# 	doRunLargeDTSTLowLocality
# 	mv Bank.csv GPUonly_low_thrs_large_DISABLED_LL_s${s}
# 	doRunLargeDTSTHighLocality
# 	mv Bank.csv GPUonly_low_thrs_large_DISABLED_HL_s${s}
# done
#
make clean ; make CMP_TYPE=DISABLED CPUEn=0 BANK_PART=2 PROFILE=1 -j 14
for s in `seq $SAMPLES`
do
	doRunSmallDTSTLowLocality
	mv Bank.csv GPUonly_low_thrs_small_DISABLED_URLL_s${s}
	doRunLargeDTSTLowLocality
	mv Bank.csv GPUonly_low_thrs_large_DISABLED_URLL_s${s}
done

# #######################
# ##########################
# #############################
### NO PR-STM just for comparison
# make clean ; make CMP_TYPE=DISABLED BANK_PART=3 DISABLE_PRSTM=1 GPU_PART=$GPU_PART \
# 	CPU_PART=$CPU_PART P_INTERSECT=$P_INTERSECT CPUEn=0 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTST
# 	mv Bank.csv GPUonly_small_NO_PRSTM_s${s}
# 	doRunLargeDTST
# 	mv Bank.csv GPUonly_large_NO_PRSTM_s${s}
# done
# #############################
# ##########################
# #######################

CPU_THREADS=$LOW_CPU_THREADS
# make clean ; make INST_CPU=0 GPUEn=0 BANK_PART=1 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doCPURunSmallDTSTLowLocality
# 	mv Bank.csv CPUonly_low_thrs_small_DISABLED_LL_s${s}
# 	doCPURunSmallDTSTHighLocality
# 	mv Bank.csv CPUonly_low_thrs_small_DISABLED_HL_s${s}
# 	doCPURunLargeDTSTLowLocality
# 	mv Bank.csv CPUonly_low_thrs_large_DISABLED_LL_s${s}
# 	doCPURunLargeDTSTHighLocality
# 	mv Bank.csv CPUonly_low_thrs_large_DISABLED_HL_s${s}
# done

make clean ; make INST_CPU=0 GPUEn=0 BANK_PART=2 PROFILE=1 -j 14
for s in `seq $SAMPLES`
do
	doCPURunSmallDTSTLowLocality
	mv Bank.csv CPUonly_low_thrs_small_DISABLED_URLL_s${s}
	doCPURunSmallDTSTHighLocality
	mv Bank.csv CPUonly_low_thrs_large_DISABLED_URLL_s${s}
done

CPU_THREADS=$HIGH_CPU_THREADS
# make clean ; make INST_CPU=0 GPUEn=0 BANK_PART=1 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doCPURunSmallDTSTLowLocality
# 	mv Bank.csv CPUonly_high_thrs_small_DISABLED_LL_s${s}
# 	doCPURunSmallDTSTHighLocality
# 	mv Bank.csv CPUonly_high_thrs_small_DISABLED_HL_s${s}
# 	doCPURunLargeDTSTLowLocality
# 	mv Bank.csv CPUonly_high_thrs_large_DISABLED_LL_s${s}
# 	doCPURunLargeDTSTHighLocality
# 	mv Bank.csv CPUonly_high_thrs_large_DISABLED_HL_s${s}
# done

make clean ; make INST_CPU=0 GPUEn=0 BANK_PART=2 PROFILE=1 -j 14
for s in `seq $SAMPLES`
do
	doCPURunSmallDTSTLowLocality
	mv Bank.csv CPUonly_high_thrs_small_DISABLED_URLL_s${s}
	doCPURunSmallDTSTHighLocality
	mv Bank.csv CPUonly_high_thrs_large_DISABLED_URLL_s${s}
done

CPU_THREADS=$LOW_CPU_THREADS
# make clean ; make LOG_TYPE=VERS INST_CPU=1 GPUEn=0 BANK_PART=1 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doCPURunSmallDTSTLowLocality
# 	mv Bank.csv CPUonly_low_thrs_small_VERS_LL_s${s}
# 	doCPURunSmallDTSTHighLocality
# 	mv Bank.csv CPUonly_low_thrs_small_VERS_HL_s${s}
# 	doCPURunLargeDTSTLowLocality
# 	mv Bank.csv CPUonly_low_thrs_large_VERS_LL_s${s}
# 	doCPURunLargeDTSTHighLocality
# 	mv Bank.csv CPUonly_low_thrs_large_VERS_HL_s${s}
# done

make clean ; make LOG_TYPE=VERS INST_CPU=1 GPUEn=0 BANK_PART=2 PROFILE=1 -j 14
for s in `seq $SAMPLES`
do
	doCPURunSmallDTSTLowLocality
	mv Bank.csv CPUonly_low_thrs_small_VERS_URLL_s${s}
	doCPURunSmallDTSTHighLocality
	mv Bank.csv CPUonly_low_thrs_large_VERS_URLL_s${s}
done

CPU_THREADS=$HIGH_CPU_THREADS
# make clean ; make LOG_TYPE=VERS INST_CPU=1 GPUEn=0 BANK_PART=1 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doCPURunSmallDTSTLowLocality
# 	mv Bank.csv CPUonly_high_thrs_small_VERS_LL_s${s}
# 	doCPURunSmallDTSTHighLocality
# 	mv Bank.csv CPUonly_high_thrs_small_VERS_HL_s${s}
# 	doCPURunLargeDTSTLowLocality
# 	mv Bank.csv CPUonly_high_thrs_large_VERS_LL_s${s}
# 	doCPURunLargeDTSTHighLocality
# 	mv Bank.csv CPUonly_high_thrs_large_VERS_HL_s${s}
# done

make clean ; make LOG_TYPE=VERS INST_CPU=1 GPUEn=0 BANK_PART=2 PROFILE=1 -j 14
for s in `seq $SAMPLES`
do
	doCPURunSmallDTSTLowLocality
	mv Bank.csv CPUonly_high_thrs_small_VERS_URLL_s${s}
	doCPURunSmallDTSTHighLocality
	mv Bank.csv CPUonly_high_thrs_large_VERS_URLL_s${s}
done

CPU_THREADS=$LOW_CPU_THREADS
# make clean ; make LOG_TYPE=BMAP INST_CPU=1 GPUEn=0 BANK_PART=1 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doCPURunSmallDTSTLowLocality
# 	mv Bank.csv CPUonly_low_thrs_small_BMAP_LL_s${s}
# 	doCPURunSmallDTSTHighLocality
# 	mv Bank.csv CPUonly_low_thrs_small_BMAP_HL_s${s}
# 	doCPURunLargeDTSTLowLocality
# 	mv Bank.csv CPUonly_low_thrs_large_BMAP_LL_s${s}
# 	doCPURunLargeDTSTHighLocality
# 	mv Bank.csv CPUonly_low_thrs_large_BMAP_HL_s${s}
# done

make clean ; make LOG_TYPE=BMAP INST_CPU=1 GPUEn=0 BANK_PART=2 PROFILE=1 -j 14
for s in `seq $SAMPLES`
do
	doCPURunSmallDTSTLowLocality
	mv Bank.csv CPUonly_low_thrs_small_BMAP_URLL_s${s}
	doCPURunSmallDTSTHighLocality
	mv Bank.csv CPUonly_low_thrs_large_BMAP_URLL_s${s}
done

CPU_THREADS=$HIGH_CPU_THREADS
# make clean ; make LOG_TYPE=BMAP INST_CPU=1 GPUEn=0 BANK_PART=1 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doCPURunSmallDTSTLowLocality
# 	mv Bank.csv CPUonly_high_thrs_small_BMAP_LL_s${s}
# 	doCPURunSmallDTSTHighLocality
# 	mv Bank.csv CPUonly_high_thrs_small_BMAP_HL_s${s}
# 	doCPURunLargeDTSTLowLocality
# 	mv Bank.csv CPUonly_high_thrs_large_BMAP_LL_s${s}
# 	doCPURunLargeDTSTHighLocality
# 	mv Bank.csv CPUonly_high_thrs_large_BMAP_HL_s${s}
# done

make clean ; make LOG_TYPE=BMAP INST_CPU=1 GPUEn=0 BANK_PART=2 PROFILE=1 -j 14
for s in `seq $SAMPLES`
do
	doCPURunSmallDTSTLowLocality
	mv Bank.csv CPUonly_high_thrs_small_BMAP_URLL_s${s}
	doCPURunSmallDTSTHighLocality
	mv Bank.csv CPUonly_high_thrs_large_BMAP_URLL_s${s}
done

# #########################
# ### VERS LOW threads
# ##################################################
# CPU_THREADS=$LOW_CPU_THREADS
# ### THE intervals must intersect! set P_INTERSECT=0.0 --> else the math to calculate the addr causes conflicts
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 BANK_PART=1 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTSTLowLocality
# 	mv Bank.csv VERS_compressed_low_THRS_small_LL_s${s}
# 	doRunSmallDTSTHighLocality
# 	mv Bank.csv VERS_compressed_low_THRS_small_HL_s${s}
# 	doRunLargeDTSTLowLocality
# 	mv Bank.csv VERS_compressed_low_THRS_large_LL_s${s}
# 	doRunLargeDTSTHighLocality
# 	mv Bank.csv VERS_compressed_low_THRS_large_HL_s${s}
# done
#
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 BANK_PART=2 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTSTLowLocality
# 	mv Bank.csv VERS_compressed_low_THRS_small_URLL_s${s}
# 	doRunLargeDTSTLowLocality
# 	mv Bank.csv VERS_compressed_low_THRS_large_URLL_s${s}
# done
#
# #############################
# ##########################
# #######################
# make clean ; make CMP_TYPE=EXPLICIT BANK_PART=3 GPU_PART=0.52 CPU_PART=0.50 \
# 	LOG_TYPE=VERS P_INTERSECT=0.0 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTST
# 	mv Bank.csv VERS_explicit_low_THRS_small_s${s}
# 	doRunLargeDTST
# 	mv Bank.csv VERS_explicit_low_THRS_large_s${s}
# done
# #######################
# ##########################
# #############################
# ##################################################
#
# #########################
# ### VERS HIGH threads
# ##################################################
# CPU_THREADS=$HIGH_CPU_THREADS
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 BANK_PART=1 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTSTLowLocality
# 	mv Bank.csv VERS_compressed_high_THRS_small_LL_s${s}
# 	doRunSmallDTSTHighLocality
# 	mv Bank.csv VERS_compressed_high_THRS_small_HL_s${s}
# 	doRunLargeDTSTLowLocality
# 	mv Bank.csv VERS_compressed_high_THRS_large_LL_s${s}
# 	doRunLargeDTSTHighLocality
# 	mv Bank.csv VERS_compressed_high_THRS_large_HL_s${s}
# done
#
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 BANK_PART=2 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTSTLowLocality
# 	mv Bank.csv VERS_compressed_high_THRS_small_URLL_s${s}
# 	doRunLargeDTSTLowLocality
# 	mv Bank.csv VERS_compressed_high_THRS_large_URLL_s${s}
# done
#
# #############################
# ##########################
# #######################
# make clean ; make CMP_TYPE=EXPLICIT BANK_PART=3 GPU_PART=0.52 CPU_PART=0.50 \
# 	LOG_TYPE=VERS P_INTERSECT=0.0 USE_TSX_IMPL=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTST
# 	mv Bank.csv VERS_explicit_high_THRS_small_s${s}
# 	doRunLargeDTST
# 	mv Bank.csv VERS_explicit_high_THRS_large_s${s}
# done
# #######################
# ##########################
# #############################
# ##################################################

#########################

# #########################
# ### BMAP LOW threads
# ##################################################
# CPU_THREADS=$LOW_CPU_THREADS
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP BANK_PART=1 USE_TSX_IMPL=1 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTSTLowLocality
# 	mv Bank.csv BMAP_compressed_low_THRS_small_LL_s${s}
# 	doRunSmallDTSTHighLocality
# 	mv Bank.csv BMAP_compressed_low_THRS_small_HL_s${s}
# 	doRunLargeDTSTLowLocality
# 	mv Bank.csv BMAP_compressed_low_THRS_large_LL_s${s}
# 	doRunLargeDTSTHighLocality
# 	mv Bank.csv BMAP_compressed_low_THRS_large_HL_s${s}
# done
#
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP BANK_PART=2 USE_TSX_IMPL=1 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTSTLowLocality
# 	mv Bank.csv BMAP_compressed_low_THRS_small_URLL_s${s}
# 	doRunLargeDTSTLowLocality
# 	mv Bank.csv BMAP_compressed_low_THRS_large_URLL_s${s}
# done

# #############################
# ##########################
# #######################
# make clean ; make CMP_TYPE=EXPLICIT BANK_PART=3 GPU_PART=0.52 CPU_PART=0.50 \
# 	LOG_TYPE=BMAP P_INTERSECT=0.0 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTST
# 	mv Bank.csv BMAP_explicit_low_THRS_small_s${s}
# 	doRunLargeDTST
# 	mv Bank.csv BMAP_explicit_low_THRS_large_s${s}
# done
# #######################
# ##########################
# #############################
# ##################################################
#
# #########################
# ### BMAP HIGH threads
# ##################################################
# CPU_THREADS=$HIGH_CPU_THREADS
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP BANK_PART=1 USE_TSX_IMPL=1 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTSTLowLocality
# 	mv Bank.csv BMAP_compressed_high_THRS_small_LL_s${s}
# 	doRunSmallDTSTHighLocality
# 	mv Bank.csv BMAP_compressed_high_THRS_small_HL_s${s}
# 	doRunLargeDTSTLowLocality
# 	mv Bank.csv BMAP_compressed_high_THRS_large_LL_s${s}
# 	doRunLargeDTSTHighLocality
# 	mv Bank.csv BMAP_compressed_high_THRS_large_HL_s${s}
# done
#
# make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP BANK_PART=2 USE_TSX_IMPL=1 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTSTLowLocality
# 	mv Bank.csv BMAP_compressed_high_THRS_small_URLL_s${s}
# 	doRunLargeDTSTLowLocality
# 	mv Bank.csv BMAP_compressed_high_THRS_large_URLL_s${s}
# done

# #############################
# ##########################
# #######################
# make clean ; make CMP_TYPE=EXPLICIT BANK_PART=3 GPU_PART=0.52 CPU_PART=0.50 \
# 	LOG_TYPE=BMAP P_INTERSECT=0.0 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTST
# 	mv Bank.csv BMAP_explicit_high_THRS_small_s${s}
# 	doRunLargeDTST
# 	mv Bank.csv BMAP_explicit_high_THRS_large_s${s}
# done
# #######################
# ##########################
# #############################
# ##################################################
