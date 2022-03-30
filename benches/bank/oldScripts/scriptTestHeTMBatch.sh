#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1
filename_tsx="Bank_TSX"
filename_tiny="Bank_Tiny"

GPU_PART="1.0"
CPU_PART="0.0"
P_INTERSECT="0.0"
DURATION=6000
BLOCKS="2 4 8 16 32 64 256 512 1024" # 512
THREADS="512" #"2 4 8 16 32 64 96 256 320 512 640 768 1024"
BATCH_SIZE="4"
SAMPLES=5
#./makeTM.sh

CPU_THREADS=4
# LOW_CPU_THREADS=2
LARGE_HIGH_CPU_THREADS=8
LARGE_VERY_HIGH_CPU_THREADS=28
SMALL_HIGH_CPU_THREADS=14
SMALL_VERY_HIGH_CPU_THREADS=50

rm -f Bank.csv

LARGE_DATASET=10000000 #2621440 # 90 000 000 is the max for my home machine
LARGE_DATASET_P21=12100000 #2621440 # 90 000 000 is the max for my home machine
LARGE_DATASET_P300=40000000 #2621440 # 90 000 000 is the max for my home machine
SMALL_DATASET=1000000 #2621440 # 90 000 000 is the max for my home machine
SMALL_DATASET_P21=1210000 #2621440 # 90 000 000 is the max for my home machine
SMALL_DATASET_P300=4000000 #2621440 # 90 000 000 is the max for my home machine

### TODO: find the best combination of parameters
function doRunSmallDTST {
	timeout 12s ./bank -n $CPU_THREADS -b 32  -x 128 -a  $SMALL_DATASET -d $DURATION -T 1
	timeout 12s ./bank -n $CPU_THREADS -b 32  -x 128 -a  $SMALL_DATASET -d $DURATION -T 2
	timeout 12s ./bank -n $CPU_THREADS -b 64  -x 128 -a  $SMALL_DATASET -d $DURATION -T 2
	timeout 12s ./bank -n $CPU_THREADS -b 64  -x 128 -a  $SMALL_DATASET -d $DURATION -T 4
	timeout 12s ./bank -n $CPU_THREADS -b 128 -x 128 -a  $SMALL_DATASET -d $DURATION -T 4
	timeout 12s ./bank -n $CPU_THREADS -b 256 -x 128 -a  $SMALL_DATASET -d $DURATION -T 4
	timeout 12s ./bank -n $CPU_THREADS -b 512 -x 128 -a  $SMALL_DATASET -d $DURATION -T 4
	timeout 12s ./bank -n $CPU_THREADS -b 512 -x 256 -a  $SMALL_DATASET -d $DURATION -T 4
	timeout 12s ./bank -n $CPU_THREADS -b 512 -x 512 -a  $SMALL_DATASET -d $DURATION -T 4
	timeout 12s ./bank -n $CPU_THREADS -b 512 -x 512 -a  $SMALL_DATASET -d $DURATION -T 8
	timeout 12s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $SMALL_DATASET -d $DURATION -T 8
}

function doRunLargeDTST {
	timeout 12s ./bank -n $CPU_THREADS -b 32  -x 128 -a  $LARGE_DATASET -d $DURATION -T 1
	timeout 12s ./bank -n $CPU_THREADS -b 32  -x 128 -a  $LARGE_DATASET -d $DURATION -T 2
	timeout 12s ./bank -n $CPU_THREADS -b 64  -x 128 -a  $LARGE_DATASET -d $DURATION -T 2
	timeout 12s ./bank -n $CPU_THREADS -b 128 -x 128 -a  $LARGE_DATASET -d $DURATION -T 2
	timeout 12s ./bank -n $CPU_THREADS -b 128 -x 128 -a  $LARGE_DATASET -d $DURATION -T 4
	timeout 12s ./bank -n $CPU_THREADS -b 256 -x 128 -a  $LARGE_DATASET -d $DURATION -T 4
	timeout 12s ./bank -n $CPU_THREADS -b 256 -x 256 -a  $LARGE_DATASET -d $DURATION -T 4
	timeout 12s ./bank -n $CPU_THREADS -b 256 -x 256 -a  $LARGE_DATASET -d $DURATION -T 8
	timeout 12s ./bank -n $CPU_THREADS -b 512 -x 256 -a  $LARGE_DATASET -d $DURATION -T 8
	timeout 12s ./bank -n $CPU_THREADS -b 512 -x 512 -a  $LARGE_DATASET -d $DURATION -T 8
	timeout 12s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -T 8
}

make clean ; make CMP_TYPE=DISABLED BANK_PART=1 GPU_PART=1.0 \
	CPU_PART=0.0 P_INTERSECT=$P_INTERSECT CPUEn=0 PROFILE=1 -j 14
for s in `seq $SAMPLES`
do
	doRunSmallDTST
	mv Bank.csv GPUonly_small_s${s}
	doRunLargeDTST
	mv Bank.csv GPUonly_large_s${s}
done

make clean ; make CMP_TYPE=DISABLED BANK_PART=1 GPU_PART=0.0 \
	CPU_PART=1.0 USE_TSX_IMPL=1 P_INTERSECT=$P_INTERSECT INST_CPU=0 GPUEn=0 PROFILE=1 -j 14
for s in `seq $SAMPLES`
do
	# CPU_THREADS=$LOW_CPU_THREADS
	# doRunSmallDTST
	# mv Bank.csv CPUonly_low_THRS_small_s${s}
	CPU_THREADS=$SMALL_HIGH_CPU_THREADS
	doRunSmallDTST
	mv Bank.csv CPUonly_high_THRS_small_s${s}
	CPU_THREADS=$SMALL_VERY_HIGH_CPU_THREADS
	doRunSmallDTST
	mv Bank.csv CPUonly_very_high_THRS_small_s${s}

	# CPU_THREADS=$LOW_CPU_THREADS
	# doRunLargeDTST
	# mv Bank.csv CPUonly_low_THRS_large_s${s}
	CPU_THREADS=$LARGE_HIGH_CPU_THREADS
	doRunLargeDTST
	mv Bank.csv CPUonly_high_THRS_large_s${s}
	CPU_THREADS=$LARGE_VERY_HIGH_CPU_THREADS
	doRunLargeDTST
	mv Bank.csv CPUonly_very_high_THRS_large_s${s}
done
#
# #########################
# ### VERS LOW threads
# ##################################################
# CPU_THREADS=$LOW_CPU_THREADS
# ### THE intervals must intersect! set P_INTERSECT=0.0 --> else the math to calculate the addr causes conflicts
# make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 GPU_PART=0.5 CPU_PART=0.51 \
# 	LOG_TYPE=VERS USE_TSX_IMPL=1 P_INTERSECT=0.0 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTST
# 	mv Bank.csv VERS_low_THRS_small_s${s}
# 	doRunLargeDTST
# 	mv Bank.csv VERS_low_THRS_large_s${s}
# done
# ##################################################
#
#########################
### VERS HIGH threads
##################################################
make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 GPU_PART=0.5 CPU_PART=0.51 \
	LOG_TYPE=VERS P_INTERSECT=0.0 USE_TSX_IMPL=1 PROFILE=1 -j 14
for s in `seq $SAMPLES`
do
	CPU_THREADS=$SMALL_HIGH_CPU_THREADS
	doRunSmallDTST
	mv Bank.csv VERS_high_THRS_small_s${s}
	CPU_THREADS=$LARGE_HIGH_CPU_THREADS
	doRunLargeDTST
	mv Bank.csv VERS_high_THRS_large_s${s}
done
##################################################

#########################
### VERS VERY HIGH threads
##################################################
make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 GPU_PART=0.5 CPU_PART=0.51 \
	LOG_TYPE=VERS P_INTERSECT=0.0 USE_TSX_IMPL=1 PROFILE=1 -j 14
for s in `seq $SAMPLES`
do
	CPU_THREADS=$SMALL_VERY_HIGH_CPU_THREADS
	doRunSmallDTST
	mv Bank.csv VERS_very_high_THRS_small_s${s}
	CPU_THREADS=$LARGE_VERY_HIGH_CPU_THREADS
	doRunLargeDTST
	mv Bank.csv VERS_very_high_THRS_large_s${s}
done
##################################################

#########################
### (Overlap) VERS HIGH threads
##################################################
make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 GPU_PART=0.5 CPU_PART=0.51 \
	LOG_TYPE=VERS P_INTERSECT=0.0 USE_TSX_IMPL=1 OVERLAP_CPY_BACK=1 PROFILE=1 -j 14
for s in `seq $SAMPLES`
do
	CPU_THREADS=$SMALL_HIGH_CPU_THREADS
	doRunSmallDTST
	mv Bank.csv VERS_overlap_high_THRS_small_s${s}
	CPU_THREADS=$LARGE_HIGH_CPU_THREADS
	doRunLargeDTST
	mv Bank.csv VERS_overlap_high_THRS_large_s${s}
done
##################################################

#########################
### (Overlap) VERS VERY HIGH threads
##################################################
make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 GPU_PART=0.5 CPU_PART=0.51 \
	LOG_TYPE=VERS P_INTERSECT=0.0 USE_TSX_IMPL=1 OVERLAP_CPY_BACK=1 PROFILE=1 -j 14
for s in `seq $SAMPLES`
do
	CPU_THREADS=$SMALL_VERY_HIGH_CPU_THREADS
	doRunSmallDTST
	mv Bank.csv VERS_overlap_very_high_THRS_small_s${s}
	CPU_THREADS=$LARGE_VERY_HIGH_CPU_THREADS
	doRunLargeDTST
	mv Bank.csv VERS_overlap_very_high_THRS_large_s${s}
done
##################################################

# ##################################################
# ####################################################################################################
# ##################################################
#
#
# #########################
# ### ADDR LOW threads
# ##################################################
# CPU_THREADS=$LOW_CPU_THREADS
# ### THE intervals must intersect! set P_INTERSECT=0.0 --> else the math to calculate the addr causes conflicts
# make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 GPU_PART=0.5 CPU_PART=0.51 \
# 	LOG_TYPE=ADDR P_INTERSECT=0.0 USE_TSX_IMPL=1 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTST
# 	mv Bank.csv ADDR_low_THRS_small_s${s}
# 	doRunLargeDTST
# 	mv Bank.csv ADDR_low_THRS_large_s${s}
# done
# ##################################################

#########################
### ADDR HIGH threads
##################################################
make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 GPU_PART=0.5 CPU_PART=0.51 \
	LOG_TYPE=ADDR P_INTERSECT=0.0 USE_TSX_IMPL=1 PROFILE=1 -j 14
for s in `seq $SAMPLES`
do
	CPU_THREADS=$SMALL_HIGH_CPU_THREADS
	doRunSmallDTST
	mv Bank.csv ADDR_high_THRS_small_s${s}

	CPU_THREADS=$LARGE_HIGH_CPU_THREADS
	doRunLargeDTST
	mv Bank.csv ADDR_high_THRS_large_s${s}
done
##################################################

#########################
### ADDR VERY HIGH threads
##################################################
make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 GPU_PART=0.5 CPU_PART=0.51 \
	LOG_TYPE=ADDR P_INTERSECT=0.0 USE_TSX_IMPL=1 PROFILE=1 -j 14
for s in `seq $SAMPLES`
do
	CPU_THREADS=$SMALL_VERY_HIGH_CPU_THREADS
	doRunSmallDTST
	mv Bank.csv ADDR_very_high_THRS_small_s${s}

	CPU_THREADS=$LARGE_VERY_HIGH_CPU_THREADS
	doRunLargeDTST
	mv Bank.csv ADDR_very_high_THRS_large_s${s}
done
##################################################

#########################
### (OVERLAP) ADDR HIGH threads
##################################################
make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 GPU_PART=0.5 CPU_PART=0.51 \
	LOG_TYPE=ADDR P_INTERSECT=0.0 USE_TSX_IMPL=1 OVERLAP_CPY_BACK=1 PROFILE=1 -j 14
for s in `seq $SAMPLES`
do
	CPU_THREADS=$SMALL_HIGH_CPU_THREADS
	doRunSmallDTST
	mv Bank.csv ADDR_overlap_high_THRS_small_s${s}

	CPU_THREADS=$LARGE_HIGH_CPU_THREADS
	doRunLargeDTST
	mv Bank.csv ADDR_overlap_high_THRS_large_s${s}
done
##################################################

#########################
### (OVERLAP) ADDR VERY HIGH threads
##################################################
make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 GPU_PART=0.5 CPU_PART=0.51 \
	LOG_TYPE=ADDR P_INTERSECT=0.0 USE_TSX_IMPL=1 OVERLAP_CPY_BACK=1 PROFILE=1 -j 14
for s in `seq $SAMPLES`
do
	CPU_THREADS=$SMALL_VERY_HIGH_CPU_THREADS
	doRunSmallDTST
	mv Bank.csv ADDR_overlap_very_high_THRS_small_s${s}

	CPU_THREADS=$LARGE_VERY_HIGH_CPU_THREADS
	doRunLargeDTST
	mv Bank.csv ADDR_overlap_very_high_THRS_large_s${s}
done
##################################################

# ##################################################
# ####################################################################################################
# ##################################################
#
# #########################
#
# #########################
# ### BMAP LOW threads
# ##################################################
# CPU_THREADS=$LOW_CPU_THREADS
# make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 GPU_PART=0.5 CPU_PART=0.51 \
# 	LOG_TYPE=BMAP P_INTERSECT=0.0 USE_TSX_IMPL=1 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTST
# 	mv Bank.csv BMAP_low_THRS_small_s${s}
# 	doRunLargeDTST
# 	mv Bank.csv BMAP_low_THRS_large_s${s}
# done
# ##################################################

#########################
### BMAP HIGH threads
##################################################
make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 GPU_PART=0.5 CPU_PART=0.51 \
	LOG_TYPE=BMAP P_INTERSECT=0.0 USE_TSX_IMPL=1 PROFILE=1 -j 14
for s in `seq $SAMPLES`
do
	CPU_THREADS=$SMALL_HIGH_CPU_THREADS
	doRunSmallDTST
	mv Bank.csv BMAP_high_THRS_small_s${s}

	CPU_THREADS=$LARGE_HIGH_CPU_THREADS
	doRunLargeDTST
	mv Bank.csv BMAP_high_THRS_large_s${s}
done
##################################################

#########################
### BMAP VERY HIGH threads
##################################################
make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 GPU_PART=0.5 CPU_PART=0.51 \
	LOG_TYPE=BMAP P_INTERSECT=0.0 USE_TSX_IMPL=1 PROFILE=1 -j 14
for s in `seq $SAMPLES`
do
	CPU_THREADS=$SMALL_VERY_HIGH_CPU_THREADS
	doRunSmallDTST
	mv Bank.csv BMAP_very_high_THRS_small_s${s}

	CPU_THREADS=$LARGE_VERY_HIGH_CPU_THREADS
	doRunLargeDTST
	mv Bank.csv BMAP_very_high_THRS_large_s${s}
done
##################################################

#########################
### (OVERLAP) BMAP HIGH threads
##################################################
make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 GPU_PART=0.5 CPU_PART=0.51 \
	LOG_TYPE=BMAP P_INTERSECT=0.0 USE_TSX_IMPL=1 OVERLAP_CPY_BACK=1 PROFILE=1 -j 14
for s in `seq $SAMPLES`
do
	CPU_THREADS=$SMALL_HIGH_CPU_THREADS
	doRunSmallDTST
	mv Bank.csv BMAP_overlap_high_THRS_small_s${s}

	CPU_THREADS=$LARGE_HIGH_CPU_THREADS
	doRunLargeDTST
	mv Bank.csv BMAP_overlap_high_THRS_large_s${s}
done
##################################################

#########################
### (OVERLAP) BMAP VERY HIGH threads
##################################################
make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 GPU_PART=0.5 CPU_PART=0.51 \
	LOG_TYPE=BMAP P_INTERSECT=0.0 USE_TSX_IMPL=1 OVERLAP_CPY_BACK=1 PROFILE=1 -j 14
for s in `seq $SAMPLES`
do
	CPU_THREADS=$SMALL_VERY_HIGH_CPU_THREADS
	doRunSmallDTST
	mv Bank.csv BMAP_overlap_very_high_THRS_small_s${s}

	CPU_THREADS=$LARGE_VERY_HIGH_CPU_THREADS
	doRunLargeDTST
	mv Bank.csv BMAP_overlap_very_high_THRS_large_s${s}
done
##################################################

###############################################################################
###############################################################################
###############################################################################
### with conflicts

# ### BMAP Low inter-contention
#
# CPU_THREADS=$LOW_CPU_THREADS
# make clean ; make CMP_TYPE=COMPRESSED BANK_PART=2 GPU_PART=0.55 CPU_PART=0.55 \
# 	LOG_TYPE=BMAP P_INTERSECT=0.1 USE_TSX_IMPL=1 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTST
# 	mv Bank.csv BMAP_LC_low_THRS_small_s${s}
# 	doRunLargeDTST
# 	mv Bank.csv BMAP_LC_low_THRS_large_s${s}
# done
#
# CPU_THREADS=$HIGH_CPU_THREADS
# make clean ; make CMP_TYPE=COMPRESSED BANK_PART=2 GPU_PART=0.55 CPU_PART=0.55 \
# 	LOG_TYPE=BMAP P_INTERSECT=0.1 USE_TSX_IMPL=1 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTST
# 	mv Bank.csv BMAP_LC_high_THRS_small_s${s}
# 	doRunLargeDTST
# 	mv Bank.csv BMAP_LC_high_THRS_large_s${s}
# done
#
# CPU_THREADS=$VERY_HIGH_CPU_THREADS
# make clean ; make CMP_TYPE=COMPRESSED BANK_PART=2 GPU_PART=0.55 CPU_PART=0.55 \
# 	LOG_TYPE=BMAP P_INTERSECT=0.1 USE_TSX_IMPL=1 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTST
# 	mv Bank.csv BMAP_LC_very_high_THRS_small_s${s}
# 	doRunLargeDTST
# 	mv Bank.csv BMAP_LC_very_high_THRS_large_s${s}
# done
#
# ### BMAP Medium inter-contention
#
# CPU_THREADS=$LOW_CPU_THREADS
# make clean ; make CMP_TYPE=COMPRESSED BANK_PART=2 GPU_PART=0.55 CPU_PART=0.55 \
# 	LOG_TYPE=BMAP P_INTERSECT=0.3 USE_TSX_IMPL=1 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTST
# 	mv Bank.csv BMAP_MC_low_THRS_small_s${s}
# 	doRunLargeDTST
# 	mv Bank.csv BMAP_MC_low_THRS_large_s${s}
# done
#
# CPU_THREADS=$HIGH_CPU_THREADS
# make clean ; make CMP_TYPE=COMPRESSED BANK_PART=2 GPU_PART=0.55 CPU_PART=0.55 \
# 	LOG_TYPE=BMAP P_INTERSECT=0.3 USE_TSX_IMPL=1 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTST
# 	mv Bank.csv BMAP_MC_high_THRS_small_s${s}
# 	doRunLargeDTST
# 	mv Bank.csv BMAP_MC_high_THRS_large_s${s}
# done
#
# CPU_THREADS=$VERY_HIGH_CPU_THREADS
# make clean ; make CMP_TYPE=COMPRESSED BANK_PART=2 GPU_PART=0.55 CPU_PART=0.55 \
# 	LOG_TYPE=BMAP P_INTERSECT=0.3 USE_TSX_IMPL=1 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTST
# 	mv Bank.csv BMAP_MC_very_high_THRS_small_s${s}
# 	doRunLargeDTST
# 	mv Bank.csv BMAP_MC_very_high_THRS_large_s${s}
# done
#
# ### BMAP High inter-contention
#
# CPU_THREADS=$LOW_CPU_THREADS
# make clean ; make CMP_TYPE=COMPRESSED BANK_PART=2 GPU_PART=0.55 CPU_PART=0.55 \
# 	LOG_TYPE=BMAP P_INTERSECT=0.9 USE_TSX_IMPL=1 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTST
# 	mv Bank.csv BMAP_HC_low_THRS_small_s${s}
# 	doRunLargeDTST
# 	mv Bank.csv BMAP_HC_low_THRS_large_s${s}
# done
#
# CPU_THREADS=$HIGH_CPU_THREADS
# make clean ; make CMP_TYPE=COMPRESSED BANK_PART=2 GPU_PART=0.55 CPU_PART=0.55 \
# 	LOG_TYPE=BMAP P_INTERSECT=0.9 USE_TSX_IMPL=1 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTST
# 	mv Bank.csv BMAP_HC_high_THRS_small_s${s}
# 	doRunLargeDTST
# 	mv Bank.csv BMAP_HC_high_THRS_large_s${s}
# done
#
# CPU_THREADS=$VERY_HIGH_CPU_THREADS
# make clean ; make CMP_TYPE=COMPRESSED BANK_PART=2 GPU_PART=0.55 CPU_PART=0.55 \
# 	LOG_TYPE=BMAP P_INTERSECT=0.9 USE_TSX_IMPL=1 PROFILE=1 -j 14
# for s in `seq $SAMPLES`
# do
# 	doRunSmallDTST
# 	mv Bank.csv BMAP_HC_very_high_THRS_small_s${s}
# 	doRunLargeDTST
# 	mv Bank.csv BMAP_HC_very_high_THRS_large_s${s}
# done
