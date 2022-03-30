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
SAMPLES=1
#./makeTM.sh

CPU_THREADS=4
LOW_CPU_THREADS=1
HIGH_CPU_THREADS=14

rm -f Bank.csv

LARGE_DATASET=50000000 #2621440 # 90 000 000 is the max for my home machine
LARGE_DATASET_P20=61000000 #2621440 # 90 000 000 is the max for my home machine
SMALL_DATASET=100000 #2621440 # 90 000 000 is the max for my home machine
SMALL_DATASET_P20=122000 #2621440 # 90 000 000 is the max for my home machine

### TODO: find the best combination of parameters
function doRunSmallDTST {
	timeout 10s ./bank -n $CPU_THREADS -b 32  -x 128 -a  $SMALL_DATASET -d $DURATION -T 1
	# timeout 10s ./bank -n $CPU_THREADS -b 32  -x 128 -a  $SMALL_DATASET -d $DURATION -T 2
	timeout 10s ./bank -n $CPU_THREADS -b 64  -x 128 -a  $SMALL_DATASET -d $DURATION -T 2
	# timeout 10s ./bank -n $CPU_THREADS -b 64  -x 128 -a  $SMALL_DATASET -d $DURATION -T 4
	timeout 10s ./bank -n $CPU_THREADS -b 128 -x 128 -a  $SMALL_DATASET -d $DURATION -T 4
	# timeout 10s ./bank -n $CPU_THREADS -b 256 -x 128 -a  $SMALL_DATASET -d $DURATION -T 4
	timeout 10s ./bank -n $CPU_THREADS -b 512 -x 128 -a  $SMALL_DATASET -d $DURATION -T 4
	# timeout 10s ./bank -n $CPU_THREADS -b 512 -x 256 -a  $SMALL_DATASET -d $DURATION -T 4
	timeout 10s ./bank -n $CPU_THREADS -b 512 -x 512 -a  $SMALL_DATASET -d $DURATION -T 4
	timeout 10s ./bank -n $CPU_THREADS -b 512 -x 512 -a  $SMALL_DATASET -d $DURATION -T 8
	timeout 10s ./bank -n $CPU_THREADS -b 640 -x 512 -a  $SMALL_DATASET -d $DURATION -T 8
	timeout 10s ./bank -n $CPU_THREADS -b 768 -x 512 -a  $SMALL_DATASET -d $DURATION -T 8
	timeout 10s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $SMALL_DATASET -d $DURATION -T 8
}

function doRunLargeDTST {
	timeout 10s ./bank -n $CPU_THREADS -b 32  -x 128 -a  $LARGE_DATASET -d $DURATION -T 1
	# timeout 10s ./bank -n $CPU_THREADS -b 32  -x 128 -a  $LARGE_DATASET -d $DURATION -T 2
	timeout 10s ./bank -n $CPU_THREADS -b 64  -x 128 -a  $LARGE_DATASET -d $DURATION -T 2
	# timeout 10s ./bank -n $CPU_THREADS -b 128 -x 128 -a  $LARGE_DATASET -d $DURATION -T 2
	timeout 10s ./bank -n $CPU_THREADS -b 128 -x 128 -a  $LARGE_DATASET -d $DURATION -T 4
	# timeout 10s ./bank -n $CPU_THREADS -b 256 -x 128 -a  $LARGE_DATASET -d $DURATION -T 4
	timeout 10s ./bank -n $CPU_THREADS -b 256 -x 256 -a  $LARGE_DATASET -d $DURATION -T 4
	# timeout 10s ./bank -n $CPU_THREADS -b 256 -x 256 -a  $LARGE_DATASET -d $DURATION -T 8
	timeout 10s ./bank -n $CPU_THREADS -b 512 -x 256 -a  $LARGE_DATASET -d $DURATION -T 8
	timeout 10s ./bank -n $CPU_THREADS -b 512 -x 512 -a  $LARGE_DATASET -d $DURATION -T 8
	timeout 10s ./bank -n $CPU_THREADS -b 640 -x 512 -a  $LARGE_DATASET -d $DURATION -T 8
	timeout 10s ./bank -n $CPU_THREADS -b 768 -x 512 -a  $LARGE_DATASET -d $DURATION -T 8
	timeout 10s ./bank -n $CPU_THREADS -b 1024 -x 512 -a $LARGE_DATASET -d $DURATION -T 8
}

# LARGE_DATASET=$LARGE_DATASET_P20
# SMALL_DATASET=$SMALL_DATASET_P20

make clean ; make CMP_TYPE=COMPRESSED BANK_PART=3 GPU_PART=0.52 CPU_PART=0.50 \
	P_INTERSECT=0.0 CPUEn=0 -j 14
for s in `seq $SAMPLES`
do
	doRunSmallDTST
	mv Bank.csv GPUonly_small_COMPRESSED_s${s}
	doRunLargeDTST
	mv Bank.csv GPUonly_large_COMPRESSED_s${s}
done

make clean ; make CMP_TYPE=EXPLICIT BANK_PART=3 GPU_PART=0.52 CPU_PART=0.50 \
	P_INTERSECT=0.0 CPUEn=0 -j 14
for s in `seq $SAMPLES`
do
	doRunSmallDTST
	mv Bank.csv GPUonly_small_EXPLICIT_s${s}
	doRunLargeDTST
	mv Bank.csv GPUonly_large_EXPLICIT_s${s}
done

make clean ; make CMP_TYPE=DISABLED BANK_PART=3 GPU_PART=0.52 CPU_PART=0.50 \
	P_INTERSECT=0.0 CPUEn=0 -j 14
for s in `seq $SAMPLES`
do
	doRunSmallDTST
	mv Bank.csv GPUonly_small_DISABLED_s${s}
	doRunLargeDTST
	mv Bank.csv GPUonly_large_DISABLED_s${s}
done

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


# #########################
# ### VERS LOW threads
# ##################################################
CPU_THREADS=$LOW_CPU_THREADS
### THE intervals must intersect! set P_INTERSECT=0.0 --> else the math to calculate the addr causes conflicts
make clean ; make CMP_TYPE=COMPRESSED BANK_PART=3 GPU_PART=0.52 CPU_PART=0.50 \
	LOG_TYPE=VERS P_INTERSECT=0.0 -j 14
for s in `seq $SAMPLES`
do
	doRunSmallDTST
	mv Bank.csv VERS_compressed_low_THRS_small_s${s}
	doRunLargeDTST
	mv Bank.csv VERS_compressed_low_THRS_large_s${s}
done

make clean ; make CMP_TYPE=EXPLICIT BANK_PART=3 GPU_PART=0.52 CPU_PART=0.50 \
	LOG_TYPE=VERS P_INTERSECT=0.0 -j 14
for s in `seq $SAMPLES`
do
	doRunSmallDTST
	mv Bank.csv VERS_explicit_low_THRS_small_s${s}
	doRunLargeDTST
	mv Bank.csv VERS_explicit_low_THRS_large_s${s}
done
# ##################################################
#
# #########################
# ### VERS HIGH threads
# ##################################################
CPU_THREADS=$HIGH_CPU_THREADS
make clean ; make CMP_TYPE=COMPRESSED BANK_PART=3 GPU_PART=0.52 CPU_PART=0.50 \
	LOG_TYPE=VERS P_INTERSECT=0.0 USE_TSX_IMPL=1 -j 14
for s in `seq $SAMPLES`
do
	doRunSmallDTST
	mv Bank.csv VERS_compressed_high_THRS_small_s${s}
	doRunLargeDTST
	mv Bank.csv VERS_compressed_high_THRS_large_s${s}
done

make clean ; make CMP_TYPE=EXPLICIT BANK_PART=3 GPU_PART=0.52 CPU_PART=0.50 \
	LOG_TYPE=VERS P_INTERSECT=0.0 USE_TSX_IMPL=1 -j 14
for s in `seq $SAMPLES`
do
	doRunSmallDTST
	mv Bank.csv VERS_explicit_high_THRS_small_s${s}
	doRunLargeDTST
	mv Bank.csv VERS_explicit_high_THRS_large_s${s}
done
# ##################################################

#########################

# #########################
# ### BMAP LOW threads
# ##################################################
CPU_THREADS=$LOW_CPU_THREADS
make clean ; make CMP_TYPE=COMPRESSED BANK_PART=3 GPU_PART=0.52 CPU_PART=0.50 \
	LOG_TYPE=BMAP P_INTERSECT=0.0 -j 14
for s in `seq $SAMPLES`
do
	doRunSmallDTST
	mv Bank.csv BMAP_compressed_low_THRS_small_s${s}
	doRunLargeDTST
	mv Bank.csv BMAP_compressed_low_THRS_large_s${s}
done

make clean ; make CMP_TYPE=EXPLICIT BANK_PART=3 GPU_PART=0.52 CPU_PART=0.50 \
	LOG_TYPE=BMAP P_INTERSECT=0.0 -j 14
for s in `seq $SAMPLES`
do
	doRunSmallDTST
	mv Bank.csv BMAP_explicit_low_THRS_small_s${s}
	doRunLargeDTST
	mv Bank.csv BMAP_explicit_low_THRS_large_s${s}
done
# ##################################################
#
# #########################
# ### BMAP HIGH threads
# ##################################################
CPU_THREADS=$HIGH_CPU_THREADS
make clean ; make CMP_TYPE=COMPRESSED BANK_PART=3 GPU_PART=0.52 CPU_PART=0.50 \
	LOG_TYPE=BMAP P_INTERSECT=0.0 -j 14
for s in `seq $SAMPLES`
do
	doRunSmallDTST
	mv Bank.csv BMAP_compressed_high_THRS_small_s${s}
	doRunLargeDTST
	mv Bank.csv BMAP_compressed_high_THRS_large_s${s}
done

make clean ; make CMP_TYPE=EXPLICIT BANK_PART=3 GPU_PART=0.52 CPU_PART=0.50 \
	LOG_TYPE=BMAP P_INTERSECT=0.0 -j 14
for s in `seq $SAMPLES`
do
	doRunSmallDTST
	mv Bank.csv BMAP_explicit_high_THRS_small_s${s}
	doRunLargeDTST
	mv Bank.csv BMAP_explicit_high_THRS_large_s${s}
done
# ##################################################
