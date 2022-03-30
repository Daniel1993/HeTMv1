#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1
filename_tsx="Bank_TSX"
filename_tiny="Bank_Tiny"

GPU_PART="1.0"
CPU_PART="0.0"
P_INTERSECT="0.0"
DURATION=30000
BLOCKS="2 4 8 16 32 64 256 512 1024" # 512
THREADS="512" #"2 4 8 16 32 64 96 256 320 512 640 768 1024"
BATCH_SIZE="4"
SAMPLES=3
#./makeTM.sh

CPU_THREADS=4
# LOW_CPU_THREADS=2
LARGE_HIGH_CPU_THREADS=4
LARGE_VERY_HIGH_CPU_THREADS=14
SMALL_HIGH_CPU_THREADS=4
SMALL_VERY_HIGH_CPU_THREADS=14

rm -f Bank.csv GPUload.csv CPUload.csv GPUsteal.csv

# LARGE_DATASET=46656
# SMALL_DATASET=7776
# VSMALL_DATASET=1296
#LARGE_DATASET=279936
# LARGE_DATASET=1679616
#VSMALL_DATASET=46656
# VSMALL_DATASET=7776
#VSMALL_DATASET=7776

LARGE_DATASET=1000000
VSMALL_DATASET=100000

#GPU_BLOCKS=512
GPU_BLOCKS=320

GPU_THREADS=256

CPU_THREADS=14

# sleep 20m

SIZE_ZIPF=2000000
GPU_INPUT="GPU_input_${SIZE_ZIPF}_099_25165824.txt"
CPU_INPUT="CPU_input_${SIZE_ZIPF}_099_1310720.txt"

if [ ! -f $GPU_INPUT ]
then
	cd java_zipf
	java Main $SIZE_ZIPF 0.99 25165824 > ../$GPU_INPUT
	cd ..
fi
if [ ! -f $CPU_INPUT ]
then
	cd java_zipf
	java Main $SIZE_ZIPF 0.99 1310720 > ../$CPU_INPUT
	cd ..
fi
#
# SIZE_ZIPF=80000000
# GPU_INPUT="GPU_input_${SIZE_ZIPF}_099_25165824.txt"
# CPU_INPUT="CPU_input_${SIZE_ZIPF}_099_1310720.txt"
#
# if [ ! -f $GPU_INPUT ]
# then
# 	cd java_zipf
# 	java Main $SIZE_ZIPF 0.99 25165824 > ../$GPU_INPUT
# 	cd ..
# fi
# if [ ! -f $CPU_INPUT ]
# then
# 	cd java_zipf
# 	java Main $SIZE_ZIPF 0.99 1310720 > ../$CPU_INPUT
# 	cd ..
# fi

# memcached read only

### TODO: use shared 0, 30, 50

function actualRun {
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		-d $DURATION -N 0.1 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f ${1} CPU_BACKOFF=300 \
		GPU_STEAL_PROB=${2} CPU_STEAL_PROB=${3} NB_CONFL_GPU_BUFFER=$NB_CONFL_GPU_BUFFER \
		NB_CONFL_CPU_BUFFER=$NB_CONFL_CPU_BUFFER CONFL_SPACE=$CONFL_SPACE
	if [ $? -ne 0 ]
	then
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		-d $DURATION -N 0.1 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f ${1} CPU_BACKOFF=300 \
		GPU_STEAL_PROB=${2} CPU_STEAL_PROB=${3} NB_CONFL_GPU_BUFFER=$NB_CONFL_GPU_BUFFER \
		NB_CONFL_CPU_BUFFER=$NB_CONFL_CPU_BUFFER CONFL_SPACE=$CONFL_SPACE
	fi
	if [ $? -ne 0 ]
	then
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		-d $DURATION -N 0.1 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f ${1} CPU_BACKOFF=300 \
		GPU_STEAL_PROB=${2} CPU_STEAL_PROB=${3} NB_CONFL_GPU_BUFFER=$NB_CONFL_GPU_BUFFER \
		NB_CONFL_CPU_BUFFER=$NB_CONFL_CPU_BUFFER CONFL_SPACE=$CONFL_SPACE
	fi
	if [ $? -ne 0 ]
	then
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		-d $DURATION -N 0.1 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f ${1} CPU_BACKOFF=300 \
		GPU_STEAL_PROB=${2} CPU_STEAL_PROB=${3} NB_CONFL_GPU_BUFFER=$NB_CONFL_GPU_BUFFER \
		NB_CONFL_CPU_BUFFER=$NB_CONFL_CPU_BUFFER CONFL_SPACE=$CONFL_SPACE
	fi
	if [ $? -ne 0 ]
	then
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		-d $DURATION -N 0.1 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f ${1} CPU_BACKOFF=300 \
		GPU_STEAL_PROB=${2} CPU_STEAL_PROB=${3} NB_CONFL_GPU_BUFFER=$NB_CONFL_GPU_BUFFER \
		NB_CONFL_CPU_BUFFER=$NB_CONFL_CPU_BUFFER CONFL_SPACE=$CONFL_SPACE
	fi
	if [ $? -ne 0 ]
	then
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		-d $DURATION -N 0.1 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f ${1} CPU_BACKOFF=300 \
		GPU_STEAL_PROB=${2} CPU_STEAL_PROB=${3} NB_CONFL_GPU_BUFFER=$NB_CONFL_GPU_BUFFER \
		NB_CONFL_CPU_BUFFER=$NB_CONFL_CPU_BUFFER CONFL_SPACE=$CONFL_SPACE
	fi
	if [ $? -ne 0 ]
	then
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		-d $DURATION -N 0.1 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f ${1} CPU_BACKOFF=300 \
		GPU_STEAL_PROB=${2} CPU_STEAL_PROB=${3} NB_CONFL_GPU_BUFFER=$NB_CONFL_GPU_BUFFER \
		NB_CONFL_CPU_BUFFER=$NB_CONFL_CPU_BUFFER CONFL_SPACE=$CONFL_SPACE
	fi
	if [ $? -ne 0 ]
	then
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		-d $DURATION -N 0.1 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f ${1} CPU_BACKOFF=300 \
		GPU_STEAL_PROB=${2} CPU_STEAL_PROB=${3} NB_CONFL_GPU_BUFFER=$NB_CONFL_GPU_BUFFER \
		NB_CONFL_CPU_BUFFER=$NB_CONFL_CPU_BUFFER CONFL_SPACE=$CONFL_SPACE
	fi
}

### TODO: no longer use REQUEST_GRANULARITY
# --> prob of stealing (at the begining of the batch)
#    CPU steal 10% of batches, 20%, 30%, 50%, 80% 100%
#    GPU steal 10% of batches, 20%, 30%, 50%, 80% 100%

# SETs (0.1%)/GETs(99.9%) (DELETEs next paper)
# -------------------
# CPU --> do round-robin
# GPU --> do the same --> the change is complecated

# reduces the load in one device, it goes into the shared queue
function doRun_HeTM_SHARED_steal {
	# BANK_PART=3 --> unif rand
	make clean ; make CMP_TYPE=COMPRESSED BANK_PART=6 \
		PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 \
		BENCH=MEMCD CPU_STEAL_ONLY_GETS=0 DISABLE_NON_BLOCKING=0 BMAP_GRAN_BITS=13 >/dev/null
	# make clean ; make CMP_TYPE=COMPRESSED BANK_PART=3 \
	# 	PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 \
	# 	BENCH=MEMCD CPU_STEAL_ONLY_GETS=0 >/dev/null
	for s in `seq 3 $SAMPLES`
	do
		### 64 k
		GPU_BLOCKS=128
		GPU_THREADS=128
		actualRun GPUsteal 0.0 0.0
		### 128 k
		GPU_BLOCKS=128
		GPU_THREADS=256
		actualRun GPUsteal 0.0 0.0
		### 256 k
		GPU_BLOCKS=256
		GPU_THREADS=256
		actualRun GPUsteal 0.0 0.0
		### 512 k
		GPU_BLOCKS=512
		GPU_THREADS=256
		actualRun GPUsteal 0.0 0.0
		### 1M
		GPU_BLOCKS=512
		GPU_THREADS=512
		actualRun GPUsteal 0.0 0.0
		###
		mv GPUsteal.csv ${1}_GPUsteal_00_s${s}
	done
	for s in `seq 3 $SAMPLES`
	do
		### 64 k
		GPU_BLOCKS=128
		GPU_THREADS=128
		actualRun GPUsteal 0.1 0.0
		### 128 k
		GPU_BLOCKS=128
		GPU_THREADS=256
		actualRun GPUsteal 0.1 0.0
		### 256 k
		GPU_BLOCKS=256
		GPU_THREADS=256
		actualRun GPUsteal 0.1 0.0
		### 512 k
		GPU_BLOCKS=512
		GPU_THREADS=256
		actualRun GPUsteal 0.1 0.0
		### 1M
		GPU_BLOCKS=512
		GPU_THREADS=512
		actualRun GPUsteal 0.1 0.0
		###
		mv GPUsteal.csv ${1}_GPUsteal_01_s${s}
	done
	for s in `seq 3 $SAMPLES`
	do
		### 64 k
		GPU_BLOCKS=128
		GPU_THREADS=128
		actualRun GPUsteal 0.5 0.0
		### 128 k
		GPU_BLOCKS=128
		GPU_THREADS=256
		actualRun GPUsteal 0.5 0.0
		### 256 k
		GPU_BLOCKS=256
		GPU_THREADS=256
		actualRun GPUsteal 0.5 0.0
		### 512 k
		GPU_BLOCKS=512
		GPU_THREADS=256
		actualRun GPUsteal 0.5 0.0
		### 1M
		GPU_BLOCKS=512
		GPU_THREADS=512
		actualRun GPUsteal 0.5 0.0
		###
		mv GPUsteal.csv ${1}_GPUsteal_05_s${s}
	done
	for s in `seq 3 $SAMPLES`
	do
		### 64 k
		GPU_BLOCKS=128
		GPU_THREADS=128
		actualRun GPUsteal 0.9 0.0
		### 128 k
		GPU_BLOCKS=128
		GPU_THREADS=256
		actualRun GPUsteal 0.9 0.0
		### 256 k
		GPU_BLOCKS=256
		GPU_THREADS=256
		actualRun GPUsteal 0.9 0.0
		### 512 k
		GPU_BLOCKS=512
		GPU_THREADS=256
		actualRun GPUsteal 0.9 0.0
		### 1M
		GPU_BLOCKS=512
		GPU_THREADS=512
		actualRun GPUsteal 0.9 0.0
		###
		mv GPUsteal.csv ${1}_GPUsteal_09_s${s}
	done
	for s in `seq 3 $SAMPLES`
	do
		### 64 k
		GPU_BLOCKS=128
		GPU_THREADS=128
		actualRun GPUsteal 1.0 0.0
		### 128 k
		GPU_BLOCKS=128
		GPU_THREADS=256
		actualRun GPUsteal 1.0 0.0
		### 256 k
		GPU_BLOCKS=256
		GPU_THREADS=256
		actualRun GPUsteal 1.0 0.0
		### 512 k
		GPU_BLOCKS=512
		GPU_THREADS=256
		actualRun GPUsteal 1.0 0.0
		### 1M
		GPU_BLOCKS=512
		GPU_THREADS=512
		actualRun GPUsteal 1.0 0.0
		###
		mv GPUsteal.csv ${1}_GPUsteal_10_s${s}
	done
}

function doRun_GPUonly {
	make clean ; make CMP_TYPE=COMPRESSED BANK_PART=6 CPUEn=0 PROFILE=1 \
		PR_MAX_RWSET_SIZE=20 -j 14 BENCH=MEMCD >/dev/null
		for s in `seq 3 $SAMPLES`
		do
			### 64 k
			GPU_BLOCKS=128
			GPU_THREADS=128
			actualRun GPUsteal 0.0 0.0
			### 128 k
			GPU_BLOCKS=128
			GPU_THREADS=256
			actualRun GPUsteal 0.0 0.0
			### 256 k
			GPU_BLOCKS=256
			GPU_THREADS=256
			actualRun GPUsteal 0.0 0.0
			### 512 k
			GPU_BLOCKS=512
			GPU_THREADS=256
			actualRun GPUsteal 0.0 0.0
			### 1M
			GPU_BLOCKS=512
			GPU_THREADS=512
			actualRun GPUsteal 0.0 0.0
			###
			mv GPUsteal.csv ${1}_GPUonly_s${s}
		done
}

function doRun_CPUonly {
	make clean ; make CMP_TYPE=DISABLED BANK_PART=6 USE_TSX_IMPL=1 INST_CPU=0 GPUEn=0 \
		PROFILE=1 PR_MAX_RWSET_SIZE=20 -j 14 BENCH=MEMCD >/dev/null
	for s in `seq 3 $SAMPLES`
	do
		### 64 k
		GPU_BLOCKS=128
		GPU_THREADS=128
		actualRun GPUsteal 0.0 0.0
		mv GPUsteal.csv ${1}_CPUonly_s${s}
	done
}

# TODO: add more input buffers for the GPU
NB_CONFL_GPU_BUFFER=10000
NB_CONFL_CPU_BUFFER=1000

DATASET=$LARGE_DATASET
CONFL_SPACE=8000000

SIZE_ZIPF=2000000
GPU_INPUT="GPU_input_${SIZE_ZIPF}_099_25165824.txt"
CPU_INPUT="CPU_input_${SIZE_ZIPF}_099_1310720.txt"

doRun_HeTM_SHARED_steal memcd_VERS_LARGE VERS
doRun_HeTM_SHARED_steal memcd_BMAP_LARGE BMAP

doRun_GPUonly memcd_LARGE
doRun_CPUonly memcd_LARGE

#################################

DATASET=$VSMALL_DATASET
CONFL_SPACE=800000

SIZE_ZIPF=2000000
GPU_INPUT="GPU_input_${SIZE_ZIPF}_099_25165824.txt"
CPU_INPUT="CPU_input_${SIZE_ZIPF}_099_1310720.txt"

doRun_HeTM_SHARED_steal memcd_VERS_SMALL VERS
doRun_HeTM_SHARED_steal memcd_BMAP_SMALL BMAP

doRun_GPUonly memcd_SMALL
doRun_CPUonly memcd_SMALL
