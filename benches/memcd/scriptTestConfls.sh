#!/bin/bash

DURATION=20000
BATCH_SIZE="4"
SAMPLES=3
#./makeTM.sh

CPU_THREADS=4
# LOW_CPU_THREADS=2
LARGE_HIGH_CPU_THREADS=4
LARGE_VERY_HIGH_CPU_THREADS=14
SMALL_HIGH_CPU_THREADS=4
SMALL_VERY_HIGH_CPU_THREADS=14

rm -f Bank.csv GPUload.csv CPUload.csv

LARGE_DATASET=1000000
LCONFL_SPACE=500000
VSMALL_DATASET=100000
SCONFL_SPACE=50000

GPU_BLOCKS=40
GPU_THREADS=256

CPU_BACKOFF=1
GPU_BACKOFF=16000

CPU_THREADS=8
CPU_TIME=0.02

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
	timeout 50s ./memcd -n $CPU_THREADS -l 8 -b $GPU_BLOCKS -x $GPU_THREADS -T 1 -a $DATASET \
		-d $DURATION -N 0.1 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f ${1} CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF \
		GPU_STEAL_PROB=${2} CPU_STEAL_PROB=${3} NB_CONFL_GPU_BUFFER=$NB_CONFL_GPU_BUFFER \
		NB_CONFL_CPU_BUFFER=$NB_CONFL_CPU_BUFFER CONFL_SPACE=$CONFL_SPACE -X $CPU_TIME
	if [ $? -ne 0 ]
	then
		timeout 50s ./memcd -n $CPU_THREADS -l 8 -b $GPU_BLOCKS -x $GPU_THREADS -T 1 -a $DATASET \
			-d $DURATION -N 0.1 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f ${1} CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF \
			GPU_STEAL_PROB=${2} CPU_STEAL_PROB=${3} NB_CONFL_GPU_BUFFER=$NB_CONFL_GPU_BUFFER \
			NB_CONFL_CPU_BUFFER=$NB_CONFL_CPU_BUFFER CONFL_SPACE=$CONFL_SPACE -X $CPU_TIME
	fi
	if [ $? -ne 0 ]
	then
		timeout 50s ./memcd -n $CPU_THREADS -l 8 -b $GPU_BLOCKS -x $GPU_THREADS -T 1 -a $DATASET \
			-d $DURATION -N 0.1 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f ${1} CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF \
			GPU_STEAL_PROB=${2} CPU_STEAL_PROB=${3} NB_CONFL_GPU_BUFFER=$NB_CONFL_GPU_BUFFER \
			NB_CONFL_CPU_BUFFER=$NB_CONFL_CPU_BUFFER CONFL_SPACE=$CONFL_SPACE -X $CPU_TIME
	fi
	if [ $? -ne 0 ]
	then
		timeout 50s ./memcd -n $CPU_THREADS -l 8 -b $GPU_BLOCKS -x $GPU_THREADS -T 1 -a $DATASET \
			-d $DURATION -N 0.1 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f ${1} CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF \
			GPU_STEAL_PROB=${2} CPU_STEAL_PROB=${3} NB_CONFL_GPU_BUFFER=$NB_CONFL_GPU_BUFFER \
			NB_CONFL_CPU_BUFFER=$NB_CONFL_CPU_BUFFER CONFL_SPACE=$CONFL_SPACE -X $CPU_TIME
	fi
	if [ $? -ne 0 ]
	then
		timeout 50s ./memcd -n $CPU_THREADS -l 8 -b $GPU_BLOCKS -x $GPU_THREADS -T 1 -a $DATASET \
			-d $DURATION -N 0.1 -S 0 -G $GPU_INPUT -C $CPU_INPUT -f ${1} CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF \
			GPU_STEAL_PROB=${2} CPU_STEAL_PROB=${3} NB_CONFL_GPU_BUFFER=$NB_CONFL_GPU_BUFFER \
			NB_CONFL_CPU_BUFFER=$NB_CONFL_CPU_BUFFER CONFL_SPACE=$CONFL_SPACE -X $CPU_TIME
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
		PR_MAX_RWSET_SIZE=20 LOG_TYPE=VERS USE_TSX_IMPL=1 PROFILE=1 -j 14 \
		BENCH=MEMCD CPU_STEAL_ONLY_GETS=0 DISABLE_NON_BLOCKING=${1} \
		LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 OVERLAP_CPY_BACK=${2} >/dev/null
	for s in `seq 1 $SAMPLES`
	do
		# ### DO GPU load
		actualRun GPUload 0.0 0.0
		# cp GPUload.csv CPUload.csv
		# actualRun GPUload 0.0 0.1
		actualRun GPUload 0.0 0.2
		# actualRun GPUload 0.0 0.3
		actualRun GPUload 0.0 0.4
		# actualRun GPUload 0.0 0.5
		actualRun GPUload 0.0 0.6
		# actualRun GPUload 0.0 0.7
		actualRun GPUload 0.0 0.8
		# actualRun GPUload 0.0 0.9
		actualRun GPUload 0.0 1.0
		### DO CPU load
		# actualRun CPUload 0.1 0.0
		# actualRun CPUload 0.2 0.0
		# # actualRun CPUload 0.3 0.0
		# actualRun CPUload 0.4 0.0
		# # actualRun CPUload 0.5 0.0
		# actualRun CPUload 0.6 0.0
		# # actualRun CPUload 0.7 0.0
		# actualRun CPUload 0.8 0.0
		# # actualRun CPUload 0.9 0.0
		# actualRun CPUload 1.0 0.0
		###
		mv GPUload.csv ${3}_GPUload_s${s}
		# # rm GPUload.csv
		# mv CPUload.csv ${3}_CPUload_s${s}
	done
}

DATASET=$LARGE_DATASET

SIZE_ZIPF=2000000
GPU_INPUT="GPU_input_${SIZE_ZIPF}_099_25165824.txt"
CPU_INPUT="CPU_input_${SIZE_ZIPF}_099_1310720.txt"

### 64 k

NB_CONFL_CPU_BUFFER=0

GPU_BLOCKS=40
GPU_THREADS=256
# GPU_THREADS=256

DATASET=$LARGE_DATASET
CONFL_SPACE=$LCONFL_SPACE
doRun_HeTM_SHARED_steal 0 1 memcd_LARGE_NON_BLOC_OVER
doRun_HeTM_SHARED_steal 1 0 memcd_LARGE_BLOC
DATASET=$VSMALL_DATASET
CONFL_SPACE=$SCONFL_SPACE
doRun_HeTM_SHARED_steal 0 1 memcd_SMALL_NON_BLOC_OVER
doRun_HeTM_SHARED_steal 1 0 memcd_SMALL_BLOC

mkdir -p DATA_memcdInterConfl
mv memcd_SMALL*_s* DATA_memcdInterConfl
mv memcd_LARGE*_s* DATA_memcdInterConfl
