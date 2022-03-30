#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1
filename_tsx="Bank_TSX"
filename_tiny="Bank_Tiny"

GPU_PART="1.0"
CPU_PART="0.0"
P_INTERSECT="0.0"
DURATION=10000
BLOCKS="2 4 8 16 32 64 256 512 1024" # 512
THREADS="512" #"2 4 8 16 32 64 96 256 320 512 640 768 1024"
BATCH_SIZE="4"
SAMPLES=2
#./makeTM.sh

CPU_THREADS=4
# LOW_CPU_THREADS=2
LARGE_HIGH_CPU_THREADS=4
LARGE_VERY_HIGH_CPU_THREADS=14
SMALL_HIGH_CPU_THREADS=4
SMALL_VERY_HIGH_CPU_THREADS=14

rm -f Bank.csv GPUload.csv CPUload.csv

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
	make clean ; make CMP_TYPE=COMPRESSED BANK_PART=5 \
		PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 \
		BENCH=MEMCD CPU_STEAL_ONLY_GETS=0 >/dev/null
	# make clean ; make CMP_TYPE=COMPRESSED BANK_PART=3 \
	# 	PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 \
	# 	BENCH=MEMCD CPU_STEAL_ONLY_GETS=0 >/dev/null
	for s in `seq 1 $SAMPLES`
	do
		# ### DO GPU load
		actualRun GPUload 0.0 0.0
		cp GPUload.csv CPUload.csv
		cp GPUload.csv CPUload_GETs.csv
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
		actualRun CPUload 0.2 0.0
		# actualRun CPUload 0.3 0.0
		actualRun CPUload 0.4 0.0
		# actualRun CPUload 0.5 0.0
		actualRun CPUload 0.6 0.0
		# actualRun CPUload 0.7 0.0
		actualRun CPUload 0.8 0.0
		# actualRun CPUload 0.9 0.0
		actualRun CPUload 1.0 0.0
		###
		mv GPUload.csv ${1}_GPUload_s${s}
		# # rm GPUload.csv
		mv CPUload.csv ${1}_CPUload_s${s}
	done
	# make clean ; make CMP_TYPE=COMPRESSED BANK_PART=2 \
	# 	PR_MAX_RWSET_SIZE=20 LOG_TYPE=${2} USE_TSX_IMPL=1 PROFILE=1 -j 14 \
	# 	BENCH=MEMCD CPU_STEAL_ONLY_GETS=1 >/dev/null
	# for s in `seq 1 $SAMPLES`
	# do
	# 	### DO CPU load
	# 	# actualRun CPUload_GETs 0.0 0.1
	# 	actualRun CPUload_GETs 0.0 0.2
	# 	# actualRun CPUload_GETs 0.0 0.3
	# 	# actualRun CPUload_GETs 0.0 0.4
	# 	# # actualRun CPUload 0.5 0.0
	# 	# actualRun CPUload_GETs 0.0 0.5
	# 	# actualRun CPUload 0.7 0.0
	# 	actualRun CPUload_GETs 0.0 0.8
	# 	# actualRun CPUload 0.9 0.0
	# 	actualRun CPUload_GETs 0.0 1.0
	# 	###
	# 	mv CPUload_GETs.csv ${1}_CPUload_GETs_s${s}
	# done
}

function doRun_GPUonly {
	# TODO: for some reason DISABLED is slower...
	make clean ; make CMP_TYPE=COMPRESSED BANK_PART=5 CPUEn=0 PROFILE=1 \
		PR_MAX_RWSET_SIZE=20 -j 14 BENCH=MEMCD >/dev/null
	for s in `seq 1 $SAMPLES`
	do
		### DO GPU load
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.1 -S 0 -G $GPU_INPUT -C $CPU_INPUT CPU_BACKOFF=300 \
			GPU_STEAL_PROB=0 CPU_STEAL_PROB=0 \
			NB_CONFL_GPU_BUFFER=$NB_CONFL_GPU_BUFFER \
			NB_CONFL_CPU_BUFFER=$NB_CONFL_CPU_BUFFER CONFL_SPACE=$CONFL_SPACE
		tail -n 1 Bank.csv > /tmp/BankLastLine.csv
		cat /tmp/BankLastLine.csv >> Bank.csv # duplicates last line
		mv Bank.csv ${1}_s${s}
	done
}

function doRun_CPUonly {
	make clean ; make CMP_TYPE=DISABLED BANK_PART=5 USE_TSX_IMPL=1 INST_CPU=0 GPUEn=0 \
		PROFILE=1 PR_MAX_RWSET_SIZE=20 -j 14 BENCH=MEMCD >/dev/null
	for s in `seq 1 $SAMPLES`
	do
		### DO GPU load
		timeout 50s ./memcd -n $CPU_THREADS -l 16 -b $GPU_BLOCKS -x $GPU_THREADS -T 4 -a $DATASET \
		  -d $DURATION -N 0.1 -S 0 -G $GPU_INPUT -C $CPU_INPUT CPU_BACKOFF=300 \
			GPU_STEAL_PROB=0 CPU_STEAL_PROB=0 \
			NB_CONFL_GPU_BUFFER=$NB_CONFL_GPU_BUFFER \
			NB_CONFL_CPU_BUFFER=$NB_CONFL_CPU_BUFFER CONFL_SPACE=$CONFL_SPACE
		tail -n 1 Bank.csv > /tmp/BankLastLine.csv
		cat /tmp/BankLastLine.csv >> Bank.csv # duplicates last line
		mv Bank.csv ${1}_s${s}
	done
}

# TODO: add more input buffers for the GPU
NB_CONFL_GPU_BUFFER=10000
NB_CONFL_CPU_BUFFER=1000

#
# # DATASET=$SMALL_DATASET #medium
# #
# # doRun_HeTM_SHARED_noConfl memcd_SHARED_noConfl_VERS VERS
# # doRun_HeTM_SHARED_noConfl memcd_SHARED_noConfl_BMAP BMAP
# #
# # doRun_HeTM_SHARED_steal memcd_SHARED_steal_VERS VERS
# # doRun_HeTM_SHARED_steal memcd_SHARED_steal_BMAP BMAP
# #
# # doRun_HeTM memcd_VERS VERS
# # doRun_HeTM memcd_BMAP BMAP
# #
# # doRun_GPUonly memcd_GPUonly
# # doRun_CPUonly memcd_CPUonly

DATASET=$LARGE_DATASET
CONFL_SPACE=8000000

SIZE_ZIPF=2000000
GPU_INPUT="GPU_input_${SIZE_ZIPF}_099_25165824.txt"
CPU_INPUT="CPU_input_${SIZE_ZIPF}_099_1310720.txt"

### 64 k

GPU_BLOCKS=128
GPU_THREADS=128
# GPU_THREADS=256
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_VERS_LARGE_B_64k VERS
doRun_HeTM_SHARED_steal memcd_SHARED_steal_BMAP_LARGE_B_64k BMAP

# doRun_GPUonly memcd_GPUonly_LARGE_B_64k
# doRun_CPUonly memcd_CPUonly_LARGE_B_64k

# ### 128 k
#
# GPU_BLOCKS=128
# GPU_THREADS=256
# # GPU_THREADS=256
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_VERS_LARGE_B_128k VERS
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_BMAP_LARGE_B_128k BMAP
# #
# # doRun_GPUonly memcd_GPUonly_LARGE_B_128k
# # doRun_CPUonly memcd_CPUonly_LARGE_B_128k

### 256 k

GPU_BLOCKS=256
GPU_THREADS=256
# GPU_THREADS=256
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_VERS_LARGE_B_256k VERS
doRun_HeTM_SHARED_steal memcd_SHARED_steal_BMAP_LARGE_B_256k BMAP

# doRun_GPUonly memcd_GPUonly_LARGE_B_256k
# doRun_CPUonly memcd_CPUonly_LARGE_B_256k

# ### 512 k
#
# GPU_BLOCKS=512
# GPU_THREADS=256
# # GPU_THREADS=256
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_VERS_LARGE_B_512k VERS
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_BMAP_LARGE_B_512k BMAP
# #
# # doRun_GPUonly memcd_GPUonly_LARGE_B_512k
# # doRun_CPUonly memcd_CPUonly_LARGE_B_512k
#
# ### 1 M

GPU_BLOCKS=512
GPU_THREADS=512
# GPU_THREADS=256
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_VERS_LARGE_B_1M VERS
doRun_HeTM_SHARED_steal memcd_SHARED_steal_BMAP_LARGE_B_1M BMAP

# doRun_GPUonly memcd_GPUonly_LARGE_B_1M
# doRun_CPUonly memcd_CPUonly_LARGE_B_1M


# GPU_THREADS=512
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_VERS_LARGE_Lk_Lb VERS
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_BMAP_LARGE_Lk_Lb BMAP
#
# doRun_GPUonly memcd_GPUonly_LARGE_Lk_Lb
# doRun_CPUonly memcd_CPUonly_LARGE_Lk_Lb

# doRun_HeTM memcd_VERS_LARGE VERS
# doRun_HeTM memcd_BMAP_LARGE BMAP

#################################

DATASET=$VSMALL_DATASET
CONFL_SPACE=800000
#GPU_BLOCKS=256
# GPU_BLOCKS=512

# doRun_HeTM_SHARED_noConfl memcd_SHARED_noConfl_VERS_SMALL VERS
# doRun_HeTM_SHARED_noConfl memcd_SHARED_noConfl_BMAP_SMALL BMAP

SIZE_ZIPF=2000000
GPU_INPUT="GPU_input_${SIZE_ZIPF}_099_25165824.txt"
CPU_INPUT="CPU_input_${SIZE_ZIPF}_099_1310720.txt"

### 64 k

GPU_THREADS=128
GPU_BLOCKS=128
# GPU_THREADS=256
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_VERS_SMALL_B_64k VERS
doRun_HeTM_SHARED_steal memcd_SHARED_steal_BMAP_SMALL_B_64k BMAP

# doRun_GPUonly memcd_GPUonly_SMALL_B_64k
# doRun_CPUonly memcd_CPUonly_SMALL_B_64k
#
# ### 128 k
#
# GPU_THREADS=128
# GPU_BLOCKS=256
# # GPU_THREADS=256
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_VERS_SMALL_B_128k VERS
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_BMAP_SMALL_B_128k BMAP
# #
# # doRun_GPUonly memcd_GPUonly_SMALL_B_128k
# # doRun_CPUonly memcd_CPUonly_SMALL_B_128k

### 256 k

GPU_THREADS=256
GPU_BLOCKS=256
# GPU_THREADS=256
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_VERS_SMALL_B_256k VERS
doRun_HeTM_SHARED_steal memcd_SHARED_steal_BMAP_SMALL_B_256k BMAP

# doRun_GPUonly memcd_GPUonly_SMALL_B_256k
# doRun_CPUonly memcd_CPUonly_SMALL_B_256k

# ### 512 k
#
# GPU_THREADS=512
# GPU_BLOCKS=256
# # GPU_THREADS=256
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_VERS_SMALL_B_512k VERS
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_BMAP_SMALL_B_512k BMAP
# #
# # doRun_GPUonly memcd_GPUonly_SMALL_B_512k
# # doRun_CPUonly memcd_CPUonly_SMALL_B_512k

### 1 M

GPU_THREADS=512
GPU_BLOCKS=256
# GPU_THREADS=256
# doRun_HeTM_SHARED_steal memcd_SHARED_steal_VERS_SMALL_B_1M VERS
doRun_HeTM_SHARED_steal memcd_SHARED_steal_BMAP_SMALL_B_1M BMAP

# doRun_GPUonly memcd_GPUonly_SMALL_B_1M
# doRun_CPUonly memcd_CPUonly_SMALL_B_1M
