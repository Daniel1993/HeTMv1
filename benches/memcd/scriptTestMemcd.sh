#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1
filename_tsx="Bank_TSX"
filename_tiny="Bank_Tiny"

GPU_PART="1.0"
CPU_PART="0.0"
P_INTERSECT="0.0"
DURATION=12000
BLOCKS="2 4 8 16 32 64 256 512 1024" # 512
THREADS="512" #"2 4 8 16 32 64 96 256 320 512 640 768 1024"
BATCH_SIZE="4"
SAMPLES=1
#./makeTM.sh

CPU_THREADS=4
# LOW_CPU_THREADS=2
LARGE_HIGH_CPU_THREADS=4
LARGE_VERY_HIGH_CPU_THREADS=8
SMALL_HIGH_CPU_THREADS=4
SMALL_VERY_HIGH_CPU_THREADS=8

rm -f Bank.csv

LARGE_DATASET=32768
SMALL_DATASET=4096

# memcached read only

### TODO: use shared 0, 30, 50

### TODO: find the best combination of parameters
function doRunSmallDTST {
	timeout 50s ./memcd -n $CPU_THREADS -l 32 -b 1024 -x 512 -T 1 -a $SMALL_DATASET -d $DURATION -N 0  -S 0 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 32 -b 1024 -x 512 -T 1 -a $SMALL_DATASET -d $DURATION -N 2  -S 0 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 32 -b 1024 -x 512 -T 1 -a $SMALL_DATASET -d $DURATION -N 4  -S 0 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 32 -b 1024 -x 512 -T 1 -a $SMALL_DATASET -d $DURATION -N 8  -S 0 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 32 -b 1024 -x 512 -T 1 -a $SMALL_DATASET -d $DURATION -N 10 -S 0 -G $GPU_INPUT -C $CPU_INPUT
}

function doRunDTST {
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 2048  -d $DURATION -N 5  -S 0 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 4096  -d $DURATION -N 5  -S 0 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 8192  -d $DURATION -N 5  -S 0 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 16384 -d $DURATION -N 5  -S 0 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 32768 -d $DURATION -N 5  -S 0 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 65536 -d $DURATION -N 5  -S 0 -G $GPU_INPUT -C $CPU_INPUT
}

function doRunDTST10 {
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 2048  -d $DURATION -N 5  -S 10 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 4096  -d $DURATION -N 5  -S 10 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 8192  -d $DURATION -N 5  -S 10 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 16384 -d $DURATION -N 5  -S 10 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 32768 -d $DURATION -N 5  -S 10 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 65536 -d $DURATION -N 5  -S 10 -G $GPU_INPUT -C $CPU_INPUT
}

function doRunDTST30 {
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 2048  -d $DURATION -N 5  -S 30 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 4096  -d $DURATION -N 5  -S 30 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 8192  -d $DURATION -N 5  -S 30 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 16384 -d $DURATION -N 5  -S 30 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 32768 -d $DURATION -N 5  -S 30 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 65536 -d $DURATION -N 5  -S 30 -G $GPU_INPUT -C $CPU_INPUT
}

function doRunDTST50 {
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 2048  -d $DURATION -N 5  -S 50 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 4096  -d $DURATION -N 5  -S 50 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 8192  -d $DURATION -N 5  -S 50 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 16384 -d $DURATION -N 5  -S 50 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 32768 -d $DURATION -N 5  -S 50 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 65536 -d $DURATION -N 5  -S 50 -G $GPU_INPUT -C $CPU_INPUT
}

function doRunDTST100 {
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 2048  -d $DURATION -N 5  -S 100 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 4096  -d $DURATION -N 5  -S 100 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 8192  -d $DURATION -N 5  -S 100 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 16384 -d $DURATION -N 5  -S 100 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 32768 -d $DURATION -N 5  -S 100 -G $GPU_INPUT -C $CPU_INPUT
	timeout 50s ./memcd -n $CPU_THREADS -l 16 -b 1024 -x 512 -T 1 -a 65536 -d $DURATION -N 5  -S 100 -G $GPU_INPUT -C $CPU_INPUT
}

CPU_THREADS=14

cd java_zipf
SIZE_ZIPF=49999999
GPU_INPUT="GPU_input_${SIZE_ZIPF}_099_4194304.txt"
CPU_INPUT="CPU_input_${SIZE_ZIPF}_099_229376.txt"
if [ ! -f ../$GPU_INPUT ]
then
	java Main $LARGE_DATASET 0.99 4194304 > ../$GPU_INPUT
fi
if [ ! -f ../$CPU_INPUT ]
then
	java Main $LARGE_DATASET 0.99 229376 > ../$CPU_INPUT
fi
cd ..

make clean ; make CMP_TYPE=DISABLED BANK_PART=1 CPUEn=0 PROFILE=1 PR_MAX_RWSET_SIZE=8 -j 14 BENCH=MEMCD
for s in `seq $SAMPLES`
do
	# doRunSmallDTST
	doRunDTST
	mv Bank.csv memcd_GPUonly_s${s}

	### TODO: ignore?
	# doRunDTST10
	# mv Bank.csv memcd10_GPUonly_s${s}
	# doRunDTST30
	# mv Bank.csv memcd30_GPUonly_s${s}
	# doRunDTST50
	# mv Bank.csv memcd50_GPUonly_s${s}
	# doRunDTST100
	# mv Bank.csv memcd100_GPUonly_s${s}
done

make clean ; make CMP_TYPE=DISABLED BANK_PART=1 USE_TSX_IMPL=1 INST_CPU=0 GPUEn=0 \
	PROFILE=1 PR_MAX_RWSET_SIZE=8 -j 14 BENCH=MEMCD
for s in `seq $SAMPLES`
do
	# doRunSmallDTST
	doRunDTST
	mv Bank.csv memcd_CPUonly_s${s}

	### TODO: ignore?
	# doRunDTST10
	# mv Bank.csv memcd10_CPUonly_s${s}
	# doRunDTST30
	# mv Bank.csv memcd30_CPUonly_s${s}
	# doRunDTST50
	# mv Bank.csv memcd50_CPUonly_s${s}
	# doRunDTST100
	# mv Bank.csv memcd100_CPUonly_s${s}
done

function RunExperiment {
	# solution is in $1
	# prob intersect is in $2
	make clean ; make CMP_TYPE=COMPRESSED BANK_PART=1 \
		PR_MAX_RWSET_SIZE=8 LOG_TYPE=$1 USE_TSX_IMPL=1 PROFILE=1 -j 14 BENCH=MEMCD
	for s in `seq $SAMPLES`
	do
		# doRunSmallDTST
		doRunDTST
		mv Bank.csv memcd_${1}_${2}_s${s}
		doRunDTST10
		mv Bank.csv memcd10_${1}_${2}_s${s}
		doRunDTST30
		mv Bank.csv memcd30_${1}_${2}_s${s}
		doRunDTST50
		mv Bank.csv memcd50_${1}_${2}_s${s}
		doRunDTST100
		mv Bank.csv memcd100_${1}_${2}_s${s}
	done
}

RunExperiment VERS 0.0
RunExperiment BMAP 0.0
