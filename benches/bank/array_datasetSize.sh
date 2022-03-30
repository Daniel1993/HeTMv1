#!/bin/bash

DURATION=12000
SAMPLES=10
#./makeTM.sh

### Fixed the amount of CPU threads
CPU_THREADS=8

rm -f Bank.csv

function doRunLargeDTST {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		# 100M 500M 1G 1.5G
		timeout 40s ./bank -n $CPU_THREADS -b 80 -x 256 -a 200000    -d $DURATION -R 0 -S 4 -l 100 -N 1 -T 1 CPU_BACKOFF=1 GPU_BACKOFF=700000 -X 0.2
		timeout 40s ./bank -n $CPU_THREADS -b 80 -x 256 -a 2000000   -d $DURATION -R 0 -S 4 -l 100 -N 1 -T 1 CPU_BACKOFF=1 GPU_BACKOFF=400000 -X 0.2
		timeout 40s ./bank -n $CPU_THREADS -b 80 -x 256 -a 20000000  -d $DURATION -R 0 -S 4 -l 100 -N 1 -T 1 CPU_BACKOFF=1 GPU_BACKOFF=290000 -X 0.2
		timeout 40s ./bank -n $CPU_THREADS -b 80 -x 256 -a 100000000 -d $DURATION -R 0 -S 4 -l 100 -N 1 -T 1 CPU_BACKOFF=1 GPU_BACKOFF=600000 -X 0.2
		timeout 40s ./bank -n $CPU_THREADS -b 80 -x 256 -a 150000000 -d $DURATION -R 0 -S 4 -l 100 -N 1 -T 1 CPU_BACKOFF=1 GPU_BACKOFF=650000 -X 0.2
		timeout 40s ./bank -n $CPU_THREADS -b 80 -x 256 -a 200000000 -d $DURATION -R 0 -S 4 -l 100 -N 1 -T 1 CPU_BACKOFF=1 GPU_BACKOFF=650000 -X 0.2
		timeout 40s ./bank -n $CPU_THREADS -b 80 -x 256 -a 250000000 -d $DURATION -R 0 -S 4 -l 100 -N 1 -T 1 CPU_BACKOFF=1 GPU_BACKOFF=600000 -X 0.2
		mv Bank.csv ${1}${s}
	done
}

###########################################################################

############### GPU-only
make clean ; make CMP_TYPE=COMPRESSED DISABLE_RS=1 USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=9 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
	STM_LOG_BUFFER_SIZE=256 DISABLE_NON_BLOCKING=1 \
	OVERLAP_CPY_BACK=0 >/dev/null
doRunLargeDTST GPUonly_rand_sep_DISABLED_s

############## CPU-only
make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
		BANK_PART=9 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
		BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
		STM_LOG_BUFFER_SIZE=256 DISABLE_NON_BLOCKING=1 \
		OVERLAP_CPY_BACK=0 >/dev/null
doRunLargeDTST CPUonly_rand_sep_DISABLED_s

############## VERS
make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=9 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
	STM_LOG_BUFFER_SIZE=256 DISABLE_NON_BLOCKING=0 \
	OVERLAP_CPY_BACK=1 >/dev/null
doRunLargeDTST VERS_rand_sep_s

############## VERS blocking
make clean ; make CMP_TYPE=COMPRESSED LOG_TYPE=BMAP USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
	BANK_PART=9 GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
	BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
	STM_LOG_BUFFER_SIZE=256 DISABLE_NON_BLOCKING=1 \
	OVERLAP_CPY_BACK=0 >/dev/null
doRunLargeDTST VERS_bloc_rand_sep_s
###########################################################################

mkdir -p array_dataset
mv *_s* array_dataset/
