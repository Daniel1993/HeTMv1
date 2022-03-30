#!/bin/bash

DURATION=15000
SAMPLES=1

rm -f Bank.csv
rm -f File1.csv
rm -f File50.csv
rm -f File10.csv
rm -f File90.csv
rm -f File100.csv

TX_SIZE=1

function actualRun {
	timeout 40s ./bank -n $CPU_THREADS -b 40 -x 256 -T 1 -a $DATASET -d $DURATION \
			-R 0 -S $TX_SIZE -l ${1} -N 1 -f ${2} CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X ${3}
	if [ $? -ne 0 ]
	then
		timeout 40s ./bank -n $CPU_THREADS -b 40 -x 256 -T 1 -a $DATASET -d $DURATION \
		-R 0 -S $TX_SIZE -l ${1} -N 1 -f ${2} CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X ${3}
	fi
	if [ $? -ne 0 ]
	then
		timeout 40s ./bank -n $CPU_THREADS -b 40 -x 256 -T 1 -a $DATASET -d $DURATION \
		-R 0 -S $TX_SIZE -l ${1} -N 1 -f ${2} CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X ${3}
	fi
	if [ $? -ne 0 ]
	then
		timeout 40s ./bank -n $CPU_THREADS -b 40 -x 256 -T 1 -a $DATASET -d $DURATION \
		-R 0 -S $TX_SIZE -l ${1} -N 1 -f ${2} CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X ${3}
	fi
	if [ $? -ne 0 ]
	then
		timeout 40s ./bank -n $CPU_THREADS -b 40 -x 256 -T 1 -a $DATASET -d $DURATION \
		-R 0 -S $TX_SIZE -l ${1} -N 1 -f ${2} CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X ${3}
	fi
}

function runMultiBatchTime {
	actualRun ${1} ${2} 0.005
	actualRun ${1} ${2} 0.01
	actualRun ${1} ${2} 0.02
	actualRun ${1} ${2} 0.04
	actualRun ${1} ${2} 0.08
}

function doRunLargeDTST_GPUonly {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq $SAMPLES`
	do
		make clean ; make CMP_TYPE=COMPRESSED USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=400 \
			BANK_PART=${2} GPU_PART=0.505 CPU_PART=0.505 P_INTERSECT=1.00 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 DISABLE_NON_BLOCKING=0 \
			OVERLAP_CPY_BACK=1 >/dev/null
		CPU_BACKOFF=0
		GPU_BACKOFF=10000000
		runMultiBatchTime 1 File1_fast
		GPU_BACKOFF=9000000
		runMultiBatchTime 5 File5_fast
		GPU_BACKOFF=4000000
		runMultiBatchTime 20 File20_fast
		# CPU_BACKOFF=10000
		# GPU_BACKOFF=20000000
		# runMultiBatchTime 1 File1_slow
		# GPU_BACKOFF=15000000
		# runMultiBatchTime 5 File5_slow
		# GPU_BACKOFF=13000000
		# runMultiBatchTime 20 File20_slow

		mv File1_fast.csv  ${1}_1_fast_s${s}
		mv File5_fast.csv  ${1}_5_fast_s${s}
		mv File20_fast.csv ${1}_20_fast_s${s}
		mv File1_slow.csv  ${1}_1_slow_s${s}
		mv File5_slow.csv  ${1}_5_slow_s${s}
		mv File20_slow.csv ${1}_20_slow_s${s}
	done
}

function doRunLargeDTST_CPUonly {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq $SAMPLES`
	do
		make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=400 \
			BANK_PART=${2} GPU_PART=0.505 CPU_PART=0.505 P_INTERSECT=1.00 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 DISABLE_NON_BLOCKING=1 \
			OVERLAP_CPY_BACK=0 >/dev/null
		CPU_BACKOFF=0
		GPU_BACKOFF=10000000
		runMultiBatchTime 1 File1_fast
		GPU_BACKOFF=9000000
		runMultiBatchTime 5 File5_fast
		GPU_BACKOFF=4000000
		runMultiBatchTime 20 File20_fast
		# CPU_BACKOFF=10000
		# GPU_BACKOFF=20000000
		# runMultiBatchTime 1 File1_slow
		# GPU_BACKOFF=15000000
		# runMultiBatchTime 5 File5_slow
		# GPU_BACKOFF=13000000
		# runMultiBatchTime 20 File20_slow

		mv File1_fast.csv  ${1}_1_fast_s${s}
		mv File5_fast.csv  ${1}_5_fast_s${s}
		mv File20_fast.csv ${1}_20_fast_s${s}
		mv File1_slow.csv  ${1}_1_slow_s${s}
		mv File5_slow.csv  ${1}_5_slow_s${s}
		mv File20_slow.csv ${1}_20_slow_s${s}
	done
}

function doRunLargeDTST_VERS {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq $SAMPLES`
	do
		make clean ; make \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=400 \
			BANK_PART=${2} GPU_PART=0.505 CPU_PART=0.505 P_INTERSECT=1.00 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=0 >/dev/null
		CPU_BACKOFF=0
		GPU_BACKOFF=10000000
		runMultiBatchTime 1 File1_fast
		GPU_BACKOFF=9000000
		runMultiBatchTime 5 File5_fast
		GPU_BACKOFF=4000000
		runMultiBatchTime 20 File20_fast
		# CPU_BACKOFF=10000
		# GPU_BACKOFF=20000000
		# runMultiBatchTime 1 File1_slow
		# GPU_BACKOFF=15000000
		# runMultiBatchTime 5 File5_slow
		# GPU_BACKOFF=13000000
		# runMultiBatchTime 20 File20_slow

		mv File1_fast.csv  ${1}_1_fast_s${s}
		mv File5_fast.csv  ${1}_5_fast_s${s}
		mv File20_fast.csv ${1}_20_fast_s${s}
		mv File1_slow.csv  ${1}_1_slow_s${s}
		mv File5_slow.csv  ${1}_5_slow_s${s}
		mv File20_slow.csv ${1}_20_slow_s${s}
	done
}

function doRunLargeDTST_VERS_OVER {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq $SAMPLES`
	do
		make clean ; make \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=400 \
			BANK_PART=${2} GPU_PART=0.505 CPU_PART=0.505 P_INTERSECT=1.00 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 >/dev/null
		CPU_BACKOFF=0
		GPU_BACKOFF=10000000
		runMultiBatchTime 1 File1_fast
		GPU_BACKOFF=9000000
		runMultiBatchTime 5 File5_fast
		GPU_BACKOFF=4000000
		runMultiBatchTime 20 File20_fast
		# CPU_BACKOFF=10000
		# GPU_BACKOFF=20000000
		# runMultiBatchTime 1 File1_slow
		# GPU_BACKOFF=15000000
		# runMultiBatchTime 5 File5_slow
		# GPU_BACKOFF=13000000
		# runMultiBatchTime 20 File20_slow

		mv File1_fast.csv  ${1}_1_fast_s${s}
		mv File5_fast.csv  ${1}_5_fast_s${s}
		mv File20_fast.csv ${1}_20_fast_s${s}
		mv File1_slow.csv  ${1}_1_slow_s${s}
		mv File5_slow.csv  ${1}_5_slow_s${s}
		mv File20_slow.csv ${1}_20_slow_s${s}
	done
}

function doRunLargeDTST_VERS_BLOC {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq $SAMPLES`
	do
		make clean ; make \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=400 \
			BANK_PART=${2} GPU_PART=0.505 CPU_PART=0.505 P_INTERSECT=1.00 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 LOG_SIZE=4096 STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=1 OVERLAP_CPY_BACK=0 >/dev/null
		CPU_BACKOFF=0
		GPU_BACKOFF=10000000
		runMultiBatchTime 1 File1_fast
		GPU_BACKOFF=9000000
		runMultiBatchTime 5 File5_fast
		GPU_BACKOFF=4000000
		runMultiBatchTime 20 File20_fast
		# CPU_BACKOFF=10000
		# GPU_BACKOFF=20000000
		# runMultiBatchTime 1 File1_slow
		# GPU_BACKOFF=15000000
		# runMultiBatchTime 5 File5_slow
		# GPU_BACKOFF=13000000
		# runMultiBatchTime 20 File20_slow

		mv File1_fast.csv  ${1}_1_fast_s${s}
		mv File5_fast.csv  ${1}_5_fast_s${s}
		mv File20_fast.csv ${1}_20_fast_s${s}
		mv File1_slow.csv  ${1}_1_slow_s${s}
		mv File5_slow.csv  ${1}_5_slow_s${s}
		mv File20_slow.csv ${1}_20_slow_s${s}
	done
}

### Fixed the amount of CPU threads
CPU_THREADS=8

############### LARGE
###########################################################################
### 600MB
DATASET=150000000
############### GPU-only
doRunLargeDTST_GPUonly inter_GPUonly_rand_sep 10

############## CPU-only
doRunLargeDTST_CPUonly inter_CPUonly_rand_sep 10

############## VERS
doRunLargeDTST_VERS inter_VERS_rand_sep 10

############## VERS_OVER
doRunLargeDTST_VERS_OVER inter_VERS_OVER_rand_sep 10

############## VERS_BLOC
doRunLargeDTST_VERS_BLOC inter_VERS_BLOC_rand_sep 10

############## BMAP
# doRunLargeDTST_BMAP inter_BMAP_rand_sep 1
###########################################################################
#
# ############### SMALL
# ###########################################################################
# ### 60MB
# DATASET=15000000
# # ############### GPU-only
# doRunLargeDTST_GPUonly inter_GPUonly_rand_sep_SMALL 10
#
# ############## CPU-only
# doRunLargeDTST_CPUonly inter_CPUonly_rand_sep_SMALL 10
#
# ############## VERS
# doRunLargeDTST_VERS inter_VERS_rand_sep_SMALL 10
#
# ############## VERS_OVER
# doRunLargeDTST_VERS_OVER inter_VERS_OVER_rand_sep_SMALL 10
#
# ############## VERS_BLOC
# doRunLargeDTST_VERS_BLOC inter_VERS_BLOC_rand_sep_SMALL 10

############## BMAP
# doRunLargeDTST_BMAP inter_BMAP_rand_sep_SMALL 1
###########################################################################
#
# ################ Zipf
#
# ############### LARGE
# ###########################################################################
# ### 1GB
# DATASET=250000000
#
# ############### GPU-only
# doRunLargeDTST_GPUonly inter_GPUonly_cont_sep 5
#
# ############## CPU-only
# doRunLargeDTST_CPUonly inter_CPUonly_cont_sep 5
#
# ############## VERS
# doRunLargeDTST_VERS inter_VERS_cont_sep 5
#
# ############## BMAP
# doRunLargeDTST_BMAP inter_BMAP_cont_sep 5
# ###########################################################################
#
# ############### SMALL
# ###########################################################################
# ### 100MB
# DATASET=25000000
#
# ############### GPU-only
# doRunLargeDTST_GPUonly inter_GPUonly_cont_sep_SMALL 5
#
# ############## CPU-only
# doRunLargeDTST_CPUonly inter_CPUonly_cont_sep_SMALL 5
#
# ############## VERS
# doRunLargeDTST_VERS inter_VERS_cont_sep_SMALL 5
#
# ############## BMAP
# doRunLargeDTST_BMAP inter_BMAP_cont_sep_SMALL 5
# ###########################################################################

mkdir -p array_inter_HTM_batch_duration
mv *_s* array_inter_HTM_batch_duration
