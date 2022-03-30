#!/bin/bash

DURATION=12000
SAMPLES=10
DURATION_ORG=12000
DURATION_GPU=8000
#./makeTM.sh
DURATION=$DURATION_ORG

rm -f Bank.csv
rm -f File50.csv

CPU_BACKOFF=0
GPU_BACKOFF=600000
TX_SIZE=4
CPU_THREADS=8

function actualRun {
	timeout 40s ./bank -n $CPU_THREADS -b 80 -x 256 -T 1 -a $DATASET -d $DURATION \
		-R 0 -S $TX_SIZE -l ${1} -N 1 -f ${2} CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.08
	if [ $? -ne 0 ]
	then
		timeout 40s ./bank -n $CPU_THREADS -b 80 -x 256 -T 1 -a $DATASET -d $DURATION \
		-R 0 -S $TX_SIZE -l ${1} -N 1 -f ${2} CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.08
	fi
	if [ $? -ne 0 ]
	then
		timeout 40s ./bank -n $CPU_THREADS -b 80 -x 256 -T 1 -a $DATASET -d $DURATION \
		-R 0 -S $TX_SIZE -l ${1} -N 1 -f ${2} CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.08
	fi
	if [ $? -ne 0 ]
	then
		timeout 40s ./bank -n $CPU_THREADS -b 80 -x 256 -T 1 -a $DATASET -d $DURATION \
		-R 0 -S $TX_SIZE -l ${1} -N 1 -f ${2} CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.08
	fi
	if [ $? -ne 0 ]
	then
		timeout 40s ./bank -n $CPU_THREADS -b 80 -x 256 -T 1 -a $DATASET -d $DURATION \
		-R 0 -S $TX_SIZE -l ${1} -N 1 -f ${2} CPU_BACKOFF=$CPU_BACKOFF GPU_BACKOFF=$GPU_BACKOFF -X 0.08
	fi
}

function doRunLargeDTST_GPUonly {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		make clean ; make CMP_TYPE=COMPRESSED DISABLE_RS=1 USE_TSX_IMPL=1 CPUEn=0 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 DISABLE_NON_BLOCKING=0 \
			OVERLAP_CPY_BACK=1 >/dev/null
		actualRun 50 File50
		# actualRun 1  File1
		# actualRun 50 File50

		# all the same point (no inter-confl with 1 device)
		tail -n 1 File50.csv > /tmp/auxFile50.csv
		# tail -n 1 File1.csv  > /tmp/auxFile1.csv
		# tail -n 1 File50.csv > /tmp/auxFile50.csv
###
		# cat /tmp/auxFile1.csv >> File1.csv
		# cat /tmp/auxFile1.csv >> File1.csv
		# cat /tmp/auxFile1.csv >> File1.csv
		# cat /tmp/auxFile1.csv >> File1.csv
###
		cat /tmp/auxFile50.csv >> File50.csv
		cat /tmp/auxFile50.csv >> File50.csv
		cat /tmp/auxFile50.csv >> File50.csv
		cat /tmp/auxFile50.csv >> File50.csv
		cat /tmp/auxFile50.csv >> File50.csv
###
		# cat /tmp/auxFile50.csv >> File50.csv
		# cat /tmp/auxFile50.csv >> File50.csv
		# cat /tmp/auxFile50.csv >> File50.csv
		# cat /tmp/auxFile50.csv >> File50.csv
###

		mv File50.csv ${1}_50_s${s}

		# mv File1.csv ${1}_1_s${s}
		# mv File50.csv ${1}_50_s${s}
	done
}

function doRunLargeDTST_CPUonly {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		make clean ; make INST_CPU=0 GPUEn=0 USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.00 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 DISABLE_NON_BLOCKING=1 \
			OVERLAP_CPY_BACK=0 >/dev/null
		actualRun 50 File50
		# actualRun 1 File1
		# actualRun 50 File50
		###
		tail -n 1 File50.csv > /tmp/auxFile50.csv
		# tail -n 1 File1.csv  > /tmp/auxFile1.csv
		# tail -n 1 File50.csv > /tmp/auxFile50.csv

		# cat /tmp/auxFile1.csv >> File1.csv
		# cat /tmp/auxFile1.csv >> File1.csv
		# cat /tmp/auxFile1.csv >> File1.csv
		# cat /tmp/auxFile1.csv >> File1.csv
		###
		cat /tmp/auxFile50.csv >> File50.csv
		cat /tmp/auxFile50.csv >> File50.csv
		cat /tmp/auxFile50.csv >> File50.csv
		cat /tmp/auxFile50.csv >> File50.csv
		cat /tmp/auxFile50.csv >> File50.csv
		###
		# cat /tmp/auxFile50.csv >> File50.csv
		# cat /tmp/auxFile50.csv >> File50.csv
		# cat /tmp/auxFile50.csv >> File50.csv
		# cat /tmp/auxFile50.csv >> File50.csv
		###
		###

		mv File50.csv ${1}_50_s${s}

		# mv File1.csv ${1}_1_s${s}
		# mv File50.csv ${1}_50_s${s}
	done
}

function doRunLargeDTST_VERS {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		make clean ; make \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.001 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=0 >/dev/null
		actualRun 50 File50
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.20 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=0 >/dev/null
		actualRun 50 File50
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.50 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=0 >/dev/null
		actualRun 50 File50
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.80 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=0 >/dev/null
		actualRun 50 File50
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.90 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=0 >/dev/null
		actualRun 50 File50
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=1.00 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=0 >/dev/null
		actualRun 50 File50
		# actualRun 1 File1
		# actualRun 50 File50
		###
		mv File50.csv ${1}_50_s${s}
		# mv File1.csv ${1}_1_s${s}
		# mv File50.csv ${1}_50_s${s}
	done
}

function doRunLargeDTST_VERS_OVER {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		make clean ; make \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.001 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 >/dev/null
		actualRun 50 File50
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.20 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 >/dev/null
		actualRun 50 File50
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.50 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 >/dev/null
		actualRun 50 File50
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.80 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 >/dev/null
		actualRun 50 File50
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.90 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 >/dev/null
		actualRun 50 File50
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=1.00 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 >/dev/null
		actualRun 50 File50
		# actualRun 1 File1
		# actualRun 50 File50
		###
		mv File50.csv ${1}_50_s${s}
		# mv File1.csv ${1}_1_s${s}
		# mv File50.csv ${1}_50_s${s}
	done
}

function doRunLargeDTST_VERS_OVER_NE {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		make clean ; make \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.001 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 DISABLE_EARLY_VALIDATION=1 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 >/dev/null
		actualRun 50 File50
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.20 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 DISABLE_EARLY_VALIDATION=1 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 >/dev/null
		actualRun 50 File50
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.50 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 DISABLE_EARLY_VALIDATION=1 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 >/dev/null
		actualRun 50 File50
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.80 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 DISABLE_EARLY_VALIDATION=1 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 >/dev/null
		actualRun 50 File50
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.90 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 DISABLE_EARLY_VALIDATION=1 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 >/dev/null
		actualRun 50 File50
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=1.00 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 DISABLE_EARLY_VALIDATION=1 \
			DISABLE_NON_BLOCKING=0 OVERLAP_CPY_BACK=1 >/dev/null
		actualRun 50 File50
		# actualRun 1 File1
		# actualRun 50 File50
		###
		mv File50.csv ${1}_50_s${s}
		# mv File1.csv ${1}_1_s${s}
		# mv File50.csv ${1}_50_s${s}
	done
}

function doRunLargeDTST_VERS_BLOC {
	# Seq. access, 18 items, prob. write {5..95}, writes 1%
	for s in `seq 1 $SAMPLES`
	do
		make clean ; make \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.001 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=1 OVERLAP_CPY_BACK=0 >/dev/null
		actualRun 50 File50
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.20 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=1 OVERLAP_CPY_BACK=0 >/dev/null
		actualRun 50 File50
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.50 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=1 OVERLAP_CPY_BACK=0 >/dev/null
		actualRun 50 File50
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.80 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=1 OVERLAP_CPY_BACK=0 >/dev/null
		actualRun 50 File50
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=0.90 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=1 OVERLAP_CPY_BACK=0 >/dev/null
		actualRun 50 File50
		# actualRun 1 File1
		# actualRun 50 File50
		###
		rm -f bank *.o src/*.o ; make simple \
			CMP_TYPE=COMPRESSED LOG_TYPE=VERS USE_TSX_IMPL=1 PR_MAX_RWSET_SIZE=20 \
			BANK_PART=${2} GPU_PART=0.55 CPU_PART=0.55 P_INTERSECT=1.00 PROFILE=1 -j 14 \
			BANK_INTRA_CONFL=0.0 BMAP_GRAN_BITS=13 LOG_SIZE=4096 \
			STM_LOG_BUFFER_SIZE=256 \
			DISABLE_NON_BLOCKING=1 OVERLAP_CPY_BACK=0 >/dev/null
		actualRun 50 File50
		# actualRun 1 File1
		# actualRun 50 File50
		###
		mv File50.csv ${1}_50_s${s}
		# mv File1.csv ${1}_1_s${s}
		# mv File50.csv ${1}_50_s${s}
	done
}

############### LARGE
###########################################################################
### 600MB
DATASET=150000000
DURATION=$DURATION_GPU
############### GPU-only
doRunLargeDTST_GPUonly inter_GPUonly_rand_sep 9

############## CPU-only
doRunLargeDTST_CPUonly inter_CPUonly_rand_sep 9

# ############## VERS
# doRunLargeDTST_VERS inter_VERS_rand_sep 9

DURATION=$DURATION_ORG
############## VERS_OVER
doRunLargeDTST_VERS_OVER inter_VERS_OVER_rand_sep 9

############## VERS_OVER
doRunLargeDTST_VERS_OVER_NE inter_VERS_OVER_NE_rand_sep 9

# ############## VERS_BLOC
# doRunLargeDTST_VERS_BLOC inter_VERS_BLOC_rand_sep 9

############## BMAP
# doRunLargeDTST_BMAP inter_BMAP_rand_sep 1
###########################################################################
#
# ############### SMALL
# ###########################################################################
# ### 60MB
# DATASET=15000000
# # ############### GPU-only
# doRunLargeDTST_GPUonly inter_GPUonly_rand_sep_SMALL 9
#
# ############## CPU-only
# doRunLargeDTST_CPUonly inter_CPUonly_rand_sep_SMALL 9
#
# ############## VERS
# doRunLargeDTST_VERS inter_VERS_rand_sep_SMALL 9
#
# ############## VERS_OVER
# doRunLargeDTST_VERS_OVER inter_VERS_OVER_rand_sep_SMALL 9
#
# ############## VERS_BLOC
# doRunLargeDTST_VERS_BLOC inter_VERS_BLOC_rand_sep_SMALL 9

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

mkdir -p array_inter_HTM
mv *_s* array_inter_HTM
