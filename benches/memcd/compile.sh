#!/bin/bash

# default flags
CMP_TYPE=COMPRESSED
HETM_CPU_EN=1
HETM_GPU_EN=1
LOG_TYPE=BMAP
USE_TSX_IMPL=0
CFG=opt
PRINT_DEB=0
PR_MAX_RWSET_SIZE=400
BANK_PART=1
CPU_PART=0.50
GPU_PART=0.50
P_INTERSECT=0.00
PROFILE=1
BMAP_GRAN_BITS=13
HETM_NB_DEVICES=2
DISABLE_RS=0
BANK_INTRA_CONFL=0
BMAP_ENC_1BIT=0

for var in "$@"
do
	a=$(echo "$var" | awk '(index($0,"=") != 0) {print}')
	if [ -n "${a}" ]
	then
		arg=$(echo "$var" | awk '{split($0,Ip,"=")} END{print Ip[1];}')
		val=$(echo "$var" | awk '{split($0,Ip,"=")} END{print Ip[2];}')
		eval "${arg}=${val}"
	fi
done

if [ $1 = "debug" ]
then
	CFG=debug
	# PRINT_DEB=1
	PRINT_DEB=0
	echo "Compile Debug"
else
	CFG=opt
	PRINT_DEB=0
	echo "Compile Optimized"
fi

make clean
make                                                                \
	CMP_TYPE=$CMP_TYPE                                                \
	LOG_TYPE=$LOG_TYPE                                                \
	HETM_CPU_EN=$HETM_CPU_EN                                          \
	HETM_GPU_EN=$HETM_GPU_EN                                          \
	USE_TSX_IMPL=$USE_TSX_IMPL                                        \
	PR_MAX_RWSET_SIZE=$PR_MAX_RWSET_SIZE                              \
	BANK_PART=$BANK_PART                                              \
	GPU_PART=$GPU_PART                                                \
	CPU_PART=$CPU_PART                                                \
	P_INTERSECT=$P_INTERSECT                                          \
	PROFILE=$PROFILE                                                  \
	BANK_INTRA_CONFL=$BANK_INTRA_CONFL                                \
	LOG_SIZE=4096                                                     \
	STM_LOG_BUFFER_SIZE=256                                           \
	DISABLE_NON_BLOCKING=0                                            \
	OVERLAP_CPY_BACK=0                                                \
	BMAP_ENC_1BIT=$BMAP_ENC_1BIT                                      \
	CFG=$CFG                                                          \
	PRINT_DEB=$PRINT_DEB                                              \
	HETM_NB_DEVICES=$HETM_NB_DEVICES                                  \
	BMAP_GRAN_BITS=$BMAP_GRAN_BITS                                    \
	-j 14                                                             \
	alldeps
	# 
make                                                                \
	CMP_TYPE=$CMP_TYPE                                                \
	LOG_TYPE=$LOG_TYPE                                                \
	HETM_CPU_EN=$HETM_CPU_EN                                          \
	HETM_GPU_EN=$HETM_GPU_EN                                          \
	USE_TSX_IMPL=$USE_TSX_IMPL                                        \
	PR_MAX_RWSET_SIZE=$PR_MAX_RWSET_SIZE                              \
	BANK_PART=$BANK_PART                                              \
	GPU_PART=$GPU_PART                                                \
	CPU_PART=$CPU_PART                                                \
	P_INTERSECT=$P_INTERSECT                                          \
	PROFILE=$PROFILE                                                  \
	BANK_INTRA_CONFL=$BANK_INTRA_CONFL                                \
	LOG_SIZE=4096                                                     \
	STM_LOG_BUFFER_SIZE=256                                           \
	DISABLE_NON_BLOCKING=0                                            \
	BMAP_ENC_1BIT=$BMAP_ENC_1BIT                                      \
	OVERLAP_CPY_BACK=0                                                \
	CFG=$CFG                                                          \
	PRINT_DEB=$PRINT_DEB                                              \
	HETM_NB_DEVICES=$HETM_NB_DEVICES                                  \
	BMAP_GRAN_BITS=$BMAP_GRAN_BITS                                    \
	-j 14

echo " ----------------------------------------- "
echo " $0 flags:                                 "
echo -e "\
	CMP_TYPE=$CMP_TYPE                                                \n\
	LOG_TYPE=$LOG_TYPE                                                \n\
	HETM_CPU_EN=$HETM_CPU_EN                                          \n\
	HETM_GPU_EN=$HETM_GPU_EN                                          \n\
	USE_TSX_IMPL=$USE_TSX_IMPL                                        \n\
	PR_MAX_RWSET_SIZE=$PR_MAX_RWSET_SIZE                              \n\
	BANK_PART=$BANK_PART                                              \n\
	GPU_PART=$GPU_PART                                                \n\
	CPU_PART=$CPU_PART                                                \n\
	P_INTERSECT=$P_INTERSECT                                          \n\
	PROFILE=$PROFILE                                                  \n\
	CFG=$CFG                                                          \n\
	PRINT_DEB=$PRINT_DEB                                              \n\
	BANK_INTRA_CONFL=$BANK_INTRA_CONFL                                \n\
	BMAP_GRAN_BITS=$BMAP_GRAN_BITS  \n\
	HETM_NB_DEVICES=$HETM_NB_DEVICES                                    \
	"
echo " ----------------------------------------- "

