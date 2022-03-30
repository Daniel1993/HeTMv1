#!/bin/bash

FOLDER=$(ls -d */ | tail -n 1)
SCRIPTS=../scripts

if [[ $# -gt 0 ]] ; then
	SCRIPTS=$1
fi

$SCRIPTS/../aux_files/normalize_CPU.R \
  ${FOLDER}CPUonly_rand_sep_DISABLED.avg \
  ${FOLDER}VERS_rand_sep.avg \
  ${FOLDER}NORM_VERS_inst_rand_sep_TSX.avg

$SCRIPTS/../aux_files/normalize_CPU.R \
  ${FOLDER}CPUonly_rand_sep_DISABLED_STM.avg \
  ${FOLDER}VERS_rand_sep_STM.avg \
  ${FOLDER}NORM_VERS_inst_rand_sep_STM.avg

$SCRIPTS/../aux_files/normalize_CPU.R \
  ${FOLDER}CPUonly_rand_sep_DISABLED.avg \
  ${FOLDER}BMAP_rand_sep.avg \
  ${FOLDER}NORM_BMAP_inst_rand_sep_TSX.avg

$SCRIPTS/../aux_files/normalize_CPU.R \
  ${FOLDER}CPUonly_rand_sep_DISABLED_STM.avg \
  ${FOLDER}BMAP_rand_sep_STM.avg \
  ${FOLDER}NORM_BMAP_inst_rand_sep_STM.avg

$SCRIPTS/../aux_files/normalize_GPU.R \
  ${FOLDER}GPUonly_rand_sep_DISABLED.avg \
  ${FOLDER}GPUonly_rand_sep_BMAP.avg \
  ${FOLDER}GPUonly_rand_sep_BMAP_NO_RS.avg \
  ${FOLDER}GPUonly_rand_sep_BMAP_NO_WS.avg \
  ${FOLDER}NORM_inst_rand_sep_GPU.avg

# receives GPU_no_inst GPU_inst GPU_NO_RS GPU_NO_WS
#$SCRIPTS/../aux_files/normalize_GPU.R \
#  ${FOLDER}GPUonly_1.avg \
#  ${FOLDER}GPUonly_1.avg \
#  ${FOLDER}GPUonly_rand_sep_BMAP_NO_RS_10b.avg \
#  ${FOLDER}GPUonly_rand_sep_BMAP_NO_WS_10b.avg \
#  ${FOLDER}NORM_inst_rand_sep_GPU_10b.avg
