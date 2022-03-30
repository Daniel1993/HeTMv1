#!/bin/bash

FOLDER=$(ls -d */ | tail -n 1)
SCRIPTS=../scripts

if [[ $# -gt 0 ]] ; then
	SCRIPTS=$1
fi

$SCRIPTS/../aux_files/normalize_theo.R \
  ${FOLDER}CPUonly_rand_sep_DISABLED_large_w100.avg \
  ${FOLDER}GPUonly_rand_sep_DISABLED_large_w100.avg \
  ${FOLDER}BMAP_rand_sep_1GPU_w100.avg \
  "X.45.DURATION_BATCH" \
  ${FOLDER}NORM_BMAP_rand_sep_1GPU_w100

$SCRIPTS/../aux_files/normalize_theo.R \
  ${FOLDER}CPUonly_rand_sep_DISABLED_large_w100.avg \
  ${FOLDER}GPUonly_rand_sep_DISABLED_large_w100.avg \
  ${FOLDER}BMAP_rand_sep_2GPU_w100.avg \
  "X.45.DURATION_BATCH" \
  ${FOLDER}NORM_BMAP_rand_sep_2GPU_w100
