#!/bin/bash

FOLDER=$(ls -d */ | tail -n 1)
FOLDER_4R_4W=$(ls -d *4R_4W/ | tail -n 1)
FOLDER_LARGE_READS=$(ls -d *LARGE_READS/ | tail -n 1)

echo "pwd $PWD"
echo "FOLDER $FOLDER $(ls -d */ | tail -n 1)"

NORM_BMAP_TSX=${FOLDER}"NORM_BMAP_inst_rand_sep_TSX.avg"
NORM_BMAP_STM=${FOLDER}"NORM_BMAP_inst_rand_sep_STM.avg"
NORM_VERS_TSX=${FOLDER}"NORM_VERS_inst_rand_sep_TSX.avg"
NORM_VERS_STM=${FOLDER}"NORM_VERS_inst_rand_sep_STM.avg"

NORM_RS_GPU=${FOLDER}"NORM_inst_rand_sep_GPU.avg"
NORM_RS_GPU_10b=${FOLDER}"NORM_inst_rand_sep_GPU_10b.avg"

NORM_4R_4W_RS_TSX=${FOLDER_4R_4W}"NORM_inst_rand_sep_TSX.avg"
NORM_LARGE_READS_RS_TSX=${FOLDER_LARGE_READS}"NORM_inst_rand_sep_TSX.avg"
NORM_4R_4W_RS_STM=${FOLDER_4R_4W}"NORM_inst_rand_sep_STM.avg"
NORM_LARGE_READS_RS_STM=${FOLDER_LARGE_READS}"NORM_inst_rand_sep_STM.avg"

NORM_4R_4W_RS_GPU=${FOLDER_4R_4W}"NORM_inst_rand_sep_GPU.avg"
NORM_4R_4W_RS_GPU_10b=${FOLDER_4R_4W}"NORM_inst_rand_sep_GPU_10b.avg"
NORM_LARGE_READS_RS_GPU=${FOLDER_LARGE_READS}"NORM_inst_rand_sep_GPU.avg"
NORM_LARGE_READS_RS_GPU_10b=${FOLDER_LARGE_READS}"NORM_inst_rand_sep_GPU_10b.avg"

VERS_RS=${FOLDER}"VERS_rand_sep.avg"
BMAP_RS=${FOLDER}"BMAP_rand_sep.avg"

