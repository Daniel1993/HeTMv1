#!/bin/bash

REMOTE_NODE=pascal
USER=dcastro

if [[ $# -gt 0 ]] ; then
	REMOTE_NODE=$1
fi

CURR_FOLDER=$(pwd)
TARGET_FOLDER=~/Documents/Data/HeTM_multiGPU/prec_writes
REMOTE_FOLDER=/home/dcastro/projs/HeTM_V1/experiments/v2_multiGPU_no_confl/data
EXPERIMENT_FOLDER=$(date +%Y-%m-%dT%H_%m_%S)
DATA_FOLDER=$TARGET_FOLDER/$EXPERIMENT_FOLDER

### Scripts
SCRIPTS=$CURR_FOLDER/../scripts
CONVERT_TO_TSV=$CURR_FOLDER/../aux_files/convertToTSV.sh
CONCAT_PREC_WRT=$CURR_FOLDER/../aux_files/concat_col_file.sh
AVG_ALL=$CURR_FOLDER/../aux_files/averageAll.sh
NORMALIZE=$CURR_FOLDER/normalize.sh
THROUGHPUT_PLOT=$CURR_FOLDER/__PLOT__sync_cost_BMAP_multiGPU.gp

mkdir -p $DATA_FOLDER

scp $REMOTE_NODE:$REMOTE_FOLDER/* $DATA_FOLDER

cp prec_write.txt $TARGET_FOLDER

bash $CURR_FOLDER/expr_prec_writes_plot.sh
