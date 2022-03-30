#!/bin/bash

REMOTE_NODE=pascal
USER=dcastro

if [[ $# -gt 0 ]] ; then
	REMOTE_NODE=$1
fi

CURR_FOLDER=$(pwd)
TARGET_FOLDER=~/Documents/Data/HeTM_multiGPU/ex2_syncCost
REMOTE_FOLDER=/home/dcastro/projs/HeTM_V1/experiments/ex2_syncCost/data
EXPERIMENT_FOLDER=$(date +%Y-%m-%dT%H_%m_%S)
DATA_FOLDER=$TARGET_FOLDER/$EXPERIMENT_FOLDER
PROC_FOLDER=$TARGET_FOLDER/0proc$EXPERIMENT_FOLDER

### Scripts
SCRIPTS=$CURR_FOLDER/../scripts
CONVERT_TO_TSV=$CURR_FOLDER/../aux_files/convertToTSV.sh
CONCAT_DUR_BATCH=$CURR_FOLDER/../aux_files/concat_col_file.sh
AVG_ALL=$CURR_FOLDER/../aux_files/averageAll.sh
NORMALIZE=$CURR_FOLDER/normalize.sh

mkdir -p $DATA_FOLDER

scp $REMOTE_NODE:$REMOTE_FOLDER/* $DATA_FOLDER

cp duration_batch.txt $TARGET_FOLDER
cd $TARGET_FOLDER

### converts to TSV
$CONVERT_TO_TSV
$CONCAT_DUR_BATCH duration_batch.txt
$AVG_ALL $SCRIPTS
$NORMALIZE $SCRIPTS
mkdir 0proc$EXPERIMENT_FOLDER
cp $EXPERIMENT_FOLDER/*.avg $PROC_FOLDER
cp $EXPERIMENT_FOLDER/NORM_* $PROC_FOLDER

### TODO: plot
