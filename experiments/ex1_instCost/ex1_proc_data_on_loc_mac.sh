#!/bin/bash

TARGET_FOLDER=~/Documents/Data/HeTM_multiGPU/ex2_syncCost
REMOTE_FOLDER=/home/dcastro/projs/HeTM_V1/experiments/ex2_syncCost/data
source ../aux_files/vars.sh

### Scripts
SCRIPTS=$CURR_FOLDER/../scripts
CONVERT_TO_TSV=$CURR_FOLDER/../aux_files/convertToTSV.sh
CONCAT_PREC_WRT=$CURR_FOLDER/../aux_files/concat_col_file.sh
AVG_ALL=$CURR_FOLDER/../aux_files/averageAll.sh
NORMALIZE=$CURR_FOLDER/normalize.sh

mkdir -p $DATA_FOLDER

scp $REMOTE_NODE:$REMOTE_FOLDER/* $DATA_FOLDER

cp prec_write.txt $TARGET_FOLDER
cd $TARGET_FOLDER

### converts to TSV
$CONVERT_TO_TSV
$CONCAT_PREC_WRT
$AVG_ALL $SCRIPTS
$NORMALIZE $SCRIPTS
