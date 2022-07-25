#!/bin/bash

TARGET_FOLDER=/Users/daniel/Documents/Data/HeTM_2022_July_extention/2GPUS/2nd_plot_inter_conf_new
OLDER_FOLDER=/Users/daniel/Documents/Data/HeTM_2022_July_extention/2GPUS/2nd_plot_inter_conf_old
REMOTE_FOLDER=/home/dcastro/projs/HeTM_V1/experiments/ex2_syncCost/data
source ../aux_files/vars.sh

cp $OLDER_FOLDER/SHeTM_* $TARGET_FOLDER/

### Scripts
SCRIPTS=$CURR_FOLDER/../scripts
CONVERT_TO_TSV=$CURR_FOLDER/../aux_files/convertToTSV.sh
CONCAT_DUR_BATCH=$CURR_FOLDER/../aux_files/concat_col_file.sh
AVG_ALL=$CURR_FOLDER/../aux_files/averageAll.sh
#NORMALIZE=$CURR_FOLDER/normalize.sh

cp conflicts.txt $TARGET_FOLDER
cd $TARGET_FOLDER

### converts to TSV
$CONVERT_TO_TSV .
$CONCAT_DUR_BATCH conflicts.txt .
$AVG_ALL $SCRIPTS .
#$NORMALIZE $SCRIPTS
mkdir -p 0proc$EXPERIMENT_FOLDER
ls
cp *.avg 0proc$EXPERIMENT_FOLDER
#cp $EXPERIMENT_FOLDER/NORM_* $PROC_FOLDER
mkdir -p trash
mv *.tsv *.avg trash

$CURR_FOLDER/pact_fig5_plot_loc_mac.sh 0proc$EXPERIMENT_FOLDER pact_fig5.png
