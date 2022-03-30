#!/bin/bash

CURR_FOLDER=$(pwd)
TARGET_FOLDER=~/Documents/Data/HeTM_multiGPU/prec_writes

EXPERIMENT_FOLDER=$(find $TARGET_FOLDER -maxdepth 1 -type d | sort | tail -n 1)
DATA_FOLDER=$EXPERIMENT_FOLDER

### Scripts
SCRIPTS=$CURR_FOLDER/../scripts
CONVERT_TO_TSV=$CURR_FOLDER/../aux_files/convertToTSV.sh
CONCAT_PREC_WRT=$CURR_FOLDER/../aux_files/concat_col_file.sh
AVG_ALL=$CURR_FOLDER/../aux_files/averageAll.sh
NORMALIZE=$CURR_FOLDER/normalize.sh
THROUGHPUT_PLOT=$CURR_FOLDER/__PLOT__sync_cost_BMAP_multiGPU.gp

cd $TARGET_FOLDER

### converts to TSV
rm $DATA_FOLDER/*.tsv $DATA_FOLDER/*.avg 
$CONVERT_TO_TSV 
$CONCAT_PREC_WRT
$AVG_ALL $SCRIPTS
#$NORMALIZE $SCRIPTS
gnuplot -c $THROUGHPUT_PLOT $DATA_FOLDER multi_gpu_throughput.pdf
echo "Plotted data into $TARGET_FOLDER/multi_gpu_throughput.pdf"
echo "(data folder is $DATA_FOLDER)"
