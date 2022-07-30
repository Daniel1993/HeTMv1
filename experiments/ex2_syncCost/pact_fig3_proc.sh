#!/bin/bash

REMOTE_NODE=pascal
USER=dcastro

if [[ $# -gt 0 ]] ; then
	REMOTE_NODE=$1
fi

TARGET_FOLDER=/Users/daniel/Documents/Data/HeTM_2022_July_extention/2GPUS/1st_fig_NEW_no_conf_100w_non_condensed
REMOTE_FOLDER=/home/dcastro/projs/HeTM_V1/benches/bank/data/sync_cost_no_conf_1bit
source ../aux_files/vars.sh
mkdir -p $DATA_FOLDER
scp $REMOTE_NODE:$REMOTE_FOLDER/* $DATA_FOLDER

source pact_fig3_proc_loc_mac.sh

