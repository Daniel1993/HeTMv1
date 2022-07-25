#!/bin/bash

REMOTE_NODE=pascal
USER=dcastro

if [[ $# -gt 0 ]] ; then
	REMOTE_NODE=$1
fi

TARGET_FOLDER=~/Documents/Data/HeTM_multiGPU/ex2_syncCost
REMOTE_FOLDER=/home/dcastro/projs/HeTM_V1/experiments/ex2_syncCost/data
source ../aux_files/vars.sh
mkdir -p $DATA_FOLDER
scp $REMOTE_NODE:$REMOTE_FOLDER/* $DATA_FOLDER

source pact_fig3_proc_loc_mac.sh

