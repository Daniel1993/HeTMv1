#!/bin/bash

DIR=projs
NAME="HeTM_V1"
NODE="pascal"

DM=$DIR/$NAME

CMAKE=cmake

command -v $CMAKE >/dev/null 2>&1 || { CMAKE=~/bins/cmake; }
command -v $CMAKE >/dev/null 2>&1 || { echo "cmake not installed. Aborting." >&2; exit 1; }

if [[ $# -gt 0 ]] ; then
	NODE=$1
fi

find . -name ".DS_Store" -delete
find . -name "._*" -delete

ssh $NODE "mkdir -p $DIR/$DM "
rsync -avzP --exclude 'sAnnealing_plots/' . $NODE:$DM

if [[ "${NODE}" == "intel14v1" ]]
then
	echo "replacing cuda_location in $NODE"
	ssh $NODE "cp $DM/cuda_location_intel14v1.mak $DM/cuda_location.mak"
fi
