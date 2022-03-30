#!/bin/bash

DIR=projs
NAME="fvs_solver"
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

