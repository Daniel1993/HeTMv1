#!/bin/bash

DIR=projs
NAME="htm_alg"
NODE="node30"

DM=$DIR/$NAME

if [[ $# -gt 0 ]] ; then
	NODE=$1
fi

ssh $NODE "mkdir $DIR ; mkdir $DM "

rsync -avz . $NODE:$DM

ssh $NODE "cd $DM ; ./compile.sh"
