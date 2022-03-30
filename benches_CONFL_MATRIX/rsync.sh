#!/bin/bash

### enter the IP or name of the node here (as defined in .ssh/config)
NODE=pascal
DM="~/projs/stamp_confl_matrix"

if [[ $# -gt 0 ]] ; then
	NODE=$1
fi

find . -name ".DS_Store" -delete
find . -name "._*" -delete

ssh $NODE "mkdir -p $DM ; echo \"$DM\" "
rsync -avz . $NODE:$DM

