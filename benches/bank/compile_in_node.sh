#!/bin/bash

DIR="projs/HeTM_V0"
NAME="bank"
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

tar -czf $NAME.tar.gz .

ssh $NODE "mkdir -p $DM "
scp $NAME.tar.gz $NODE:$DM
ssh $NODE "cd $DM ; gunzip $NAME.tar.gz ; tar -xf $NAME.tar ; \
      rm *.tar *.tar.gz ; "

rm $NAME.tar.gz
