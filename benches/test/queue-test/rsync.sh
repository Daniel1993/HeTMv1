#!/bin/bash

DIR=projs
NAME="../queue-test"
NODE="pascal"

DM=$DIR/$NAME

find . -name .DS_* -delete

if [[ $# -gt 0 ]] ; then
	NODE=$1
fi

rm $(find . -name "*.DS_*")
rm $(find . -name "._*")
rm $(find . -name "*~")

ssh $NODE "mkdir $DIR ; mkdir $DM "

rsync -avz $NAME $NODE:$DIR
