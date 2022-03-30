#!/bin/bash

FOLDER=$(find . -maxdepth 1 -type d | sort | tail -n 1)
PREC_WRITE_FILE=prec_write.txt

if [[ $# -gt 0 ]] ; then
	PREC_WRITE_FILE=$1
fi

if [[ $# -gt 1 ]] ; then
	FOLDER=$2
fi

for f in $FOLDER/*.tsv
do
	paste -d "	" $f $PREC_WRITE_FILE > $f.tmp
	mv $f.tmp $f
done
