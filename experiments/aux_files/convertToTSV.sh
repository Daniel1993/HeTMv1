#!/bin/bash

FOLDER=$(find . -maxdepth 1 -type d | sort | tail -n 1)

if [[ $# -gt 0 ]] ; then
	FOLDER=$1
fi

for f in ${FOLDER}/*_s*
do
	sed "s/;/	/g" $f | sed 1d > $f.tsv
	echo "convert $f -> $f.tsv"
done
