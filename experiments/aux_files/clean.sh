#!/bin/bash

FOLDER=$(find . -maxdepth 1 -type d | sort | tail -n 1)

if [[ $# -gt 0 ]] ; then
	FOLDER=$1
fi

rm $FOLDER/*.tsv $FOLDER/*.avg 

