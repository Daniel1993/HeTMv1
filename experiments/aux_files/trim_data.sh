#!/bin/bash

POINTS_NAME_FILE=./
VALUES=18
MAX_VALUE=18

if [[ $# -gt 0 ]] ; then
	POINTS_NAME_FILE=$1
fi

if [[ $# -gt 1 ]] ; then
	VALUES=$2
fi

if [[ $# -gt 2 ]] ; then
	MAX_VALUE=$3
fi

for f in ${POINTS_NAME_FILE}_s*.tsv
do
  i=2
  for l in $(sed '1d' $f | cut -f$VALUES)
  do
    if (( $(echo "$l > $MAX_VALUE" | bc -l) ))
    then
      echo "$f[$i]: $l"
      echo "   sed -n '${i}p' $f"
    fi
    i=$(echo "$i+1" | bc -l)
  done 
done

