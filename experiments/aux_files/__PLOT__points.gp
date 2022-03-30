#!/bin/bash

### assumes the files are _s*.tsv
POINTS_NAME_FILE=./
X_AXIS=1
Y_AXIS=2
RAND_OFFSET=3

if [[ $# -gt 0 ]] ; then
	POINTS_NAME_FILE=$1
fi

if [[ $# -gt 1 ]] ; then
	X_AXIS=$2
fi

if [[ $# -gt 2 ]] ; then
	Y_AXIS=$3
fi

if [[ $# -gt 3 ]] ; then
	RAND_OFFSET=$4
fi

NUMBER_OF_FILES=$(ls ${POINTS_NAME_FILE}_s*.tsv | wc -l)
PDF_NAME=$(echo "${POINTS_NAME_FILE}" | xargs -I{} basename {})

gnuplot <<EOF
set term postscript color eps noenhanced 22 size 7,3.2
set output sprintf("|ps2pdf -dEPSCrop - $PDF_NAME.pdf")

#set grid ytics
#set grid xtics
#
#set logscale x 2
#set format y '%.2tE%T'
#set format y '%.1t'
set mxtics

set grid y
set grid x

set multiplot layout 1, 1
#margins 0.1,0.995,0.2,0.98 spacing 0.005,0.02

firstrow = system('head -1 '.'$POINTS_NAME_FILE'.'_s'.'1'.'.tsv')
set xlabel word(firstrow, $X_AXIS)
set ylabel word(firstrow, $Y_AXIS)

# set ylabel "Throughput (normalized)" font ",28" tc lt 1 offset 2.7,-0.0
# set xlabel "write transactions (%)" font ",28" offset -18.0,0.6 tc lt 2

set ytics nomirror tc lt 1 offset 0.8,0.0 font ",20"

set key font ",20" at graph -0.02,0.12 left top Left reverse width -4 height -5 \
spacing 1.4 maxrows 4

# set xtics tc lt 2 offset 0.0,0.4

# label --> :(sprintf('%d', \$$Y_AXIS))
plot \
  for [i=1:$NUMBER_OF_FILES] '$POINTS_NAME_FILE'.'_s'.i.'.tsv' \
    u (\$$X_AXIS + (rand(0) - 0.5) * $RAND_OFFSET):$Y_AXIS \
    with point pointtype 9 ps 1 lc rgb "blue" notitle

unset multiplot


