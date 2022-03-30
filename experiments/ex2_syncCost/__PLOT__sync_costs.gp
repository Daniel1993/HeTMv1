#!/bin/bash

EX1_PATH=./

if [[ $# -gt 0 ]] ; then
	EX1_PATH=$1
fi

echo "source files: $EX1_PATH/__PLOT__paths.gp"
source $EX1_PATH/__PLOT__paths.gp

echo "using files: $NORM_VERS"
echo "using files: $NORM_BMAP"

gnuplot <<EOF
set term postscript color eps enhanced 22 size 9,4
set output sprintf("|ps2pdf -dEPSCrop - __PLOT__inst_cost.pdf")

#set grid ytics
#set grid xtics
#
#set logscale x 2
#set format y '%.2tE%T'
#set format y '%.1t'
set mxtics

set xrange [7:107]
set yrange [0:*]
set grid y

####
load "$EX1_PATH/../aux_files/PLOT_style.gp"
####

set multiplot layout 1, 3 margins 0.066,0.995,0.13,0.98 spacing 0.005,0.02

set ylabel "Throughput (normalized)" font ",28" tc lt 1 offset 2.7,-0.0

set ytics nomirror tc lt 1 font ",20"
#  offset 0.8,0.0 f

#  at graph -0.02,0.12 left top Left  reverse width -4 height -5 spacing 1.4 maxrows 4
set key font ",20" 

set xtics tc lt 2 offset 0.0,0.4
set noxlabel

set yrange [0.3:1.2]
plot \
  '$NORM_VERS' u (\$1):(\$2):(\$2-\$3):(\$2+\$3) notitle         w yerrorbars  ls 1, \
  '$NORM_VERS' u (\$1):(\$2)           title "SHeTM^{VERS} CPU"  w linespoints ls 1, \
  '$NORM_BMAP' u (\$1):(\$2):(\$2-\$3):(\$2+\$3) notitle      w yerrorbars  ls 2, \
  '$NORM_BMAP' u (\$1):(\$2)       title "SHeTM^{BMAP} CPU"   w linespoints ls 2, \

set ytics("" 1.1, "" 1.0, "" 0.9, "" 0.8, "" 0.7, "" 0.6)
set noylabel
#set nokey
set xlabel "Duration batch (ms)" font ",28"  tc lt 2
#offset -18.0,0.6 tc lt 2

plot \
  '$NORM_VERS' u (\$1):(\$4):(\$4-\$5):(\$4+\$5) notitle         w yerrorbars  ls 5, \
  '$NORM_VERS' u (\$1):(\$4)           title "SHeTM^{VERS} GPU"  w linespoints ls 5, \
  '$NORM_BMAP' u (\$1):(\$4):(\$4-\$5):(\$4+\$5) notitle      w yerrorbars  ls 6, \
  '$NORM_BMAP' u (\$1):(\$4)       title "SHeTM^{BMAP} GPU"   w linespoints ls 6, \

set ytics("" 1.1, "" 1.0, "" 0.9, "" 0.8, "" 0.7, "" 0.6)
set noylabel
#set nokey
set noxlabel

plot \
  '$NORM_VERS' u (\$1):(\$6):(\$6-\$7):(\$6+\$7) notitle             w yerrorbars  ls 3, \
  '$NORM_VERS' u (\$1):(\$6)           title "SHeTM^{VERS} CPU+GPU"  w linespoints ls 3, \
  '$NORM_BMAP' u (\$1):(\$6):(\$6-\$7):(\$6+\$7) notitle          w yerrorbars  ls 4, \
  '$NORM_BMAP' u (\$1):(\$6)       title "SHeTM^{BMAP} CPU+GPU"   w linespoints ls 4, \

unset multiplot
