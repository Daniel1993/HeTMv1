#!/bin/bash

EX1_PATH=./

if [[ $# -gt 0 ]] ; then
	EX1_PATH=$1
fi

echo "source files: $EX1_PATH/__PLOT__paths.gp"
source $EX1_PATH/__PLOT__paths.gp

echo "using files: $NORM_VERS_TSX"
echo "using files: $NORM_VERS_STM"
echo "using files: $NORM_BMAP_TSX"
echo "using files: $NORM_BMAP_STM"

gnuplot <<EOF
set term postscript color eps enhanced 22 size 7,3.2
set output sprintf("|ps2pdf -dEPSCrop - __PLOT__inst_cost.pdf")

#set grid ytics
#set grid xtics
#
#set logscale x 2
#set format y '%.2tE%T'
#set format y '%.1t'
set mxtics

set xrange [7:93]
set yrange [0:*]
set grid y

####
load "$EX1_PATH/../aux_files/PLOT_style.gp"
####

set multiplot layout 1, 2 margins 0.066,0.995,0.13,0.98 spacing 0.005,0.02

set ylabel "Throughput (normalized)" font ",28" tc lt 1 offset 2.7,-0.0

set ytics nomirror tc lt 1 offset 0.8,0.0 font ",20"

set key font ",20" at graph -0.02,0.12 left top Left reverse width -4 height -5 \
spacing 1.4 maxrows 4

set xtics tc lt 2 offset 0.0,0.4
set noxlabel

set yrange [0.3:1.2]
plot \
  '$NORM_RS_GPU' u (\$1):(\$8):(\$8-\$9):(\$8+\$9) notitle                       w yerrorbars  ls 5, \
  '$NORM_RS_GPU' u (\$1):(\$8)           title "SHeTM^{PR-STM} (large bmp, W1)"  w linespoints ls 5, \
  '$NORM_RS_GPU_10b' u (\$1):(\$8):(\$8-\$9):(\$8+\$9) notitle                   w yerrorbars  ls 6, \
  '$NORM_RS_GPU_10b' u (\$1):(\$8)       title "SHeTM^{PR-STM} (small bmp, W1)"  w linespoints ls 6, \
  # NORM_LARGE_READS_RS_GPU u (\$1):(\$8):(\$8-\$9):(\$8+\$9) notitle     w yerrorbars  ls 7, \
  # NORM_LARGE_READS_RS_GPU u (\$1):(\$8)     title "SHeTM^{PR-STM} (large bmp, W2)"  w linespoints ls 7, \
  # NORM_LARGE_READS_RS_GPU_10b u (\$1):(\$8):(\$8-\$9):(\$8+\$9) notitle w yerrorbars  ls 8, \
  # NORM_LARGE_READS_RS_GPU_10b u (\$1):(\$8) title "SHeTM^{PR-STM} (small bmp, W2)"  w linespoints ls 8, \

set ytics("" 1.1, "" 1.0, "" 0.9, "" 0.8, "" 0.7, "" 0.6)
set noylabel
#set nokey
set xlabel "write transactions (%)" font ",28" offset -18.0,0.6 tc lt 2

plot \
  '$NORM_BMAP_TSX' u (\$1):(\$2)       title "SHeTM^{BMAP TSX} (W1)"  w linespoints ls 1, \
  '$NORM_BMAP_TSX' u (\$1):(\$2):(\$2-\$3):(\$2+\$3) notitle          w yerrorbars  ls 1, \
  '$NORM_VERS_TSX' u (\$1):(\$2)       title "SHeTM^{VERS TSX} (W1)" w linespoints ls 2, \
  '$NORM_VERS_TSX' u (\$1):(\$2):(\$2-\$3):(\$2+\$3) notitle         w yerrorbars  ls 2, \
  '$NORM_VERS_STM' u (\$1):(\$2)       title "SHeTM^{VERS TinySTM} (W1)"  w linespoints ls 9, \
  '$NORM_VERS_STM' u (\$1):(\$2):(\$2-\$3):(\$2+\$3) notitle              w yerrorbars  ls 9, \
  '$NORM_BMAP_STM' u (\$1):(\$2)       title "SHeTM^{BMAP TinySTM} (W1)"  w linespoints ls 8, \
  '$NORM_BMAP_STM' u (\$1):(\$2):(\$2-\$3):(\$2+\$3) notitle              w yerrorbars  ls 8, \
  # NORM_LARGE_READS_RS_TSX u (\$1):(\$2):(\$2-\$3):(\$2+\$3) notitle   w yerrorbars  ls 3, \
  # NORM_LARGE_READS_RS_TSX u (\$1):(\$2) title "SHeTM^{TSX} (W2)"  w linespoints ls 3, \
  # NORM_LARGE_READS_RS_STM u (\$1):(\$2):(\$2-\$3):(\$2+\$3) notitle   w yerrorbars  ls 11, \
  # NORM_LARGE_READS_RS_STM u (\$1):(\$2) title "SHeTM^{TinySTM} (W2)"  w linespoints ls 11, \

unset multiplot
