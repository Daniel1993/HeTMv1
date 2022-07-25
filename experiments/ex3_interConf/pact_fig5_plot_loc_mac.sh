#!/bin/bash

SOURCE=${BASH_SOURCE[0]}
DIRNAME=$( dirname "$SOURCE" )
gnuplot -c $DIRNAME/__PLOT__inter_confl.gp ${1} ${2}
