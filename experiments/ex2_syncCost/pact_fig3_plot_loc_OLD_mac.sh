#!/bin/bash

SOURCE=${BASH_SOURCE[0]}
DIRNAME=$( dirname "$SOURCE" )
gnuplot -c $DIRNAME/__PLOT__sync_cost_OLD_multiGPU.gp ${1} ${2}
