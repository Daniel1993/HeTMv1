#!/bin/bash

# This starts are bank/scripts
#cd .. # goes to bank folder

iter=1
filename_tsx="BankTSX"
filename_tiny="BanksTiny"

GPU_PART="0.9 0.6"
CPU_PART="0.9 0.6"
P_INTERSECT="0.001 0.002 0.005 0.01 0.02 0.05"
DATASET="131072 1048576 16777216" # 90 000 000 is the max for my home machine
THREADS=14
GPU_THREADS=512
GPU_BLOCKS=512
GPU_TXS=2
DURATION=10000
SAMPLES=5
#./makeTM.sh

rm -f Bank.csv

for s in `seq $SAMPLES`
do
	for d in $DATASET
	do
		for g in $GPU_PART
		do
			for c in $CPU_PART
			do
				for i in $P_INTERSECT
				do
					# TSX
					make clean ; make GPU_PART=$g CPU_PART=$c P_INTERSECT=$i BANK_PART=1 USE_TSX_IMPL=1 -j 14
					./bank -n $THREADS -a $d -d $DURATION -x $GPU_THREADS -b $GPU_BLOCKS -T $GPU_TXS
				done
				mv Bank.csv "TSX_${d}Acc_${g}GPU_${c}CPU_s${s}"

				for i in $P_INTERSECT
				do
					# Tiny
					make clean ; make GPU_PART=$g CPU_PART=$c P_INTERSECT=$i BANK_PART=1 USE_TSX_IMPL=0 -j 14
					./bank -n $THREADS -a $d -d $DURATION -x $GPU_THREADS -b $GPU_BLOCKS -T $GPU_TXS
				done
				mv Bank.csv "Tiny_${d}Acc_${g}GPU_${c}CPU_s${s}"
			done
		done
	done
done
