#!/bin/bash

PART="0.1 0.5 0.55 0.60 0.65 0.70 0.75 0.80 0.90 0.99"
DATASET="1000000 10000000 100000000"

DEFAULT_BATCH="8"
DEFAULT_GPU_THREADS="128"
DEFAULT_BLOCKS="128"

THREADS="2"
DURATION=10000
SAMPLES=10

rm -f Bank.csv

### TODO: comment this
# sleep 20m

for s in `seq $SAMPLES`
do
	for d in $DATASET
	do
		for p in $PART
		do
			##################################
			# Tiny --- GPU_INV
			make clean ; make HETM_CMP_TYPE=COMPRESSED GPU_PART=$p CPU_PART=$p \
				P_INTERSECT=0.5 CPU_INV=0 USE_TSX_IMPL=0 BANK_PART=0 -j 14
			timeout 20s ./bank -n $THREADS -a $d -d $DURATION -b $DEFAULT_BLOCKS \
			 	-x $DEFAULT_GPU_THREADS -T $DEFAULT_BATCH
			##################################
		done
		mv Bank.csv HeTM_Tiny_GPU_INV_d${d}_s${s}
	done
done
