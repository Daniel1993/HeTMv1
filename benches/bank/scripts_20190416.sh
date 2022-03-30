#!/bin/bash

./array_batch_duration_4.sh
./array_batch_duration_4_LARGE_TXS.sh
./array_inter_contention_HTM.sh

cd ../memcd

./scriptTestMemcd_v5.sh
