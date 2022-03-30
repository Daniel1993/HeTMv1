#!/bin/bash

./array_prec_write_inst_cost.sh
./array_prec_write_inst_cost_TX2.sh
./array_prec_write_inst_cost_LARGE_READS.sh

./array_batch_duration_4.sh

./array_inter_contention_HTM.sh

cd ../memcd

./scriptTestMemcd_v5.sh
