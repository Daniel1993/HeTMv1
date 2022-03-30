#!/bin/bash

PROG=./bank
THRS="1 2 4 8 16 32 64 128 256 512"
BLKS="1 2 4 8 16 32 64 128 256 512"

rm -f stats.txt

b=128
for t in $THRS
do
  $PROG $b $t
done

mv stats.txt 128_blocks_old.txt

t=32
for b in $BLKS
do
  $PROG $b $t
done

mv stats.txt 32_threads_old.txt
