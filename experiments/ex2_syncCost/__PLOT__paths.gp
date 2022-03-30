#!/bin/bash

FOLDER=$(ls -d */ | tail -n 1)
FOLDER_4R_4W=$(ls -d *4R_4W/ | tail -n 1)
FOLDER_LARGE_READS=$(ls -d *LARGE_READS/ | tail -n 1)

echo "pwd $PWD"
echo "FOLDER $FOLDER $(ls -d */ | tail -n 1)"

NORM_BMAP=${FOLDER}"NORM_BMAP_rand_sep_large_w100"
NORM_VERS=${FOLDER}"NORM_VERS_BLOC_rand_sep_large_w100"

