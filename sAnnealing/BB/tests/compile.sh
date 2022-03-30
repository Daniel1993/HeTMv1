#!/bin/bash

cd .. ; make clean ; make DEBUG=1 libbb.a ; cd - ; gcc -g -o boggus_mat boggus_mat.c -L .. -I .. -l bb -l m
