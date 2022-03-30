#!/bin/bash

iter=1
filename="BankTSXLog"
filename2=$filename
echo $filename2
#start=128
#end=1024
#step=64
start=2
end=128
step=4

duration=6000
threads=14
runflags=""
# DFirstx= LC: 1; MC: 5; HC: 6
flagsGPU="-DfirstX=1"
# DLimAcc= LC: 0; MC: 650; HC: 200
flagsCPU="-DDIFFERENT_POOLS=0 -DGPUEn=0 -DCPUEn=1"
SMALL=512
MEDIUM=8192
LARGE=524288

./makeTM.sh

#make clean
#nvcc \
#	-I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin \
#	-lhtm_sgl --compiler-options='-mrtm' -DUSE_TSX_IMPL -DHTM_NO_WR_INST \
#	-L tinystm/lib -lstm -lpthread \
#	-I tinystm/include -I tinystm/src -c $flagsCPU -o bank.o bankCPU.c
#make USE_TSX_IMPL=1
#for ((  i = $start  ;  i <= $end;  i+=$step  ))
#do
#	# ./bank -n $threads -a $i  -u 100 -f $filename2'-Vanilla' -d $duration
#	./bank -n $threads -a $LARGE -u 100 -f $filename2'-Vanilla' -s $i -d $duration
#done

########################################## THREADS
#TODO: threads 56
for ((  i = 1  ;  i <= 56;  i+=1  ))
do
	./bank -n $i -a $SMALL  -u 100 -f $filename2'-Vanilla-thr-small' -d $duration
	./bank -n $i -a $MEDIUM -u 100 -f $filename2'-Vanilla-thr-mediu' -d $duration
	./bank -n $i -a $LARGE  -u 100 -f $filename2'-Vanilla-thr-large' -d $duration
done

#  make clean
#  nvcc \
#  	-I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm' -DUSE_TSX_IMPL \
#  	bankSerl.c -c -L tinystm/lib -lstm -lpthread -I tinystm/include -I tinystm/src -o bank.o $flagsCPU
#  make USE_TSX_IMPL=1
#
#  ./bank -n 1 -a $SMALL  -r 0 -w 0 -f $filename2'-Vanilla-thr-small' -i $iter -t 1 -d $duration -j
#  ./bank -n 1 -a $MEDIUM -r 0 -w 0 -f $filename2'-Vanilla-thr-mediu' -i $iter -t 1 -d $duration -j
#  ./bank -n 1 -a $LARGE  -r 0 -w 0 -f $filename2'-Vanilla-thr-large' -i $iter -t 1 -d $duration -j
########################################## THREADS

make clean
nvcc \
	-I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin \
	-lhtm_sgl --compiler-options='-mrtm' -DUSE_TSX_IMPL -L tiny0/lib -lstm -lpthread \
	-I tiny0/include -I tiny0/src -c $flagsCPU -o bank.o bankCPU.c
make ROOT=tiny0 USE_TSX_IMPL=1

for u in 10 50 90
do
	for ((  i = $start  ;  i <= $end;  i+=$step  ))
	do
		# ./bank -n $threads -a $i  -r 0 -w 0 -f $filename2'-S' -i $iter -d $duration -j
		# ./bank -n $threads -a $i -u 100 -f $filename2'-S' -d $duration
		./bank -n 1 -a $LARGE -u $u -f $filename2'-S'$u -s $i -d $duration # THREADS SET TO 1
	done
done

# make clean
# nvcc bank.c -L tiny1/lib -lstm -lpthread -I tiny1/include -I tiny1/src -c $flagsCPU
# make ROOT=tiny1
# for ((  i = $start  ;  i >= $end;  i-=$step  ))
# do
	# ./bank -n $threads -a $i  -r 0 -w 0 -f $filename2'-B' -i $iter -d $duration -j
# done

make clean
nvcc \
	-I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin \
	-lhtm_sgl --compiler-options='-mrtm' -DUSE_TSX_IMPL -L tiny2/lib -lstm -lpthread \
	-I tiny2/include -I tiny2/src -c $flagsCPU -o bank.o bankCPU.c
make ROOT=tiny2 USE_TSX_IMPL=1
#
for u in 10 50 90
do
	for ((  i = $start  ;  i <= $end;  i+=$step  ))
	do
		# ./bank -n $threads -a $i  -r 0 -w 0 -f $filename2'-V' -i $iter -d $duration -j
		# ./bank -n $threads -a $i -u 100 -f $filename2'-V' -d $duration
		./bank -n 1 -a $LARGE -u $u -f $filename2'-V'$u -s $i -d $duration
	done
done

# make clean
# nvcc bank.c -L lib -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c -DDEFAULT_TransEachThread=250 -DGPUEn=0
# make

# for ((  i = $start  ;  i <= $end;  i+=1  ))
# do
	# ./bank -n $i -a $acct  -r 0 -w 0 -f  $filename'LC' -i $iter -d $duration $runflags
	# ./bank -n $i -a 650  -r 0 -w 0 -f  $filename'MC' -i $iter -d $duration
	# ./bank -n $i -a 200  -r 0 -w 0 -f  $filename'HC' -i $iter -d $duration
# done

# make clean
# nvcc pr-stm.cu -L lib -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c -DDEFAULT_TransEachThread=250
# nvcc bank.c -L lib -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c -DDEFAULT_TransEachThread=250
# make
# for ((  i = $start  ;  i <= $end;  i+=1  ))
# do
	# ./bank -n $i -a $acct  -r 0 -w 0 -f  $filename'StressTest' -i $iter -d $duration $runflags
# done
