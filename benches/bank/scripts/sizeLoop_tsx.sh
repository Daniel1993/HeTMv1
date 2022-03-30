#!/bin/bash

iter=10
filename="BankFinalSizeTSX"
filename2=$filename''
echo $filename2
start=30000
end=1 #1000000
step=30000
start2=3276800
end2=65536000
step2=3276800
acct=524288
duration=2500
threads=4
runflags="-T 1000 -j"
cuda_inc="-L /usr/local/cuda/lib64 -lcudart -lcuda -I /usr/local/cuda/include"

arch_deps="-I ~/projs/arch_dep/include"
htm_alg="-I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm'"
use_tsx="$arch_deps $htm_alg -DUSE_TSX_IMPL"

# DFirstx= LC: 1; MC: 4; HC: 6
flagsGPU="--default-stream per-thread -DfirstX=1"
# DLimAcc= LC: 0; MC: 1000; HC: 200
flagsCPUb="-DLimAcc=0"
flagsCPUL="-DLimAcc=0"
flagsCPUM="-DLimAcc=1000"
flagsCPUH="-DLimAcc=200"
# LC: "-DCuda_Confl=1 -DCudaConflVal=10 -DDCuda_Confl2=0.50"
# MC: "-DCuda_Confl=1 -DCudaConflVal=1 -DCuda_Confl2=0.80"
# HC: "-DCuda_Confl=1 -DCudaConflVal=1 -DDCuda_Confl2=0.9995"
flags50="-DDIFFERENT_POOLS=1 -DCuda_Confl=1 -DDCuda_Confl2=0.50"
factor50L=16384
factor50H=1024

# LC: "-DCuda_Confl=1 -DCudaConflVal=5 -DDCuda_Confl2=0.50"
# MC: "-DCuda_Confl=1 -DCudaConflVal=1 -DCuda_Confl2=0.9999"
# HC: "-DCuda_Confl=1 -DCudaConflVal=1 -DDCuda_Confl2=0.99999"
flags25="-DDIFFERENT_POOLS=1 -DCuda_Confl=1 -DDCuda_Confl2=0.60"
factor25L=65536
factor25H=2048

BASELINE=1
SML=1
BIT=1
VER=1
VERNB=1
VERBAL=1
VERINV=1

RUN0=1
RUN25=1
RUN50=1
RUN100=1

tmincludes=${includes[$k]}
logtype='-I tiny2/include -I tiny2/src -lcudart -lcuda'
val=0
enabled[0]=$SML
enabled[1]=$BIT
enabled[2]=$VER
enabled[3]=$VERNB
enabled[4]=$VERBAL
enabled[5]=$VERINV
runtext[0]='-S'
runtext[1]='-B'
runtext[2]='-V'
runtext[3]='-VNB'
runtext[4]='-VBL'
runtext[5]='-VI'
includes[0]='-I tiny0/include -I tiny0/src -L tiny1/lib'
includes[1]='-I tiny1/include -I tiny1/src -L tiny1/lib'
includes[2]='-I tiny2/include -I tiny2/src -L tiny2/lib -UNO_LOCK -USYNC_BALANCING'
includes[3]='-I tiny2/include -I tiny2/src -L tiny2/lib -DNO_LOCK -USYNC_BALANCING'
includes[4]='-I tiny2/include -I tiny2/src -L tiny2/lib -DNO_LOCK -DSYNC_BALANCING'
includes[5]='-I tiny2/include -I tiny2/src -L tiny2/lib -DNO_LOCK'
rootloc[0]='tiny0'
rootloc[1]='tiny1'
rootloc[2]='tiny2'
rootloc[3]='tiny2'
rootloc[4]='tiny2'
rootloc[5]='tiny2'
cpuf[0]=''
cpuf[1]=''
cpuf[2]='-UNO_LOCK -USYNC_BALANCING'
cpuf[3]='-DNO_LOCK -USYNC_BALANCING'
cpuf[4]='-DNO_LOCK -DSYNC_BALANCING'
cpuf[5]='-DNO_LOCK'

./makeTM.sh

if [ "$BASELINE" == 1 ]; then
	#GEN CPU
	make clean
	nvcc \
		-I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm' -DUSE_TSX_IMPL \
		pr-stm.cu -I tiny1/include -I tiny1/src -c $flagsGPU -DgenOnce=1
	nvcc \
		-I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm' -DUSE_TSX_IMPL \
		 bank.c -lstm -lpthread -I tiny1/include -I tiny1/src -c -DCPUEn=1 -DGPUEn=0 $flagsCPU
	make USE_TSX_IMPL=1
	for ((  i = $start  ;  i <= $end;  i+=$step ))
	do
		./bank -n $threads -a $i  -r 0 -w 0 -f  $filename'-CPU-L' -i $iter -d 2000 $runflags
	done
	for ((  i = $start2  ;  i <= $end2;  i+=$step2  ))
	do
		make USE_TSX_IMPL=1
		./bank -n $threads -a $i  -r 0 -w 0 -f  $filename'-CPU-H' -i $iter -d 2000 $runflags
	done
fi


for ((  k = 0  ;  k <= 5;  k++  ))
do

	tmincludes=${includes[$k]}
	logtype=${runtext[$k]}
	val=${enabled[$k]}
	root=${rootloc[$k]}
	flagsCPU=$flagsCPUl${cpuf[$k]}

	filename2=$filename$logtype

	echo 'Running '$k': '$tmincludes

	if [ "$val" == 0  ]; then
		continue
	fi

	if [ "$RUN0" == 1 ]; then
		make clean
		nvcc \
			-I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm' -DUSE_TSX_IMPL \
			pr-stm.cu $tmincludes -c $flagsGPU -DgenOnce=1
		nvcc \
			-I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm' -DUSE_TSX_IMPL \
			bank.c -c -DCPUEn=1 -DGPUEn=1 -lstm -lpthread $tmincludes -DCuda_Confl=0 $flagsCPU
		make USE_TSX_IMPL=1 ROOT=$root

		for ((  i = $start  ;  i <= $end;  i+=$step ))
		do
			./bank -n 4 -a $i  -r 0 -w 0 -f $filename2'-L0' -i $iter -d $duration $runflags
		done
		for ((  i = $start2  ;  i <= $end2;  i+=$step2  ))
		do
			./bank -n 4 -a $i  -r 0 -w 0 -f $filename2'-H0' -i $iter -d $duration $runflags
		done
	fi

done

# make clean
# nvcc bank.c -L lib -lstm -lpthread -I tiny2/include -I tiny2/src -c -DDEFAULT_TransEachThread=250 -DGPUEn=0
# make

# for ((  i = $start  ;  i <= $end;  i+=1  ))
# do
	# ./bank -n $i -a $acct  -r 0 -w 0 -f  $filename'LC' -i $iter -d $duration $runflags
	# ./bank -n $i -a 650  -r 0 -w 0 -f  $filename'MC' -i $iter -d $duration
	# ./bank -n $i -a 200  -r 0 -w 0 -f  $filename'HC' -i $iter -d $duration
# done

# make clean
# nvcc pr-stm.cu -I tiny2/include -I tiny2/src -c -DDEFAULT_TransEachThread=250
# nvcc bank.c -L lib -lstm -lpthread -I tiny2/include -I tiny2/src -c -DDEFAULT_TransEachThread=250
# make
# for ((  i = $start  ;  i <= $end;  i+=1  ))
# do
	# ./bank -n $i -a $acct  -r 0 -w 0 -f  $filename'StressTest' -i $iter -d $duration $runflags
# done
