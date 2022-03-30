#!/bin/bash

iter=1
filename="BankFinalSizeTSX"
filename2=$filename'-NB'
echo $filename2
start=5000
end=230000
step1=15000
start2=1000000
end2=100000000
step=5000000
acct=524288
duration=8000
threads=14
runflags=""
# DFirstx= LC: 1; MC: 5; HC: 6
flagsGPU="-DfirstX=1"
# DLimAcc= LC: 0; MC: 650; HC: 200
flagsCPU="-DLimAcc=0"
# LC: "-DCuda_Confl=1 -DCudaConflVal=10 -DDCuda_Confl2=0.50"
# MC: "-DCuda_Confl=1 -DCudaConflVal=1 -DDCuda_Confl2=0.995"
# HC: "-DCuda_Confl=1 -DCudaConflVal=1 -DDCuda_Confl2=0.9995"
flags50="-DCuda_Confl=1 -DCudaConflVal=10 -DDCuda_Confl2=0.40"
# LC: "-DCuda_Confl=1 -DCudaConflVal=5 -DDCuda_Confl2=0.50"
# MC: "-DCuda_Confl=1 -DCudaConflVal=1 -DDCuda_Confl2=0.9995"
# HC: "-DCuda_Confl=1 -DCudaConflVal=1 -DDCuda_Confl2=0.9999"
flags25="-DCuda_Confl=1 -DCudaConflVal=10 -DDCuda_Confl2=0.999"

samples=1
timeout_max="12s"

BASELINE=1
RUNNC=1
RUNLC=1
RUNMC=1
RUNHC=1

./makeTM.sh
if [ "$BASELINE" == 1 ]; then
	#GEN CPU
	make clean
	nvcc \
		-I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin \
		-lhtm_sgl --compiler-options='-mrtm' -DUSE_TSX_IMPL pr-stm.cu -L lib -lstm \
		-lpthread -I ../tinystm/include -I ../tinystm/src -c $flagsGPU
	nvcc \
		-I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin \
		-lhtm_sgl --compiler-options='-mrtm' -DUSE_TSX_IMPL bank.c -L lib -lstm \
		-lpthread -I ../tinystm/include -I ../tinystm/src -c -DGPUEn=0
	make USE_TSX_IMPL=1

	for j in `seq $samples`
	do
		for ((  i = $start  ;  i <= $end;  i+=$step1  ))
		do
			timeout $timeout_max ./bank -n $threads -a $i  -r 0 -w 0 -f  $filename'-CPU-L_'$j -i 1 -t 1 -d 2000 $runflags
		done
		for ((  i = $start2  ;  i <= $end2;  i+=$step  ))
		do
			make USE_TSX_IMPL=1
			timeout $timeout_max ./bank -n $threads -a $i  -r 0 -w 0 -f  $filename'-CPU-H_'$j -i 1 -t 1 -d 2000 $runflags
		done
	done

	make clean
	nvcc \
		-I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin \
		-lhtm_sgl --compiler-options='-mrtm' -DUSE_TSX_IMPL pr-stm.cu -g -G -L lib -lstm -lpthread \
		-I ../tinystm/include -I ../tinystm/src -c $flagsGPU
	nvcc \
		-I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin \
		-lhtm_sgl --compiler-options='-mrtm' -DUSE_TSX_IMPL bank.c -g -G -L lib -lstm -lpthread \
		-I ../tinystm/include -I ../tinystm/src -c -DCPUEn=0
	make USE_TSX_IMPL=1
	#GEN GPU
	for j in `seq $samples`
	do
		for ((  i = $start  ;  i <= $end;  i+=$step1  ))
		do
			timeout $timeout_max  ./bank -n $threads -a $i  -r 0 -w 0 -f  $filename'-GPU-L_'$j -i 1 -t 1 -d 2000 $runflags
		done
		for ((  i = $start2  ;  i <= $end2;  i+=$step  ))
		do
			make USE_TSX_IMPL=1
			timeout $timeout_max ./bank -n $threads -a $i  -r 0 -w 0 -f  $filename'-GPU-H_'$j -i 1 -t 1 -d 2000 $runflags
		done
	done
fi


if [ "$RUNNC" == 1 ]; then

	### VERSION
	make clean
	nvcc \
		-I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin \
		-lhtm_sgl --compiler-options='-mrtm' -DUSE_TSX_IMPL pr-stm.cu -L lib -lstm \
		-lpthread -I tiny2/include -I tiny2/src -c $flagsGPU
	nvcc \
		-I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin \
		-lhtm_sgl --compiler-options='-mrtm' -DUSE_TSX_IMPL  bank.c -L lib -lstm \
		-lpthread -I tiny2/include -I tiny2/src -c -DCuda_Confl=0
	make USE_TSX_IMPL=1 ROOT=tiny2
	for j in `seq $samples`
	do
		for ((  i = $start  ;  i <= $end;  i+=$step1  ))
		do
			timeout $timeout_max ./bank -n $threads -a $i  -r 0 -w 0 -f $filename2'-V-L0_'$j -i $iter -t 1 -d $duration -j
		done
		for ((  i = $start2  ;  i <= $end2;  i+=$step  ))
		do
			timeout $timeout_max ./bank -n $threads -a $i  -r 0 -w 0 -f $filename2'-V-H0_'$j -i $iter -t 1 -d $duration -j
		done
	done

	### ADDRESS
	make clean
	nvcc \
		-I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin \
		-lhtm_sgl --compiler-options='-mrtm' -DUSE_TSX_IMPL pr-stm.cu -L lib -lstm \
		-lpthread -I tiny0/include -I tiny0/src -c $flagsGPU
	nvcc \
		-I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin \
		-lhtm_sgl --compiler-options='-mrtm' -DUSE_TSX_IMPL bank.c -L lib -lstm \
		-lpthread -I tiny0/include -I tiny0/src -c -DCuda_Confl=0
	make USE_TSX_IMPL=1 ROOT=tiny0
	for j in `seq $samples`
	do
		for ((  i = $start  ;  i <= $end;  i+=$step1  ))
		do
			timeout $timeout_max ./bank -n $threads -a $i  -r 0 -w 0 -f $filename2'-S-L0_'$j -i $iter -t 1 -d $duration -j
		done
		for ((  i = $start2  ;  i <= $end2;  i+=$step  ))
		do
			timeout $timeout_max ./bank -n $threads -a $i  -r 0 -w 0 -f $filename2'-S-H0_'$j -i $iter -t 1 -d $duration -j
		done
	done
fi

############# LOW, MEDIUM, HIGH contention --> always abort GPU
# if [ "$RUNLC" == 1 ]; then
# 	make clean
# 	nvcc \
# 		-I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin \
# 		-lhtm_sgl --compiler-options='-mrtm' -DUSE_TSX_IMPL pr-stm.cu -L lib -lstm \
# 		-lpthread -I ../tinystm/include -I ../tinystm/src -c $flagsGPU
# 	nvcc \
# 		-I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin \
# 		-lhtm_sgl --compiler-options='-mrtm' -DUSE_TSX_IMPL bank.c -L lib -lstm \
# 		-lpthread -I ../tinystm/include -I ../tinystm/src -c $flags25
# 	make USE_TSX_IMPL=1
# 	for j in `seq $samples`
# 	do
# 		for ((  i = $start  ;  i <= $end;  i+=$start  ))
# 		do
# 			./bank -n $threads -a $i  -r 0 -w 0 -f $filename2'-L25_'$j -i $iter -t 1 -d $duration
# 		done
# 			for ((  i = $start2  ;  i <= $end2;  i+=$step  ))
# 		do
# 			./bank -n $threads -a $i  -r 0 -w 0 -f $filename2'-H25_'$j -i $iter -t 1 -d $duration
# 		done
# 	done
# fi
#
# if [ "$RUNMC" == 1 ]; then
# 	make clean
# 	nvcc \
# 		-I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin \
# 		-lhtm_sgl --compiler-options='-mrtm' -DUSE_TSX_IMPL pr-stm.cu -L lib -lstm \
# 		-lpthread -I ../tinystm/include -I ../tinystm/src -c $flagsGPU
# 	nvcc \
# 		-I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin \
# 		-lhtm_sgl --compiler-options='-mrtm' -DUSE_TSX_IMPL bank.c -L lib -lstm \
# 		-lpthread -I ../tinystm/include -I ../tinystm/src -c $flags50
# 	make USE_TSX_IMPL=1
# 	for j in `seq $samples`
# 	do
# 		for ((  i = $start  ;  i <= $end;  i+=$start  ))
# 		do
# 			./bank -n $threads -a $i  -r 0 -w 0 -f $filename2'-L50_'$j -i $iter -t 1 -d $duration
# 		done
# 			for ((  i = $start2  ;  i <= $end2;  i+=$step  ))
# 		do
# 			./bank -n $threads -a $i  -r 0 -w 0 -f $filename2'-H50_'$j -i $iter -t 1 -d $duration
# 		done
# 	done
# fi
#
# if [ "$RUNHC" == 1 ]; then
# 	make clean
# 	nvcc \
# 		-I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin \
# 		-lhtm_sgl --compiler-options='-mrtm' -DUSE_TSX_IMPL pr-stm.cu -L lib -lstm \
# 		-lpthread -I ../tinystm/include -I ../tinystm/src -c $flagsGPU
# 	nvcc \
# 		-I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin \
# 		-lhtm_sgl --compiler-options='-mrtm' -DUSE_TSX_IMPL bank.c -L lib -lstm \
# 		-lpthread -I ../tinystm/include -I ../tinystm/src -c -DDIFFERENT_POOLS=0
# 	make USE_TSX_IMPL=1
# 	for j in `seq $samples`
# 	do
# 		for ((  i = $start  ;  i <= $end;  i+=$start  ))
# 		do
# 			./bank -n $threads -a $i  -r 0 -w 0 -f $filename2'-L100_'$j -i $iter -t 1 -d $duration
# 		done
# 			for ((  i = $start2  ;  i <= $end2;  i+=$step  ))
# 		do
# 			./bank -n $threads -a $i  -r 0 -w 0 -f $filename2'-H100_'$j -i $iter -t 1 -d $duration
# 		done
# 	done
# fi


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
