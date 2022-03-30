iter=30
filename="BankFinalSize"
filename2=$filename'-NB'
echo $filename2
start=50000
end=1000000
start2=800000
end2=50500000
step=3000000
acct=524288
duration=2000
threads=4
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
flags25="-DCuda_Confl=1 -DCudaConflVal=2 -DDCuda_Confl2=0.60"

BASELINE=0
RUNNC=0
RUNLC=1
RUNMC=1
RUNHC=1

./makeTM.sh
if [ "$BASELINE" == 1 ]; then
	#GEN CPU
	make clean
	nvcc pr-stm.cu -L lib -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c $flagsGPU
	nvcc bank.c -L lib -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c -DGPUEn=0
	make
	for ((  i = $start  ;  i <= $end;  i+=$start  ))
	do
		./bank -n $threads -a $i  -r 0 -w 0 -f  $filename'-CPU-L' -i 10 -d 1000 $runflags
	done
	for ((  i = $start2  ;  i <= $end2;  i+=$step  ))
	do
		make
		./bank -n $threads -a $i  -r 0 -w 0 -f  $filename'-CPU-H' -i 10 -d 1000 $runflags
	done

	make clean
	nvcc pr-stm.cu -L lib -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c $flagsGPU
	nvcc bank.c -L lib -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c -DCPUEn=0
	make
	#GEN GPU
	for ((  i = $start  ;  i <= $end;  i+=$start  ))
	do
		./bank -n $threads -a $i  -r 0 -w 0 -f  $filename'-GPU-L' -i 40 -d 250 $runflags
	done
	for ((  i = $start2  ;  i <= $end2;  i+=$step  ))
	do
		make
		./bank -n $threads -a $i  -r 0 -w 0 -f  $filename'-GPU-H' -i 40 -d 250 $runflags
	done
fi


if [ "$RUNNC" == 1 ]; then
	make clean
	nvcc pr-stm.cu -L lib -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c $flagsGPU
	nvcc bank.c -L lib -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c -DCuda_Confl=0
	make
		for ((  i = $start  ;  i <= $end;  i+=$start  ))
	do
		./bank -n 4 -a $i  -r 0 -w 0 -f $filename2'-L0' -i $iter -d $duration -j
	done
		for ((  i = $start2  ;  i <= $end2;  i+=$step  ))
	do
		./bank -n 4 -a $i  -r 0 -w 0 -f $filename2'-H0' -i $iter -d $duration -j
	done
fi

if [ "$RUNLC" == 1 ]; then
	make clean
	nvcc pr-stm.cu -L lib -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c $flagsGPU
	nvcc bank.c -L lib -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c $flags25
	make
		for ((  i = $start  ;  i <= $end;  i+=$start  ))
	do
		./bank -n 4 -a $i  -r 0 -w 0 -f $filename2'-L25' -i $iter -d $duration 
	done
		for ((  i = $start2  ;  i <= $end2;  i+=$step  ))
	do
		./bank -n 4 -a $i  -r 0 -w 0 -f $filename2'-H25' -i $iter -d $duration 
	done
fi

if [ "$RUNMC" == 1 ]; then
	make clean
	nvcc pr-stm.cu -L lib -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c $flagsGPU
	nvcc bank.c -L lib -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c $flags50
	make
		for ((  i = $start  ;  i <= $end;  i+=$start  ))
	do
		./bank -n 4 -a $i  -r 0 -w 0 -f $filename2'-L50' -i $iter -d $duration 
	done
		for ((  i = $start2  ;  i <= $end2;  i+=$step  ))
	do
		./bank -n 4 -a $i  -r 0 -w 0 -f $filename2'-H50' -i $iter -d $duration 
	done
fi

if [ "$RUNHC" == 1 ]; then
	make clean
	nvcc pr-stm.cu -L lib -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c $flagsGPU
	nvcc bank.c -L lib -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c -DDIFFERENT_POOLS=0
	make
		for ((  i = $start  ;  i <= $end;  i+=$start  ))
	do
		./bank -n 4 -a $i  -r 0 -w 0 -f $filename2'-L100' -i $iter -d $duration 
	done
		for ((  i = $start2  ;  i <= $end2;  i+=$step  ))
	do
		./bank -n 4 -a $i  -r 0 -w 0 -f $filename2'-H100' -i $iter -d $duration 
	done
fi


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