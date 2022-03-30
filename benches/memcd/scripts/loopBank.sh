#make
#./bank -n 7 -a 500000 -r 0 -w 0 -i 50 -d 5000


#for ((  i = 500 ;  i <= 5000;  i+=125  ))
#do
#	./bank -n 7 -a 1000000 -r 0 -w 0 -f Bank-0G-0C-0I -i 10 -d $i
	#./bank -n 7 -a 2048 -r 5 -w 0 -f BankGPULog3 -d $i
	#./bank -n 8 -a 2048 -r 5 -w 0 -f BankHC -d $i
	#./bank -n 8 -a 2048 -r 5 -w 0 -f BankHC -d $i
	#./bank -n 8 -a 2048 -j -r 0 -w 0 -f BankLC -d $i
	#./bank -n 8 -a 2048 -j -r 0 -w 0 -f BankLC -d $i
#done

#Combined:
#
#	./bank -n 7 -a 100000 -r 5 -w 0 -f Bank2GPU1s -i 20 -d $i
#
#Nabo:
#
#	./bank -n 7 -a 2048 -j -r 0 -w 0 -f BankLC -d $i
#
#Low contention:
#
#	./bank -n 8 -a 258 -j -r 0 -w 0 -f BankLC -d $i
#
#High contention:
#
#	./bank -n 8 -a 258 -r 5 -w 0 -f BankHC -d $i

iter=20
filename="BankFinalTS"
filename2=""
start=2
end=12
acct=262144
duration=2500
threads=4

runflags="-T 25"
cuda_inc="-L /extra/cuda-7.5/lib64 -lcudart -lcuda -I /extra/cuda-7.5/include"

# DFirstx= LC: 1; MC: 4; HC: 6
flagsGPU="--default-stream per-thread -DfirstX=1"
flagsGPUL="--default-stream per-thread -DfirstX=1 -DgenOnce=1"
flagsGPUM="--default-stream per-thread -DfirstX=2 -DgenOnce=0"
flagsGPUH="--default-stream per-thread -DfirstX=4 -DgenOnce=0"
# DLimAcc= LC: 0; MC: 1000; HC: 200
flagsCPU="-DLimAcc=0"
flagsCPUL="-DLimAcc=0"
flagsCPUM="-DLimAcc=1000"
flagsCPUH="-DLimAcc=200"

# LC: "-DCuda_Confl=1 -DCudaConflVal=5 -DDCuda_Confl2=0.50"
# MC: "-DCuda_Confl=1 -DCudaConflVal=1 -DCuda_Confl2=0.9999"
# HC: "-DCuda_Confl=1 -DCudaConflVal=1 -DDCuda_Confl2=0.99999"
flags5="-DCuda_Confl=1 -DCudaConflVal=16 -DCudaConflVal2=0.99999  -DCudaConflVal3=1"
flags5L="-DCuda_Confl=1 -DCudaConflVal=16 -DCudaConflVal2=0.99999  -DCudaConflVal3=8"
flags5M="-DCuda_Confl=1 -DCudaConflVal=16 -DCudaConflVal2=0.99999  -DCudaConflVal3=60"
flags5H="-DCuda_Confl=1 -DCudaConflVal=16 -DCudaConflVal2=0.99999  -DCudaConflVal3=80"


# LC: "-DCuda_Confl=1 -DCudaConflVal=10 -DDCuda_Confl2=0.50"
# MC: "-DCuda_Confl=1 -DCudaConflVal=1 -DCuda_Confl2=0.80"
# HC: "-DCuda_Confl=1 -DCudaConflVal=1 -DDCuda_Confl2=0.9995"
flags25="-DCuda_Confl=1 -DCudaConflVal=16 -DCudaConflVal2=0.99999  -DCudaConflVal3=64"
flags25L="-DCuda_Confl=1 -DCudaConflVal=16 -DCudaConflVal2=0.99999  -DCudaConflVal3=2"
flags25M="-DCuda_Confl=1 -DCudaConflVal=16 -DCudaConflVal2=0.99999  -DCudaConflVal3=12"
flags25H="-DCuda_Confl=1 -DCudaConflVal=16 -DCudaConflVal2=0.99999  -DCudaConflVal3=12"


flags50="-DCuda_Confl=1 -DCudaConflVal=64 -DCudaConflVal2=0.99999  -DCudaConflVal3=64"
flags50L="-DCuda_Confl=1 -DCudaConflVal=16 -DCudaConflVal2=0.99995 -DCudaConflVal3=2"
flags50M="-DCuda_Confl=1 -DCudaConflVal=16 -DCudaConflVal2=0.99999 -DCudaConflVal3=3"
flags50H="-DCuda_Confl=1 -DCudaConflVal=16 -DCudaConflVal2=0.99999 -DCudaConflVal3=3"

flags100="-DCuda_Confl=1 -DCudaConflVal=16 -DCudaConflVal2=0.9999 -DCudaConflVal3=2"
flags100L="-DCuda_Confl=1 -DCudaConflVal=16 -DCudaConflVal2=0.9999 -DCudaConflVal3=2"
flags100M="-DCuda_Confl=1 -DCudaConflVal=16 -DCudaConflVal2=0.99999 -DCudaConflVal3=1"
flags100H="-DCuda_Confl=1 -DCudaConflVal=16 -DCudaConflVal2=0.99999 -DCudaConflVal3=1"


BASELINE=1

SML=1
BIT=0
VER=1
VERNB=0
VERBAL=0
VERINV=0

RUNLC=1
RUNMC=1
RUNHC=1

RUN0=1
RUN5=0
RUN25=0
RUN50=0
RUN100=0


tmincludes='-I ../tinystm/include -I ../tinystm/src -lstm'
logtype='-I ../tinystm/include -I ../tinystm/src -lstm'
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
runtext[4]='-VBAL'
runtext[5]='-VI'
includes[0]='-I tiny0/include -I tiny0/src'
includes[1]='-I tiny1/include -I tiny1/src'
includes[2]='-I tiny2/include -I tiny2/src -UNO_LOCK -USYNC_BALANCING -UCPU_INV'
includes[3]='-I tiny2/include -I tiny2/src -DNO_LOCK -USYNC_BALANCING -UCPU_INV'
includes[4]='-I tiny2/include -I tiny2/src -DNO_LOCK -DSYNC_BALANCING -UCPU_INV'
includes[5]='-I tiny2/include -I tiny2/src -DNO_LOCK -USYNC_BALANCING -DCPU_INV'
rootloc[0]='tiny0'
rootloc[1]='tiny1'
rootloc[2]='tiny2'
rootloc[3]='tiny2'
rootloc[4]='tiny2'
rootloc[5]='tiny2'

#./makeTM.sh
if [ "$BASELINE" == 1 ]; then
	for ((  j = 1  ;  j <= 3;  j++  ))
	do

		if [ "$j" == 1  ]; then  
			if [ "$RUNLC" == 1  ]; then
				flagsCPU=$flagsCPUL 
				flagsGPU=$flagsGPUL
				filename2=$filename'LC'
				runflags='-j'
			else
				continue
			fi
		fi
		
		if [ "$j" == 2  ]; then  
			if [ "$RUNMC" == 1  ]; then
				flagsCPU=$flagsCPUM 
				flagsGPU=$flagsGPUM
				filename2=$filename'MC'
				runflags=' '
			else
				continue
			fi
		fi
		
		if [ "$j" == 3  ]; then  
			if [ "$RUNHC" == 1  ]; then
				flagsCPU=$flagsCPUH
				flagsGPU=$flagsGPUH
				filename2=$filename'HC'
				runflags=' '
			else
				continue
			fi
		fi	
		
		for ((  i = $start  ;  i <= $end;  i+=2  ))
		do
			make clean
			nvcc pr-stm.cu -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c $flagsGPU -DBANK_NB_TRANSFERS=$i
			nvcc bank.c -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c -DCPUEn=0 $flagsCPU
			make
			./bank -n $threads -a $acct  -r 0 -w 0 -f $filename2'-GPU' -i $iter -d $duration -t $i $runflags 
		done

		# make clean
		# nvcc pr-stm.cu -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c $flagsGPU
		# nvcc bank.c -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c -DGPUEn=0 $flagsCPU
		# make
		
		# for ((  i = $start  ;  i <= $end;  i+=2  ))
		# do
			# ./bank -n $threads -a $acct  -r 0 -w 0 -f $filename2'-CPU' -i $iter -d $duration -t $i $runflags
		# done
		
	done
fi


for ((  k = 0  ;  k <= 5;  k++  ))
do

	tmincludes=${includes[$k]}
	logtype=${runtext[$k]}
	val=${enabled[$k]}
	root=${rootloc[$k]}
	
	echo 'Running '$k': '$tmincludes
	
	if [ "$val" == 0  ]; then
		continue
	fi

	for ((  j = 1  ;  j <= 3;  j++  ))
	do

		if [ "$j" == 1  ]; then  
			if [ "$RUNLC" == 1  ]; then
				flagsCPU=$flagsCPUL 
				flagsGPU=$flagsGPUL
				flags5=$flags5L
				flags25=$flags25L
				flags50=$flags50L
				filename2=$filename'LC'$logtype
				runflags='-j'
			else
				continue
			fi
		fi
		
		if [ "$j" == 2  ]; then  
			if [ "$RUNMC" == 1  ]; then
				flagsCPU=$flagsCPUM 
				flagsGPU=$flagsGPUM
				flags5=$flags5M
				flags25=$flags25M
				flags50=$flags50M
				filename2=$filename'MC'$logtype
				runflags=' '
			else
				continue
			fi
		fi
		
		if [ "$j" == 3  ]; then  
			if [ "$RUNHC" == 1  ]; then
				flagsCPU=$flagsCPUH
				flagsGPU=$flagsGPUH
				flags5=$flags5H
				flags25=$flags25H
				flags50=$flags50H
				filename2=$filename'HC'$logtype
				runflags=' '
			else
				continue
			fi
		fi	
		
	
		#make clean
		#make ROOT=$root
		
		if [ "$RUN0" == 1 ]; then
			for ((  i = $start  ;  i <= $end;  i+=2  ))
			do
				make clean
				nvcc pr-stm.cu $tmincludes -lstm -lpthread -c -DBANK_NB_TRANSFERS=$i $flagsGPU
				nvcc kernel.cu $tmincludes -lstm -lpthread -c -DBANK_NB_TRANSFERS=$i $flagsGPU
				nvcc bank.c $tmincludes -lstm -lpthread -c $flagsCPU -DCuda_Confl=0 
				make ROOT=$root
				./bank -n $threads -a $acct  -r 0 -w 0 -f $filename2'-0I' -i $iter -d $duration -t $i $runflags
			done
		fi

		if [ "$RUN5" == 1 ]; then
			for ((  i = $start  ;  i <= $end;  i+=2  ))
			do
				make clean
				nvcc pr-stm.cu $tmincludes -lstm -lpthread -c -DBANK_NB_TRANSFERS=$i $flagsGPU 
				nvcc bank.c $tmincludes -lstm -lpthread -c $flagsCPU -DCuda_Confl=0 $flags5
				make ROOT=$root
				./bank -n $threads -a $acct  -r 0 -w 0 -f $filename2'-5I' -i $iter -d $duration -t $i $runflags
			done
		fi

		if [ "$RUN25" == 1 ]; then
			for ((  i = $start  ;  i <= $end;  i+=2  ))
			do
				make clean
				nvcc pr-stm.cu $tmincludes -lstm -lpthread -c -DBANK_NB_TRANSFERS=$i $flagsGPU 
				nvcc bank.c $tmincludes -lstm -lpthread -c $flagsCPU -DCuda_Confl=0 $flags25
				make ROOT=$root
				./bank -n $threads -a $acct  -r 0 -w 0  -f $filename2'-25I' -i $iter -d $duration -t $i $runflags
			done
		fi
		
		if [ "$RUN100" == 1 ]; then
			for ((  i = $start  ;  i <= $end;  i+=2  ))
			do
				make clean
				nvcc pr-stm.cu $tmincludes -lstm -lpthread -c -DBANK_NB_TRANSFERS=$i $flagsGPU 
				nvcc bank.c $tmincludes -lstm -lpthread -c $flagsCPU -DDIFFERENT_POOLS=0 $flags50
				make ROOT=$root
				./bank -n $threads -a $acct  -r 0 -w 0 -f $filename2'-50I' -i $iter -d $duration -t $i $runflags
			done
		fi
		
		if [ "$RUN100" == 1 ]; then
			for ((  i = $start  ;  i <= $end;  i+=2  ))
			do
				make clean
				nvcc pr-stm.cu $tmincludes -lstm -lpthread -c -DBANK_NB_TRANSFERS=$i $flagsGPU 
				nvcc bank.c $tmincludes -lstm -lpthread -c $flagsCPU -DDIFFERENT_POOLS=0 $flags100
				make ROOT=$root
				./bank -n $threads -a $acct  -r 0 -w 0 -f $filename2'-100I' -i $iter -d $duration -t $i $runflags
			done
		fi
	done
done


# acct=150

# make clean
# make
# for ((  i = $start  ;  i <= $end;  i+=2  ))
# do
	# ./bank -n 4 -a $acct  -r 0 -w 0 -f $filename'HCL-Tiny' -i $iter -d 5000 -t $i
# done

# acct=650

# make clean
# make
# for ((  i = $start  ;  i <= $end;  i+=2  ))
# do
	# ./bank -n 4 -a $acct  -r 0 -w 0 -f $filename'MCL-Tiny' -i $iter -d 5000 -t $i
# done