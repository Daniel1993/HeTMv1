iter=20
filename="BankFinal_TSX_W"
filename2=$filename
start=50
end=400
step=50
small_array=( 1 5 10 25)
small_array2=( 1 )
bal_array=(1 5 10 15 20)
bal_array2=(1 10 25 50)
acct=262144
duration=5000
threads=14
runflags=""
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
VERBALLOOP=1

SML=1
BIT=0
VER=1
VERNB=1
VERBAL=0
VERINV=1

RUNLC=0
RUNMC=1
RUNHC=1

RUN0=1
RUN5=1
RUN25=1
RUN50=1
RUN100=0

tsx_flags=""

tmincludes='-I ../tinystm/include -I ../tinystm/src -lstm '\
logtype='-I ../tinystm/include -I ../tinystm/src -lstm -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options="-mrtm" '
val=0
enabled[0]=$SML
enabled[1]=$BIT
enabled[2]=$VER
enabled[3]=$VERNB
enabled[4]=$VERBAL
enabled[5]=$VERINV
runtext[0]='-SZ'
runtext[1]='-B'
runtext[2]='-V'
runtext[3]='-VNB'
runtext[4]='-VBAL'
runtext[5]='-VI'
includes[0]='-I tiny0/include -I tiny0/src -UCPU_INV -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options="-mrtm"'
includes[1]='-I tiny1/include -I tiny1/src -UCPU_INV -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options="-mrtm"'
includes[2]='-I tiny2/include -I tiny2/src -UNO_LOCK -USYNC_BALANCING -UCPU_INV -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options="-mrtm"'
includes[3]='-I tiny2/include -I tiny2/src -DNO_LOCK -USYNC_BALANCING -UCPU_INV -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options="-mrtm"'
includes[4]='-I tiny2/include -I tiny2/src -DNO_LOCK -DSYNC_BALANCING -UCPU_INV -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options="-mrtm"'
includes[5]='-I tiny2/include -I tiny2/src -DNO_LOCK -USYNC_BALANCING -DCPU_INV -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options="-mrtm"'
rootloc[0]='tiny0'
rootloc[1]='tiny1'
rootloc[2]='tiny2'
rootloc[3]='tiny2'
rootloc[4]='tiny2'
rootloc[5]='tiny2'


# ./makeTM.sh

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

		#GEN CPU
		for i in "${small_array2[@]}"
		do
			make clean
			nvcc pr-stm.cu -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c -DDEFAULT_TransEachThread=$i $flagsGPU  -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm'
			nvcc bank.c -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c -DDEFAULT_TransEachThread=$i -DGPUEn=0 $flagsCPU  -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm'
			make USE_TSX_IMPL=1
			./bank -n $threads -a $acct  -r 0 -w 0 -f  $filename2'-CPU' -i $iter -d 2000 $runflags
		done
		for ((  i = $start  ;  i <= $end;  i+=$step  ))
		do
			make clean
			nvcc  -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm'  pr-stm.cu -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c -DDEFAULT_TransEachThread=$i $flagsGPU
			nvcc  -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm'  bank.c -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c -DDEFAULT_TransEachThread=$i -DGPUEn=0 $flagsCPU
			make
			./bank -n $threads -a $acct  -r 0 -w 0 -f  $filename2'-CPU' -i $iter -d 2000 $runflags
		done

		make clean
		nvcc  -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm'  pr-stm.cu $tmincludes -lpthread -c $flagsGPU
		nvcc  -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm'  bank.c $tmincludes -lpthread -c -DCuda_Confl=0 $flagsCPU $flagsGPU -DCPUEn=0
		make ROOT=tiny2

		#GEN GPU
		for i in "${small_array[@]}"
		do
			./bank -n $threads -a $acct  -r 0 -w 0 -f  $filename2'-GPU' -i $iter -d 2000 $runflags -T $i
		done
		for ((  i = $start  ;  i <= $end;  i+=$step  ))
		do
			./bank -n $threads -a $acct  -r 0 -w 0 -f  $filename2'-GPU' -i $iter -d 2000 $runflags -T $i
		done
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
				flags100=$flags100L
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
				flags100=$flags100M
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
				flags100=$flags100H
				filename2=$filename'HC'$logtype
				runflags=' '
			else
				continue
			fi
		fi

		make clean
		nvcc  -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm'  pr-stm.cu $tmincludes -lstm -lpthread -c $flagsGPU
		nvcc  -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm'  bank.c $tmincludes -lstm -lpthread -c -DCuda_Confl=0 $flagsCPU
		make ROOT=$root USE_TSX_IMPL=1

		if [ "$RUN0" == 1 ]; then
			#GEN NC
			rm -f  bank bank.o
			nvcc  -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm'  bank.c $tmincludes -lstm -lpthread -c -DCuda_Confl=0 $flagsCPU
			make ROOT=$root USE_TSX_IMPL=1

			for i in "${small_array[@]}"
			do
				./bank -n $threads -a $acct  -r 0 -w 0 -f  $filename2'-0I' -i $iter -d $duration $runflags -T $i
			done
			for ((  i = $start  ;  i <= $end;  i+=$step  ))
			do
				./bank -n $threads -a $acct  -r 0 -w 0 -f  $filename2'-0I' -i $iter -d $duration $runflags -T $i
			done
		fi

		if [ "$RUN5" == 1 ]; then
			#GEN LC
			rm -f  bank bank.o
			nvcc  -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm'  bank.c $tmincludes -lstm -lpthread -c $flagsCPU $flags5
			make ROOT=$root USE_TSX_IMPL=1

			for i in "${small_array[@]}"
			do
				./bank -n $threads -a $acct  -r 0 -w 0 -f  $filename2'-5I' -i $iter -d $duration $runflags -T $i
			done
			for ((  i = $start  ;  i <= $end;  i+=$step  ))
			do
				./bank -n $threads -a $acct  -r 0 -w 0 -f  $filename2'-5I' -i $iter -d $duration $runflags -T $i
			done
		fi

		if [ "$RUN25" == 1 ]; then
			#GEN MC

			rm -f  bank bank.o
			nvcc  -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm'  bank.c $tmincludes -lstm -lpthread -c $flagsCPU $flags25
			make ROOT=$root USE_TSX_IMPL=1

			for i in "${small_array[@]}"
			do
				./bank -n $threads -a $acct  -r 0 -w 0 -f  $filename2'-25I' -i $iter -d $duration $runflags -T $i
			done
			for ((  i = $start  ;  i <= $end;  i+=$step  ))
			do
				./bank -n $threads -a $acct  -r 0 -w 0 -f  $filename2'-25I' -i $iter -d $duration $runflags -T $i
			done
		fi

		if [ "$RUN50" == 1 ]; then
			#GEN HC

			rm -f  bank bank.o
			nvcc  -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm'  bank.c $tmincludes -lstm -lpthread -c $flags50 $flagsCPU
			make ROOT=$root USE_TSX_IMPL=1

			for i in "${small_array[@]}"
			do
				./bank -n $threads -a $acct  -r 0 -w 0 -f  $filename2'-50I' -i $iter -d $duration $runflags -T $i
			done
			for ((  i = $start  ;  i <= $end;  i+=$step  ))
			do
				./bank -n $threads -a $acct  -r 0 -w 0 -f  $filename2'-50I' -i $iter -d $duration $runflags -T $i
			done
		fi

		if [ "$RUN100" == 1 ]; then
			#GEN HC

			rm -f  bank bank.o
			nvcc  -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm'  bank.c $tmincludes -lstm -lpthread -c $flags100 $flagsCPU
			make ROOT=$root USE_TSX_IMPL=1

			for i in "${small_array[@]}"
			do
				./bank -n $threads -a $acct  -r 0 -w 0 -f  $filename2'-100I' -i $iter -d $duration $runflags -T $i
			done
			for ((  i = $start  ;  i <= $end;  i+=$step  ))
			do
				./bank -n $threads -a $acct  -r 0 -w 0 -f  $filename2'-100I' -i $iter -d $duration $runflags -T $i
			done
		fi
	done
done

if [ "$VERBALLOOP" == 1 ]; then
	k=4
	tmincludes=${includes[$k]}
	logtype=${runtext[$k]}
	val=${enabled[$k]}
	root=${rootloc[$k]}
	flagsCPU=$flagsCPUL
	flagsGPU=$flagsGPUL
	flags5=$flags5L
	flags25=$flags25L
	flags50=$flags50L
	flags100=$flags100L
	filename2=$filename'LC'
	runflags=' '


	make clean
	nvcc  -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm'  pr-stm.cu $tmincludes -lstm -lpthread -c $flagsGPU
	make ROOT=$root USE_TSX_IMPL=1

	if [ "$RUN50" == 1 ]; then
		for i in "${bal_array[@]}"
		do
			for k in "${small_array[@]}"
			do
				rm -f  bank bank.o
				nvcc  -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm'  bank.c -lstm -lpthread -I tiny2/include -I tiny2/src -c -DCPUEn=1 -DGPUEn=1 $flagsCPU -DSYNC_BALANCING_VALF=10 -DSYNC_BALANCING_VALS=$i -DLABEL=$i $flags50
				make
				./bank -n $threads -a $acct  -r 0 -w 0 -f $filename2'-VBAL'$i'-50I' -i $iter -d $duration -T $k $runflags
			done
			for ((  k = $start  ;  k <= $end;  k+=$step  ))
			do
				rm -f  bank bank.o
				nvcc  -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm'  bank.c -lstm -lpthread -I tiny2/include -I tiny2/src -c -DCPUEn=1 -DGPUEn=1 $flagsCPU -DSYNC_BALANCING_VALF=10 -DSYNC_BALANCING_VALS=$i -DLABEL=$i $flags50
				make
				./bank -n $threads -a $acct  -r 0 -w 0 -f $filename2'-VBAL'$i'-50I' -i $iter -d $duration -T $k $runflags
			done

		done
	fi

	if [ "$RUN50" == 1 ]; then
		for i in "${bal_array2[@]}"
		do
			for k in "${small_array[@]}"
			do
				rm -f  bank bank.o
				nvcc  -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm'  bank.c -lstm -lpthread -I tiny2/include -I tiny2/src -c -DCPUEn=1 -DGPUEn=1 $flagsCPU -DSYNC_BALANCING_VALS=$i -DSYNC_BALANCING_VALS=$i -DLABEL=$i $flags100
				make
				./bank -n $threads -a $acct  -r 0 -w 0 -f $filename2'-VBAL'$i'-100I' -i $iter -d $duration -T $k $runflags
			done
			for ((  k = $start  ;  k <= $end;  k+=$step  ))
			do
				rm -f  bank bank.o
				nvcc  -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm'  bank.c -lstm -lpthread -I tiny2/include -I tiny2/src -c -DCPUEn=1 -DGPUEn=1 $flagsCPU -DSYNC_BALANCING_VALF=$i -DSYNC_BALANCING_VALS=$i -DLABEL=$i $flags100
				make
				./bank -n $threads -a $acct  -r 0 -w 0 -f $filename2'-VBAL'$i'-100I' -i $iter -d $duration -T $k $runflags
			done

		done
	fi

	#./loopBank.sh
fi


#./makeTM.sh
# for ((  i = 50000;  i <= 1000000;  i+=50000  ))
# do
	# ./bank -n 4 -a $i  -r 0 -w 0 -f BankFinalS-V -i $iter -d 1000 -j
# done
# for ((  i = 128*1024;  i <= 8*1024*1024;  i+=512*1024  ))
# do
	# ./bank -n 4 -a $i  -r 0 -w 0 -f BankFinalS3-V -i $iter -d 1000 -j
# done

# start=2
# end=12
# for ((  i = $start  ;  i <= $end;  i+=2  ))
# do
	# make clean
	# nvcc  -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm'  pr-stm.cu $tmincludes -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c -DBANK_NB_TRANSFERS=$i -DfirstX=1
	# make
	# ./bank -n 4 -a $acct  -r 0 -w 0 -f $filename'-GPU' -i $iter -d $duration  -t $i
# done

# for ((  i = $start  ;  i <= $end;  i+=2  ))
# do
	# make clean
	# nvcc  -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm'  pr-stm.cu $tmincludes -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c -DBANK_NB_TRANSFERS=$i -DfirstX=5
	# make
	# ./bank -n 4 -a $acct  -r 0 -w 0 -f $filename'MCL-GPU' -i $iter -d $duration -t $i
# done

# for ((  i = $start  ;  i <= $end;  i+=2  ))
# do
	# make clean
	# nvcc  -DUSE_TSX_IMPL -I ~/projs/arch_dep/include -I ~/projs/htm_alg/include -L ~/projs/htm_alg/bin -lhtm_sgl --compiler-options='-mrtm'  pr-stm.cu $tmincludes -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c -DBANK_NB_TRANSFERS=$i -DfirstX=7
	# make
	# ./bank -n 4 -a $acct  -r 0 -w 0 -f $filename'HCL-GPU' -i $iter -d $duration -t $i
# done
