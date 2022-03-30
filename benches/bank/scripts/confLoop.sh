iter=30

# for ((  i = 125 ;  i <= 8000;  i=i*2  ))
# do
	# make clean
	# nvcc kernel.cu -L lib -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c -DTransEachThread=$i -DgenOnce=1 
	# make
	# ./bank -n 7 -a 100000  -r 0 -w 0 -f Bank4-0G-0C-0I -i $iter -d 5000 -j
# done

for ((  i = 125 ;  i <= 8000;  i=i*2  ))
do
	make clean
	nvcc kernel.cu -L lib -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c -DTransEachThread=$i -DgenOnce=1 -DCuda_Confl=1
	make
	./bank -n 7 -a 100000  -r 0 -w 0 -f Bank4-0G-0C-50I -i $iter -d 5000 -j
done

# for ((  i = 125 ;  i <= 8000;  i=i*2  ))
# do
	# make clean
	# nvcc kernel.cu -L lib -lstm -lpthread -I ../tinystm/include -I ../tinystm/src -c -DTransEachThread=$i -DgenOnce=1  -DDIFFERENT_POOLS=0
	# make
	# ./bank -n 7 -a 100000  -r 0 -w 0 -f Bank4-0G-0C-100I -i $iter -d 5000 -j
# done