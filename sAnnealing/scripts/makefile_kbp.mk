kbp_8_%.data    : ITERS ::= 22
kbp_8_%.data1   : ITERS ::= 22
kbp_8_%.data2   : ITERS ::= 22
kbp_8_%.data3   : ITERS ::= 22
kbp_8_%.greedy  : ITERS ::= 22
kbp_8_%.gnoinit : ITERS ::= 22
kbp_8_%.noinit  : ITERS ::= 22
kbp_8_%.cold    : ITERS ::= 22
kbp_8_%.bb      : ITERS ::= 31

kbp_32_%.data    : ITERS ::= 24
kbp_32_%.data1   : ITERS ::= 24
kbp_32_%.data2   : ITERS ::= 24
kbp_32_%.data3   : ITERS ::= 24
kbp_32_%.greedy  : ITERS ::= 24
kbp_32_%.gnoinit : ITERS ::= 24
kbp_32_%.noinit  : ITERS ::= 24
kbp_32_%.cold    : ITERS ::= 24
kbp_32_%.bb      : ITERS ::= 21

kbp_32_%.data1   : HotD ::= 2
kbp_32_%.data1   : ColdD ::= 1
kbp_32_%.data1   : ProbHot ::= 0.01
kbp_32_%.data1   : ProbCold ::= 0.0001

kbp_32_%.data2   : HotD ::= 2
kbp_32_%.data2   : ColdD ::= 1
kbp_32_%.data2   : ProbHot ::= 0.001
kbp_32_%.data2   : ProbCold ::= 0.00001

kbp_32_%.data3   : HotD ::= 2
kbp_32_%.data3   : ColdD ::= 1
kbp_32_%.data3   : ProbHot ::= 0.0001
kbp_32_%.data3   : ProbCold ::= 0.000001

kbp_64_%.data    : ITERS ::= 25
kbp_64_%.data1   : ITERS ::= 25
kbp_64_%.data2   : ITERS ::= 25
kbp_64_%.data3   : ITERS ::= 25
kbp_64_%.greedy  : ITERS ::= 25
kbp_64_%.gnoinit : ITERS ::= 25
kbp_64_%.noinit : ITERS ::= 25
kbp_64_%.cold    : ITERS ::= 25
kbp_64_%.bb      : ITERS ::= 22

kbp_128_%.data    : ITERS ::= 25
kbp_128_%.data1   : ITERS ::= 25
kbp_128_%.data2   : ITERS ::= 25
kbp_128_%.data3   : ITERS ::= 25
kbp_128_%.greedy  : ITERS ::= 25
kbp_128_%.gnoinit : ITERS ::= 25
kbp_128_%.noinit : ITERS ::= 25
kbp_128_%.cold    : ITERS ::= 25
kbp_128_%.bb      : ITERS ::= 22

kbp_256_%.data    : ITERS ::= 26
kbp_256_%.data1   : ITERS ::= 26
kbp_256_%.data2   : ITERS ::= 26
kbp_256_%.data3   : ITERS ::= 26
kbp_256_%.greedy  : ITERS ::= 26
kbp_256_%.gnoinit : ITERS ::= 26
kbp_256_%.noinit : ITERS ::= 26
kbp_256_%.cold    : ITERS ::= 26
kbp_256_%.bb      : ITERS ::= 22

kbp_512_%.data : ProbHot ::= 0.05
kbp_512_%.data : ProbCold ::= 0.01
kbp_512_%.data : ITERS ::= 28
kbp_512_%.cold : ITERS ::= 28
kbp_512_%.bb : ITERS ::= 28

kbp_1024_%.data : ProbHot ::= 0.05
kbp_1024_%.data : ProbCold ::= 0.01
kbp_1024_%.data : ITERS ::= 30
kbp_1024_%.cold : ITERS ::= 30
kbp_1024_%.bb : ITERS ::= 30

kbp_2048_%.data : ProbHot ::= 0.05
kbp_2048_%.data : ProbCold ::= 0.01
kbp_2048_%.data : ITERS ::= 32
kbp_2048_%.cold : ITERS ::= 32
kbp_2048_%.bb : ITERS ::= 32

kbp_4096_%.data : ProbHot ::= 0.05
kbp_4096_%.data : ProbCold ::= 0.01
kbp_4096_%.data : ITERS ::= 34
kbp_4096_%.cold : ITERS ::= 34
kbp_4096_%.bb : ITERS ::= 34

kbp_8192_%.data : ProbHot ::= 0.05
kbp_8192_%.data : ProbCold ::= 0.01
kbp_8192_%.data : ITERS ::= 35
kbp_8192_%.cold : ITERS ::= 35
kbp_8192_%.bb : ITERS ::= 35


deps_kbp_max_cost : kbp_32_2_4_4.data.max_cost kbp_64_2_4_4.data.max_cost kbp_128_2_4_4.data.max_cost kbp_256_2_4_4.data.max_cost 
deps_kbp_data : kbp_32_2_4_4.png kbp_64_2_4_4.png kbp_128_2_4_4.png kbp_256_2_4_4.png
deps_kbp_data_biter1 : kbp_32_2_4_4.data.biter1 kbp_64_2_4_4.data.biter1 kbp_128_2_4_4.data.biter1 kbp_256_2_4_4.data.biter1 
deps_kbp_data1_biter1 : kbp_32_2_4_4.data1.biter1 kbp_64_2_4_4.data1.biter1 kbp_128_2_4_4.data1.biter1 kbp_256_2_4_4.data1.biter1 
deps_kbp_data2_biter1 : kbp_32_2_4_4.data2.biter1 kbp_64_2_4_4.data2.biter1 kbp_128_2_4_4.data2.biter1 kbp_256_2_4_4.data2.biter1 
deps_kbp_data3_biter1 : kbp_32_2_4_4.data3.biter1 kbp_64_2_4_4.data3.biter1 kbp_128_2_4_4.data3.biter1 kbp_256_2_4_4.data3.biter1 
deps_kbp_cold_biter1 : kbp_32_2_4_4.cold.biter1 kbp_64_2_4_4.cold.biter1 kbp_128_2_4_4.cold.biter1 kbp_256_2_4_4.cold.biter1 
deps_kbp_greedy_biter1 : kbp_32_2_4_4.greedy.biter1 kbp_64_2_4_4.greedy.biter1 kbp_128_2_4_4.greedy.biter1 kbp_256_2_4_4.greedy.biter1 
deps_kbp_gnoinit_biter1 : kbp_32_2_4_4.gnoinit.biter1 kbp_64_2_4_4.gnoinit.biter1 kbp_128_2_4_4.gnoinit.biter1 kbp_256_2_4_4.gnoinit.biter1 
deps_kbp_biter1 : deps_kbp_data_biter1 deps_kbp_data1_biter1 deps_kbp_data2_biter1 deps_kbp_data3_biter1 deps_kbp_cold_biter1 deps_kbp_greedy_biter1 deps_kbp_gnoinit_biter1

kbp.max_cost : deps_kbp_max_cost
	rm kbp.max_cost ; \
	j=0 ; \
	for i in 32 64 128 256; do \
	echo -ne "$$(($$j - 1))\t" >> kbp.max_cost ; \
	cat kbp_$${i}_2_4_4.data.max_cost >> kbp.max_cost ; \
	echo -ne "\n$${j}\t" >> kbp.max_cost ; j=$$(($$j + 1)) ; \
	cat kbp_$${i}_2_4_4.data.max_cost >> kbp.max_cost ; \
	echo "" >> kbp.max_cost ; \
	done

kbp.biter1.png : deps_kbp_data deps_kbp_biter1 kbp.max_cost
	-python proc_stats_per_graph.py kbp.data.biter1 kbp_32_2_4_4.data.biter1 kbp_64_2_4_4.data.biter1 kbp_128_2_4_4.data.biter1 kbp_256_2_4_4.data.biter1 
	-python proc_stats_per_graph.py kbp.data1.biter1 kbp_32_2_4_4.data1.biter1 kbp_64_2_4_4.data1.biter1 kbp_128_2_4_4.data1.biter1 kbp_256_2_4_4.data1.biter1 
	-python proc_stats_per_graph.py kbp.data2.biter1 kbp_32_2_4_4.data2.biter1 kbp_64_2_4_4.data2.biter1 kbp_128_2_4_4.data2.biter1 kbp_256_2_4_4.data2.biter1 
	-python proc_stats_per_graph.py kbp.data3.biter1 kbp_32_2_4_4.data3.biter1 kbp_64_2_4_4.data3.biter1 kbp_128_2_4_4.data3.biter1 kbp_256_2_4_4.data3.biter1 
	-python proc_stats_per_graph.py kbp.cold.biter1 kbp_32_2_4_4.cold.biter1 kbp_64_2_4_4.cold.biter1 kbp_128_2_4_4.cold.biter1 kbp_256_2_4_4.cold.biter1 
	-python proc_stats_per_graph.py kbp.greedy.biter1 kbp_32_2_4_4.greedy.biter1 kbp_64_2_4_4.greedy.biter1 kbp_128_2_4_4.greedy.biter1 kbp_256_2_4_4.greedy.biter1
	-python proc_stats_per_graph.py kbp.gnoinit.biter1 kbp_32_2_4_4.gnoinit.biter1 kbp_64_2_4_4.gnoinit.biter1 kbp_128_2_4_4.gnoinit.biter1 kbp_256_2_4_4.gnoinit.biter1 
	-gnuplot -c candlesticks3.gp kbp png 100 biter1 14 "Visited vertexes per iteration"

