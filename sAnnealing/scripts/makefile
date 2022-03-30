SHELL := /bin/bash

GRAPH_PATH_M ?= ../TX_graph_examples
HOME_PATH_M  ?= ..
NB_BUCKETS = 4

.PHONY: all clean
# secondary without prerequisites keeps all intermediate files
.SECONDARY:
# .PRECIOUS: %.data %.data1 %.data2 %.data3 %.cold %.greedy %.noinit %.gnoinit %.bb %.hydata $(GRAPH_PATH_M)/%.in %.guess \
# 	%.png %.biter.sol.png %.btime.sol.png %.biter.solSA.png %.btime.solSA.png %.biter.vert.png %.btime.vert.png \
# 	%.jpg %.biter.sol.jpg %.btime.sol.jpg %.biter.solSA.jpg %.btime.solSA.jpg %.biter.vert.jpg %.btime.vert.jpg \
# 	%.tex %.biter.sol.tex %.btime.sol.tex %.biter.solSA.tex %.btime.solSA.tex %.biter.vert.tex %.btime.vert.tex

### TODO: change ProbHot ProbCold
# 16_thrs_G1.in.data : ColdD ::= 1
# REPS ::= 4
# CORES ::= 4
REPS ::= 24
CORES ::= 12

all: \
tor_64.guess \
tor_64.jpg \
tor_64.biter.solSA.pdf \
tor_64.biter.sol.pdf \
tor_128.guess \
tor_128.jpg \
tor_128.biter.solSA.pdf \
tor_128.biter.sol.pdf \
tor_256.guess \
tor_256.jpg \
tor_256.biter.solSA.pdf \
tor_256.biter.sol.pdf \
tor_512.guess \
tor_512.jpg \
tor_512.biter.solSA.pdf \
tor_512.biter.sol.pdf \
tor_1024.guess \
tor_1024.jpg \
tor_1024.biter.solSA.pdf \
tor_1024.biter.sol.pdf

include makefile_env.mk
include makefile_graphs.mk
include makefile_kbp.mk
include makefile_tor.mk
include makefile_reg.mk
include makefile_tx_apps.mk

clean:
	rm -f tor_16.png
	rm -f tor_64.png
	rm -f reg_65536_1.png
	rm -f genome_1024_thrs_G1.png
	rm -f intruder_1024_thrs_G1.png

###
# GUESS is known for torus and k-regular, calculated with BB for the others

%.biter %.btime : % 
	-python proc_stats.py $* $(NB_BUCKETS)

%.biter1 %.btime1 : % 
	-python proc_stats.py $* 1
	mv $*.btime $*.btime1
	mv $*.biter $*.biter1

%.biter.solSA.png %.biter.solSA.jpg %.biter.solSA.tex %.biter.solSA.pdf : %.data.biter %.data1.biter %.data2.biter %.data3.biter %.cold.biter %.greedy.biter %.gnoinit.biter %.noinit.biter %.guess 
	-gnuplot -c candlesticks1.gp $* png $$(cat $*.guess) biter 3 "Normalized to guessed solution" "Number vertexes in solution" 2> /dev/null
	-gnuplot -c candlesticks1.gp $* jpg $$(cat $*.guess) biter 3 "Normalized to guessed solution" "Number vertexes in solution" 2> /dev/null
	-gnuplot -c candlesticks1.gp $* tex $$(cat $*.guess) biter 3 "Normalized to guessed solution" "Number vertexes in solution" 2> /dev/null
	-gnuplot -c candlesticks1.gp $* pdf $$(cat $*.guess) biter 3 "Normalized to guessed solution" "Number vertexes in solution" 2> /dev/null
	mv $*.biter.png $*.biter.solSA.png
	mv $*.biter.jpg $*.biter.solSA.jpg
	mv $*.biter.tex $*.biter.solSA.tex
	mv $*.biter.pdf $*.biter.solSA.pdf

%.btime.solSA.png %.btime.solSA.jpg %.btime.solSA.tex %.btime.solSA.pdf : %.data.btime %.data1.btime %.data2.btime %.data3.btime %.cold.btime %.greedy.btime %.gnoinit.btime %.noinit.btime %.guess 
	-gnuplot -c candlesticks1.gp $* png $$(cat $*.guess) btime 3 "Normalized to guessed solution" "Number vertexes in solution" 2> /dev/null
	-gnuplot -c candlesticks1.gp $* jpg $$(cat $*.guess) btime 3 "Normalized to guessed solution" "Number vertexes in solution" 2> /dev/null
	-gnuplot -c candlesticks1.gp $* tex $$(cat $*.guess) btime 3 "Normalized to guessed solution" "Number vertexes in solution" 2> /dev/null
	-gnuplot -c candlesticks1.gp $* pdf $$(cat $*.guess) btime 3 "Normalized to guessed solution" "Number vertexes in solution" 2> /dev/null
	mv $*.btime.png $*.btime.solSA.png
	mv $*.btime.jpg $*.btime.solSA.jpg
	mv $*.btime.tex $*.btime.solSA.tex
	mv $*.btime.pdf $*.btime.solSA.pdf

%.biter.sol.png %.biter.sol.jpg %.biter.sol.tex %.biter.sol.pdf : %.data.biter %.data1.biter %.data2.biter %.data3.biter %.cold.biter %.greedy.biter %.gnoinit.biter %.noinit.biter %.guess 
	-gnuplot -c candlesticks2.gp $* png $$(cat $*.guess) biter 3 "Normalized to guessed solution" "Number vertexes in solution" 2> /dev/null
	-gnuplot -c candlesticks2.gp $* jpg $$(cat $*.guess) biter 3 "Normalized to guessed solution" "Number vertexes in solution" 2> /dev/null
	-gnuplot -c candlesticks2.gp $* tex $$(cat $*.guess) biter 3 "Normalized to guessed solution" "Number vertexes in solution" 2> /dev/null
	-gnuplot -c candlesticks2.gp $* pdf $$(cat $*.guess) biter 3 "Normalized to guessed solution" "Number vertexes in solution" 2> /dev/null
	mv $*.biter.png $*.biter.sol.png
	mv $*.biter.jpg $*.biter.sol.jpg
	mv $*.biter.tex $*.biter.sol.tex
	mv $*.biter.pdf $*.biter.sol.pdf

%.btime.sol.png %.btime.sol.jpg %.btime.sol.tex %.btime.sol.pdf : %.data.btime %.data1.btime %.data2.btime %.data3.btime %.cold.btime %.greedy.btime %.gnoinit.btime %.noinit.btime %.guess 
	-gnuplot -c candlesticks2.gp $* png $$(cat $*.guess) btime 3 "Normalized to guessed solution" "Number vertexes in solution" 2> /dev/null
	-gnuplot -c candlesticks2.gp $* jpg $$(cat $*.guess) btime 3 "Normalized to guessed solution" "Number vertexes in solution" 2> /dev/null
	-gnuplot -c candlesticks2.gp $* tex $$(cat $*.guess) btime 3 "Normalized to guessed solution" "Number vertexes in solution" 2> /dev/null
	-gnuplot -c candlesticks2.gp $* pdf $$(cat $*.guess) btime 3 "Normalized to guessed solution" "Number vertexes in solution" 2> /dev/null
	mv $*.btime.png $*.btime.sol.png
	mv $*.btime.jpg $*.btime.sol.jpg
	mv $*.btime.tex $*.btime.sol.tex
	mv $*.btime.pdf $*.btime.sol.pdf

%.biter.vert.png %.biter.vert.jpg %.biter.vert.tex %.biter.vert.pdf : %.data.biter %.data1.biter %.data2.biter %.data3.biter %.cold.biter %.greedy.biter %.gnoinit.biter %.noinit.biter %.guess 
	-gnuplot -c candlesticks1.gp $* png $$(cat $*.data.max_cost) biter 13 "Normalized to O(d(v)logV)" "Visited vertexes" 2> /dev/null
	-gnuplot -c candlesticks1.gp $* jpg $$(cat $*.data.max_cost) biter 13 "Normalized to O(d(v)logV)" "Visited vertexes" 2> /dev/null
	-gnuplot -c candlesticks1.gp $* tex $$(cat $*.data.max_cost) biter 13 "Normalized to O(d(v)logV)" "Visited vertexes" 2> /dev/null
	-gnuplot -c candlesticks1.gp $* pdf $$(cat $*.data.max_cost) biter 13 "Normalized to O(d(v)logV)" "Visited vertexes" 2> /dev/null
	mv $*.biter.png $*.biter.vert.png
	mv $*.biter.jpg $*.biter.vert.jpg
	mv $*.biter.tex $*.biter.vert.tex
	mv $*.biter.pdf $*.biter.vert.pdf

%.btime.vert.png %.btime.vert.jpg %.btime.vert.tex %.btime.vert.pdf : %.data.btime %.data1.btime %.data2.btime %.data3.btime %.cold.btime %.greedy.btime %.gnoinit.btime %.noinit.btime %.guess 
	-gnuplot -c candlesticks1.gp $* png $$(cat $*.data.max_cost) btime 13 "Normalized to O(d(v)logV)" "Visited vertexes" 2> /dev/null
	-gnuplot -c candlesticks1.gp $* jpg $$(cat $*.data.max_cost) btime 13 "Normalized to O(d(v)logV)" "Visited vertexes" 2> /dev/null
	-gnuplot -c candlesticks1.gp $* tex $$(cat $*.data.max_cost) btime 13 "Normalized to O(d(v)logV)" "Visited vertexes" 2> /dev/null
	-gnuplot -c candlesticks1.gp $* pdf $$(cat $*.data.max_cost) btime 13 "Normalized to O(d(v)logV)" "Visited vertexes" 2> /dev/null
	mv $*.btime.png $*.btime.vert.png
	mv $*.btime.jpg $*.btime.vert.jpg
	mv $*.btime.tex $*.btime.vert.tex
	mv $*.btime.pdf $*.btime.vert.pdf

%.png %.jpg %.SA.png %.SA.jpg : %.data %.data1 %.data2 %.data3 %.cold %.guess %.greedy %.gnoinit %.noinit %.bb %.btime.vert.png %.biter.vert.png %.btime.sol.png %.biter.sol.png %.btime.solSA.png %.biter.solSA.png
	touch $@
	-gnuplot -c regression1.gp $* png $$(cat $*.guess) 2> /dev/null
	-gnuplot -c regression1.gp $* jpg $$(cat $*.guess) 2> /dev/null
	-gnuplot -c regression2.gp $* png $$(cat $*.guess) 2> /dev/null
	-gnuplot -c regression2.gp $* jpg $$(cat $*.guess) 2> /dev/null
	-gnuplot -c regression1_logScale.gp $* png $$(cat $*.guess) 2> /dev/null
	-gnuplot -c regression1_logScale.gp $* jpg $$(cat $*.guess) 2> /dev/null
	-gnuplot -c regression1a_logScale.gp $* png $$(cat $*.guess) 2> /dev/null
	-gnuplot -c regression1a_logScale.gp $* jpg $$(cat $*.guess) 2> /dev/null
# TODO: put more targets....
# -gnuplot -c candlesticks2.gp $* png $$(cat $*.guess) bucket_time 3 "Normalized to guessed solution" "Number vertexes in solution" 2> /dev/null
# -gnuplot -c candlesticks2.gp $* png $$(cat $*.guess) bucket_iteration 3 "Normalized to guessed solution" "Number vertexes in solution" 2> /dev/null

%.txt : %.txt.bz2
	bunzip2 $<

%.data : $(GRAPH_PATH_M)/%.in
	echo $@ | awk -F"." "{print \$$1}" > DATA_FILE
	echo $@ | awk -F"." "{print \$$2}" > DATA_EXT
	bash temperatureTest.sh $* $$(cat DATA_EXT) ${REPS} ${CORES} ${ITERS} \
		${ProbHot} ${HotD} \
		${ProbCold} 1
	rm DATA_FILE DATA_EXT

%.hydata : $(GRAPH_PATH_M)/%.in
	echo $@ | awk -F"." "{print \$$1}" > DATA_FILE
	echo $@ | awk -F"." "{print \$$2}" > DATA_EXT
	bash temperatureTest.sh $* $$(cat DATA_EXT) ${REPS} ${CORES} ${ITERS} \
		${ProbHot} ${HotD} \
		${ProbCold} 1
	rm DATA_FILE DATA_EXT

%.data1 : $(GRAPH_PATH_M)/%.in
	echo $@ | awk -F"." "{print \$$1}" > DATA_FILE
	echo $@ | awk -F"." "{print \$$2}" > DATA_EXT
	bash temperatureTest.sh $* $$(cat DATA_EXT) ${REPS} ${CORES} ${ITERS} \
		${ProbHot} ${HotD} \
		${ProbCold} 1
	rm DATA_FILE DATA_EXT

%.data2 : $(GRAPH_PATH_M)/%.in
	echo $@ | awk -F"." "{print \$$1}" > DATA_FILE
	echo $@ | awk -F"." "{print \$$2}" > DATA_EXT
	bash temperatureTest.sh $* $$(cat DATA_EXT) ${REPS} ${CORES} ${ITERS} \
		${ProbHot} ${HotD} \
		${ProbCold} 1
	rm DATA_FILE DATA_EXT

%.data3 : $(GRAPH_PATH_M)/%.in
	echo $@ | awk -F"." "{print \$$1}" > DATA_FILE
	echo $@ | awk -F"." "{print \$$2}" > DATA_EXT
	bash temperatureTest.sh $* $$(cat DATA_EXT) ${REPS} ${CORES} ${ITERS} \
		${ProbHot} ${HotD} \
		${ProbCold} 1
	rm DATA_FILE DATA_EXT

%.cold : $(GRAPH_PATH_M)/%.in
	bash temperatureTest.sh $* cold ${REPS} ${CORES} ${ITERS} \
	${ProbHot} ${HotD} \
	${ProbCold} 1

%.greedy : $(GRAPH_PATH_M)/%.in
	bash temperatureTest.sh $* greedy ${REPS} ${CORES} ${ITERS} \
	${ProbHot} ${HotD} \
	${ProbCold} 1

%.noinit : $(GRAPH_PATH_M)/%.in
	bash temperatureTest.sh $* noinit ${REPS} ${CORES} ${ITERS} \
	${ProbHot} ${HotD} \
	${ProbCold} 1

%.gnoinit : $(GRAPH_PATH_M)/%.in
	bash temperatureTest.sh $* gnoinit ${REPS} ${CORES} ${ITERS} \
	${ProbHot} ${HotD} \
	${ProbCold} 1

$(GRAPH_PATH_M)/%.pruned.in : $(GRAPH_PATH_M)/%.in
	$(HOME_PATH_M)/BB/project_CHECK_ONLY $^ 0 0
	mv in.max_scc $@
