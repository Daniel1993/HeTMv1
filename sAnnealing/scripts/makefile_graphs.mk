$(GRAPH_PATH_M)/reg_128_%.in :
	$(HOME_PATH_M)/tests/k-regular $* 128 > $@
$(GRAPH_PATH_M)/reg_512_%.in :
	$(HOME_PATH_M)/tests/k-regular $* 512 > $@
$(GRAPH_PATH_M)/reg_1024_%.in :
	$(HOME_PATH_M)/tests/k-regular $* 1024 > $@
$(GRAPH_PATH_M)/reg_2048_%.in :
	$(HOME_PATH_M)/tests/k-regular $* 2048 > $@
$(GRAPH_PATH_M)/reg_4096_%.in :
	$(HOME_PATH_M)/tests/k-regular $* 4096 > $@
$(GRAPH_PATH_M)/reg_65536_%.in :
	$(HOME_PATH_M)/tests/k-regular $* 65536 > $@
$(GRAPH_PATH_M)/reg_262144_%.in :
	$(HOME_PATH_M)/tests/k-regular $* 262144 > $@
$(GRAPH_PATH_M)/reg_1048576_%.in :
	$(HOME_PATH_M)/tests/k-regular $* 1048576 > $@

$(GRAPH_PATH_M)/kbp_%.in : kbp_%.guess
	echo $* | awk -F"_" "{print \$$1}" > KBP_L
	echo $* | awk -F"_" "{print \$$2}" > KBP_F
	echo $* | awk -F"_" "{print \$$3}" > KBP_D
	echo $* | awk -F"_" "{print \$$4}" > KBP_K
	$(HOME_PATH_M)/tests/bipartite-k-regular $$(cat KBP_L) $$(cat KBP_F) $$(cat KBP_D) $$(cat KBP_K) > $@
	rm KBP_L KBP_F KBP_D KBP_K

$(GRAPH_PATH_M)/tor_%.in :
	$(HOME_PATH_M)/tests/torus $* > $@

kmeans%.bb :
	for i in $$(seq 0 7) ; do \
		$(HOME_PATH_M)/BB/project $(GRAPH_PATH_M)/kmeans$*.in ${ITERS} 4 $$i > $@.core$$i & \
	done ; wait 
	rm -f $@
	for i in $$(seq 0 7) ; do \
		cat $@.core$$i >> $@ ; rm $@.core$$i ; done

tpcc%.bb :
	for i in $$(seq 0 7) ; do \
		$(HOME_PATH_M)/BB/project $(GRAPH_PATH_M)/tpcc$*.in ${ITERS} 4 $$i > $@.core$$i & \
	done ; wait 
	rm -f $@
	for i in $$(seq 0 7) ; do \
		cat $@.core$$i >> $@ ; rm $@.core$$i ; done

tpcc30%.bb :
	for i in $$(seq 0 7) ; do \
		$(HOME_PATH_M)/BB/project $(GRAPH_PATH_M)/tpcc30$*.in ${ITERS} 4 $$i > $@.core$$i & \
	done ; wait 
	rm -f $@
	for i in $$(seq 0 7) ; do \
		cat $@.core$$i >> $@ ; rm $@.core$$i ; done

genome%.bb :
	for i in $$(seq 0 7) ; do \
		$(HOME_PATH_M)/BB/project $(GRAPH_PATH_M)/genome$*.in ${ITERS} 4 $$i > $@.core$$i & \
	done ; wait 
	rm -f $@
	for i in $$(seq 0 7) ; do \
		cat $@.core$$i >> $@ ; rm $@.core$$i ; done

intruder%.bb :
	for i in $$(seq 0 7) ; do \
		$(HOME_PATH_M)/BB/project $(GRAPH_PATH_M)/intruder$*.in ${ITERS} 4 $$i > $@.core$$i & \
	done ; wait 
	rm -f $@
	for i in $$(seq 0 7) ; do \
		cat $@.core$$i >> $@ ; rm $@.core$$i ; done

kbp_%.bb : $(GRAPH_PATH_M)/kbp_%.in
	$(HOME_PATH_M)/BB/project $(GRAPH_PATH_M)/kbp_$*.in ${ITERS} 4 > $@

tor_2.bb : $(GRAPH_PATH_M)/tor_2.in
	$(MAKE) tor_2.guess
	bash temperatureTest.sh tor_2 bb ${REPS} ${CORES} ${ITERS} \
		${ProbHot} ${HotD} \
		${ProbCold} 1 all

tor_3.bb : $(GRAPH_PATH_M)/tor_3.in
	$(MAKE) tor_3.guess
	bash temperatureTest.sh tor_3 bb ${REPS} ${CORES} ${ITERS} \
		${ProbHot} ${HotD} \
		${ProbCold} 1 all

tor_4.bb : $(GRAPH_PATH_M)/tor_4.in
	$(MAKE) tor_4.guess
	bash temperatureTest.sh tor_4 bb ${REPS} ${CORES} ${ITERS} \
		${ProbHot} ${HotD} \
		${ProbCold} 1 all

tor_5.bb : $(GRAPH_PATH_M)/tor_5.in
	$(MAKE) tor_5.guess
	bash temperatureTest.sh tor_5 bb ${REPS} ${CORES} ${ITERS} \
		${ProbHot} ${HotD} \
		${ProbCold} 1 all

tor_6.bb : $(GRAPH_PATH_M)/tor_6.in
	$(MAKE) tor_6.guess
	bash temperatureTest.sh tor_6 bb ${REPS} ${CORES} ${ITERS} \
		${ProbHot} ${HotD} \
		${ProbCold} 1

tor_7.bb : $(GRAPH_PATH_M)/tor_7.in
	$(MAKE) tor_7.guess
	bash temperatureTest.sh tor_7 bb ${REPS} ${CORES} ${ITERS} \
		${ProbHot} ${HotD} \
		${ProbCold} 1

reg_%.bb : $(GRAPH_PATH_M)/reg_%.in
	$(HOME_PATH_M)/BB/project $(GRAPH_PATH_M)/reg_$*.in ${ITERS} 4 > $@

tpcc%.guess : tpcc%.bb
	$(shell cat $^ | tail -n 1 | sed 's/^# //' > $@)

genome%.guess : genome%.bb
	$(shell cat $^ | tail -n 1 | sed 's/^# //' > $@)

kmeans%.guess : kmeans%.bb
	$(shell cat $^ | tail -n 1 | sed 's/^# //' > $@)

intruder%.guess : intruder%.bb
	$(shell cat $^ | tail -n 1 | sed 's/^# //' > $@)

reg_%_1.guess :
	echo "$* - 1" | bc > $@
reg_%_8.guess :
	echo "$* - 8" | bc > $@
reg_%_128.guess :
	echo "$* - 128" | bc > $@
reg_%_256.guess : 
	echo "$* - 256" | bc > $@
reg_%_512.guess : 
	echo "$* - 512" | bc > $@
reg_%_1024.guess : 
	echo "$* - 1024" | bc > $@

kbp_%.guess : 
	echo $* | awk -F"_" "{print \$$1}" > KBP_L
	echo $* | awk -F"_" "{print \$$2}" > KBP_F
	echo $* | awk -F"_" "{print \$$3}" > KBP_D
	echo $* | awk -F"_" "{print \$$4}" > KBP_K
	echo $$(echo "$$(cat KBP_L) * $$(cat KBP_F)" | bc) > $@
	cat $@ 
	rm KBP_L KBP_F KBP_D KBP_K

tor_%.guess :
	echo $$(echo "($* * $*) - $*" | bc) > $@

