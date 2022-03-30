kbp_%.in :
	echo $* | awk -F"_" "{print \$$1}" > KBP_L
	echo $* | awk -F"_" "{print \$$2}" > KBP_F
	echo $* | awk -F"_" "{print \$$3}" > KBP_D
	echo $* | awk -F"_" "{print \$$4}" > KBP_K
	../tests/bipartite-k-regular $$(cat KBP_L) $$(cat KBP_F) $$(cat KBP_D) $$(cat KBP_K) > $@
	rm KBP_L KBP_F KBP_D KBP_K

tor_%.in :
	../tests/torus $* > $@
