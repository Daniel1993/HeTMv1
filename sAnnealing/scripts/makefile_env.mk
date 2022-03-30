### ENV vars for the script
%.hydata : export GRAPH_PATH = $(GRAPH_PATH_M)
%.hydata : export HOME_PATH = $(HOME_PATH_M)
%.hydata : export REP_OPS = 0.3
%.data : export GRAPH_PATH = $(GRAPH_PATH_M)
%.data : export HOME_PATH = $(HOME_PATH_M)
%.data : export REP_OPS = 0.4
%.bb : export GRAPH_PATH = $(GRAPH_PATH_M)
%.bb : export HOME_PATH = $(HOME_PATH_M)
%.bb : export REP_OPS = 0.3
%.data1 : export GRAPH_PATH = $(GRAPH_PATH_M)
%.data1 : export HOME_PATH = $(HOME_PATH_M)
%.data1 : export REP_OPS = 0.4
%.data2 : export GRAPH_PATH = $(GRAPH_PATH_M)
%.data2 : export HOME_PATH = $(HOME_PATH_M)
%.data2 : export REP_OPS = 0.3
%.data3 : export GRAPH_PATH = $(GRAPH_PATH_M)
%.data3 : export HOME_PATH = $(HOME_PATH_M)
%.data3 : export REP_OPS = 0.4
%.greedy : export GRAPH_PATH = $(GRAPH_PATH_M)
%.greedy : export HOME_PATH = $(HOME_PATH_M)
%.greedy : export REP_OPS = 0.4
%.noinit : export GRAPH_PATH = $(GRAPH_PATH_M)
%.noinit : export HOME_PATH = $(HOME_PATH_M)
%.noinit : export REP_OPS = 0.4
%.gnoinit : export GRAPH_PATH = $(GRAPH_PATH_M)
%.gnoinit : export HOME_PATH = $(HOME_PATH_M)
%.gnoinit : export REP_OPS = 0.4
%.cold : export GRAPH_PATH = $(GRAPH_PATH_M)
%.cold : export HOME_PATH = $(HOME_PATH_M)
%.cold : export REP_OPS = 0.4

%.data    : HotD ::= 2
%.data    : ColdD ::= 1
%.data    : ProbHot ::= 0.1
%.data    : ProbCold ::= 0.001

%.data1   : HotD ::= 1
%.data1   : ColdD ::= 1
%.data1   : ProbHot ::= 0.009747
%.data1   : ProbCold ::= 0.0001

%.data2   : HotD ::= 1
%.data2   : ColdD ::= 1
%.data2   : ProbHot ::= 0.0078
%.data2   : ProbCold ::= 0.00008

%.hydata   : HotD ::= 1
%.hydata   : ColdD ::= 1
%.hydata   : ProbHot ::= 0.0078
%.hydata   : ProbCold ::= 0.00008

%.data3   : HotD ::= 1
%.data3   : ColdD ::= 1
%.data3   : ProbHot ::= 0.00585
%.data3   : ProbCold ::= 0.00006

### missing data4 --> 0.0208

%.cold : HotD ::= 1
%.cold : ColdD ::= 1
%.cold : ProbHot ::= 0.000001
%.cold : ProbCold ::= 0.00000001

%.noinit   : HotD ::= 1
%.noinit   : ColdD ::= 1
%.noinit   : ProbHot ::= 0.000001
%.noinit   : ProbCold ::= 0.00000001

%.greedy   : HotD ::= 1
%.greedy   : ColdD ::= 1
%.greedy   : ProbHot ::= 0.1
%.greedy   : ProbCold ::= 0.001

%.gnoinit   : HotD ::= 1
%.gnoinit   : ColdD ::= 1
%.gnoinit   : ProbHot ::= 0.1
%.gnoinit   : ProbCold ::= 0.001

%.bb   : HotD ::= 1
%.bb   : ColdD ::= 1
%.bb   : ProbHot ::= 0.1
%.bb   : ProbCold ::= 0.001
