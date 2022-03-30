reg_%.data : ProbHot ::= 0.05
reg_%.data : ProbCold ::= 0.01
reg_%.data : ITERS ::= 16
reg_%.cold : ITERS ::= 16

reg_%_128.data    : ITERS ::= 17
reg_%_128.cold    : ITERS ::= 17
reg_%_128.greedy  : ITERS ::= 17
reg_%_128.gnoinit : ITERS ::= 17
reg_%_128.bb      : ITERS ::= 26

reg_%_256.data : ProbHot ::= 0.05
reg_%_256.data : ProbCold ::= 0.01
reg_%_256.data : ITERS ::= 18
reg_%_256.cold : ITERS ::= 18

reg_%_512.data : ProbHot ::= 0.05
reg_%_512.data : ProbCold ::= 0.01
reg_%_512.data : ITERS ::= 20
reg_%_512.cold : ITERS ::= 20

reg_%_1024.data : ProbHot ::= 0.05
reg_%_1024.data : ProbCold ::= 0.01
reg_%_1024.data : ITERS ::= 22
reg_%_1024.cold : ITERS ::= 22
