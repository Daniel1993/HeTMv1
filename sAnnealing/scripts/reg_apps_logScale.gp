set terminal dumb
#..........................................................
#..............................................................................................................
#  FFFFFFF IIIII TTTTTTT TTTTTTT IIIII NN   NN   GGGG      SSSSS  EEEEEEE  CCCCC  TTTTTTT IIIII  OOOOO  NN   NN 
#  FF       III    TTT     TTT    III  NNN  NN  GG  GG    SS      EE      CC    C   TTT    III  OO   OO NNN  NN 
#  FFFF     III    TTT     TTT    III  NN N NN GG          SSSSS  EEEEE   CC        TTT    III  OO   OO NN N NN 
#  FF       III    TTT     TTT    III  NN  NNN GG   GG         SS EE      CC    C   TTT    III  OO   OO NN  NNN 
#  FF      IIIII   TTT     TTT   IIIII NN   NN  GGGGGG     SSSSS  EEEEEEE  CCCCC    TTT   IIIII  OOOO0  NN   NN 
#..............................................................................................................
#..........................................................

# Basic fitting function
# basic line
# curve(x, a, b, c, d) = a+b/(x+c)**d
# f(x, a, b, c, d, s) = x > s ? curve(s, a, b, c, d) : curve(x, a, b, c, d)

# Parameter restrictions all >=0
# g(x, a, b, c, d, s) = f(x, abs(a), abs(b), abs(c), abs(d), abs(s))
max(x, y) = (x > y ? x : y)

set grid ytics
set fit results

# Initial Guess for annealing
# c_An=1.0
# d_An=1.0
stats sprintf("%s.data2", ARG1) using 2 nooutput
# s_An=(STATS_max-STATS_min)/2
# stats sprintf("%s.data", ARG1) using 1
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< IMPORTANT : Min needed for plotting bellow
Solution = ARG3
ARGMAXX = ARG4
MinX = STATS_min
if (ARGMAXX > 0) {
     MaxX = ARGMAXX
} else {
     MaxX = STATS_max
}

stats sprintf("%s.data2", ARG1) using 4 nooutput
MinY = STATS_min
MaxY = STATS_max

# a_An=STATS_min
# b_An=STATS_max-a_An

#fit g(x,a_An,b_An,c_An,d_An,s_An) sprintf("%s.data", ARG1) using 3:1 via s_An,c_An

# Initial Guess for COLD

# Initial Guess for annealing
# c_Co=1.0
# d_Co=1.0
# stats sprintf("%s.cold", ARG1) using 3
# s_Co=(STATS_max-STATS_min)/2
# stats sprintf("%s.cold", ARG1) using 1
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<< IMPORTANT : STATS_max needed for plotting bellow
# if(STATS_min < Min){
# 	     Min=STATS_min
# }
# if(STATS_max > Max){
# 	     Max=STATS_max
# }
# a_Co=STATS_min
# b_Co=STATS_max-a_Co

#fit g(x,a_Co,b_Co,c_Co,s_Co) sprintf("%s.cold", ARG1) using 3:1 via s_Co,a_Co

#..........................................................
#................................................................................................................
#  PPPPPP  LL       OOOOO  TTTTTTT IIIII NN   NN   GGGG      SSSSS  EEEEEEE  CCCCC  TTTTTTT IIIII  OOOOO  NN   NN 
#  PP   PP LL      OO   OO   TTT    III  NNN  NN  GG  GG    SS      EE      CC    C   TTT    III  OO   OO NNN  NN 
#  PPPPPP  LL      OO   OO   TTT    III  NN N NN GG          SSSSS  EEEEE   CC        TTT    III  OO   OO NN N NN 
#  PP      LL      OO   OO   TTT    III  NN  NNN GG   GG         SS EE      CC    C   TTT    III  OO   OO NN  NNN 
#  PP      LLLLLLL  OOOO0    TTT   IIIII NN   NN  GGGGGG     SSSSS  EEEEEEE  CCCCC    TTT   IIIII  OOOO0  NN   NN 
#................................................................................................................
#..........................................................

keyBB = "BB"
keySA = "SA"
keySAcold = "SA cold"
keySAhot = "SA hot"
keySAhottest = "SA hottest"
keyHyData = "Hybrid"
keyGreedy = "Greedy"
keyGreedyNoInit = "Greedy (no init)"
keySAcondNoInit = "SA cold (no init)"

if (ARG2[strlen(ARG2)-2:] eq 'tex') {
     #set terminal cairolatex size 4,3
     set terminal cairolatex size 2.80,2.0
     set output sprintf("%s_time.tex", ARG1)
     set title system(sprintf("echo %s | tr '_' '-'", ARG1))
} else { if (ARG2[strlen(ARG2)-2:] eq 'pdf') {
     #set terminal cairolatex size 4,3
     set terminal pdf size 5,4
     set output sprintf("%s_time.pdf", ARG1)
     set title system(sprintf("echo %s | tr '_' '-'", ARG1))
} else { if (ARG2[strlen(ARG2)-2:] eq 'jpg') {
     set terminal jpeg enhanced large size 800,560
     set output sprintf("%s_time.jpg", ARG1)
     set title system(sprintf("echo %s | tr '_' '-'", ARG1))
     # set xtics rotate by 20
} else {
     set terminal pngcairo noenhanced size 800,560
     set output sprintf("%s_time.png", ARG1)
     set title sprintf("%s", ARG1)
}}}

set ytics nomirror
# set y2tics

xunit=1
set xlabel "Time (millis)"

# if(MaxX > 1000.0) {
# 	xunit=1000
# 	set xlabel "Time (seconds)"
# }

# if(MaxX > 90.0*60.0) {
# 	xunit=1000*60
# 	set xlabel "Time (minutes)"
# }

#set ylabel "Approximation factor"
#set y2label "Number of 1's"

y_diff=MaxY-MinY

### TODO: what to put here?
set link y2 via Solution-((1-y)*Solution) inverse 1-((Solution-y)/Solution)
set y2tics (MinY, floor(y_diff*0.2+MinY), floor(y_diff*0.4+MinY), floor(y_diff*0.6+MinY), floor(y_diff*0.8+MinY), MaxY)
set y2range [MinY-MinY*0.01:MaxY+MaxY*0.01]
set yrange[1-((Solution-MinY)/Solution)-0.05:1.01]
set xrange[0:MaxX/xunit]
# set ytics ((Solution-floor(y_diff*0.4+STATS_min))/Solution, (Solution-floor(y_diff*0.2+STATS_min))/Solution, 0.00)
# set ytics (0.00, 0.10, 0.20)
#, (Solution-floor(y_diff*0.4+STATS_min))/Solution, (Solution-floor(y_diff*0.6+STATS_min))/Solution, (Solution-floor(y_diff*0.8+STATS_min))/Solution, (Solution-STATS_max))/Solution)

# if(Min < Max){
#      set yrange [1.0-0.10*((Max-Min)/Min):(Max/Min)+0.10*((Max-Min)/Min)]
# } else {
#      set yrange [0.9:1.1]
# }
set key maxrow 3 bot right

set logscale x 10
# set logscale x

# plot sprintf("%s.cold", ARG1) using ($3/xunit):1 axis x1y2 \
#      with dots linecolor rgbcolor "#000AC4",\


plot \
     sprintf("%s.data2", ARG1) using ($2/xunit):($4+(rand(0)-0.5)/2) notitle axis x1y2 \
     with dots linecolor rgbcolor "#FF0000", \
     sprintf("%s.hydata", ARG1) using ($2/xunit):($4+(rand(0)-0.5)/2) notitle axis x1y2 \
     with dots linecolor rgbcolor "#BA009A", \
     sprintf("%s.bb", ARG1) using ($2/xunit):($4+(rand(0)-0.5)/2) notitle axis x1y2 \
     with dots linecolor rgbcolor "#13FF03", \
     1/0 with points linecolor rgbcolor "#13FF03"  pt 5 lw 3 ps 1 ti keyBB, \
     1/0 with points linecolor rgbcolor "#FF0000"  pt 5 lw 3 ps 1 ti keySA, \
     1/0 with points linecolor rgbcolor "#BA009A"  pt 5 lw 3 ps 1 ti keyHyData


#     g(x,a_An,b_An,c_An,d_An,s_An) axis x1y2,\
#     g(x,a_Co,b_Co,c_Co,d_An,s_Co) axis x1y2

# plot sprintf("%s.data", ARG1) using 3:1 axis x1y2 with dots,\
#  '' using 3:1:(1000.0) axis x1y2 smooth acsplines, \
#  sprintf("%s.cold", ARG1) using 3:1 axis x1y2 with dots,\
#  '' using 3:1:(1000.0) axis x1y2 smooth acsplines, \
