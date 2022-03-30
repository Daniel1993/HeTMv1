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
set grid ytics
set grid noxtics

set fit results

FIT_PARAM_MIN = int(ARG5)
FIT_PARAM_10P = int(ARG5 + 1)
FIT_PARAM_MED = int(ARG5 + 2)
FIT_PARAM_90P = int(ARG5 + 3)
FIT_PARAM_MAX = int(ARG5 + 4)

stats sprintf("%s.hydata.%s", ARG1, ARG4) using 2 nooutput
Solution = ARG3
MinX = STATS_min
MaxX = STATS_max

stats sprintf("%s.hydata.%s", ARG1, ARG4) using FIT_PARAM_MIN nooutput
MinY = STATS_min
stats sprintf("%s.data2.%s", ARG1, ARG4) using FIT_PARAM_MIN nooutput
if (STATS_min < MinY) {
     MinY = STATS_min
}
stats sprintf("%s.bb.%s", ARG1, ARG4) using FIT_PARAM_MIN nooutput
if (STATS_min < MinY) {
     MinY = STATS_min
}

stats sprintf("%s.hydata.%s", ARG1, ARG4) using FIT_PARAM_MAX nooutput
MaxY = STATS_max
stats sprintf("%s.data2.%s", ARG1, ARG4) using FIT_PARAM_MAX nooutput
if (STATS_max > MaxY) {
     MaxY = STATS_max
}
stats sprintf("%s.bb.%s", ARG1, ARG4) using FIT_PARAM_MAX nooutput
if (STATS_max > MaxY) {
     MaxY = STATS_max
}

#..........................................................
#................................................................................................................
#  PPPPPP  LL       OOOOO  TTTTTTT IIIII NN   NN   GGGG      SSSSS  EEEEEEE  CCCCC  TTTTTTT IIIII  OOOOO  NN   NN 
#  PP   PP LL      OO   OO   TTT    III  NNN  NN  GG  GG    SS      EE      CC    C   TTT    III  OO   OO NNN  NN 
#  PPPPPP  LL      OO   OO   TTT    III  NN N NN GG          SSSSS  EEEEE   CC        TTT    III  OO   OO NN N NN 
#  PP      LL      OO   OO   TTT    III  NN  NNN GG   GG         SS EE      CC    C   TTT    III  OO   OO NN  NNN 
#  PP      LLLLLLL  OOOO0    TTT   IIIII NN   NN  GGGGGG     SSSSS  EEEEEEE  CCCCC    TTT   IIIII  OOOO0  NN   NN 
#................................................................................................................
#..........................................................


if (ARG2[strlen(ARG2)-2:] eq 'tex') {
     #set terminal cairolatex size 4,3
     set terminal cairolatex size 2.80,2.0
     set output sprintf("%s.%s.tex", ARG1, ARG4)
     set title system(sprintf("echo %s | tr '_' '-'", ARG1))
     set xtics rotate by -45
} else { if (ARG2[strlen(ARG2)-2:] eq 'pdf') {
     #set terminal cairolatex size 4,3
     set terminal pdf noenhanced size 5,4
     set output sprintf("%s.%s.pdf", ARG1, ARG4)
     set title system(sprintf("echo %s | tr '_' '-'", ARG1))
     set xtics rotate by -45
} else { if (ARG2[strlen(ARG2)-2:] eq 'jpg') {
     set terminal jpeg large enhanced size 800,560
     set output sprintf("%s.%s.jpg", ARG1, ARG4)
     set title system(sprintf("echo %s | tr '_' '-'", ARG1))
} else { if (ARG2[strlen(ARG2)-2:] eq 'eps') {
     set terminal postscript eps color noenhanced size 4,3
     set output sprintf("%s.%s.eps", ARG1, ARG4)
     set title system(sprintf("echo %s | tr '_' '-'", ARG1))
     set xtics rotate by -20
} else {
     set terminal pngcairo noenhanced size 800,560
     set output sprintf("%s.%s.png", ARG1, ARG4)
     set title sprintf("%s", ARG1)
     set xtics rotate by -45
}}}}

set ytics nomirror
set xlabel "Input graph"

#set ylabel "Approximation factor"
#set y2label "Number of 1's"

set key maxrow 3 top left

# set logscale x

# plot sprintf("%s.cold", ARG1) using ($3/xunit):1 axis x1y2 \
#      with dots linecolor rgbcolor "#000AC4",\

set xrange[MinX-0.4:MaxX+0.4]
# set xrange[-0.01:1.01]
set boxwidth 0.13
set style fill solid

y_diff=MaxY-MinY
# set link y2 via Solution-((1-y)*Solution) inverse 1-((Solution-y)/Solution)
# set y2tics (MinY, floor(y_diff*0.2+MinY), floor(y_diff*0.4+MinY), floor(y_diff*0.6+MinY), floor(y_diff*0.8+MinY), MaxY)
# set y2range [MinY-MinY*0.1:MaxY+MaxY*0.1]
set ylabel ARG6
# set y2label ARG7
set logscale y 10

plot \
     sprintf("%s.hydata.%s", ARG1, ARG4) using ($2-0.3):FIT_PARAM_10P:FIT_PARAM_MIN:FIT_PARAM_MAX:FIT_PARAM_90P:xtic(1) axis x1y1 \
          ti "Hybrid" with candlesticks linecolor rgbcolor "#FF0000" whiskerbars, \
     "" using ($2-0.3):FIT_PARAM_MED:FIT_PARAM_MED:FIT_PARAM_MED:FIT_PARAM_MED axis x1y1 notitle \
          with candlesticks linecolor rgbcolor "#AA0000" whiskerbars, \
     sprintf("%s.data2.%s", ARG1, ARG4) using ($2-0.15):FIT_PARAM_10P:FIT_PARAM_MIN:FIT_PARAM_MAX:FIT_PARAM_90P axis x1y1 \
          ti "SA hot"  with candlesticks linecolor rgbcolor "#FFA042" whiskerbars, \
     "" using ($2-0.15):FIT_PARAM_MED:FIT_PARAM_MED:FIT_PARAM_MED:FIT_PARAM_MED axis x1y1 notitle \
          with candlesticks linecolor rgbcolor "#AA6012" whiskerbars, \
     sprintf("%s.bb.%s", ARG1, ARG4) using ($2):FIT_PARAM_10P:FIT_PARAM_MIN:FIT_PARAM_MAX:FIT_PARAM_90P axis x1y1 \
          ti "BB" with candlesticks linecolor rgbcolor "#13FF03" whiskerbars, \
     "" using ($2):FIT_PARAM_MED:FIT_PARAM_MED:FIT_PARAM_MED:FIT_PARAM_MED axis x1y1 notitle \
          with candlesticks linecolor rgbcolor "#03AA00" whiskerbars \
# sprintf("%s.bb.%s", ARG1, ARG4) using ($2+0.15):FIT_PARAM_10P:FIT_PARAM_MIN:FIT_PARAM_MAX:FIT_PARAM_90P axis x1y2 \
#      ti "Branch&Bound" with candlesticks linecolor rgbcolor "#AAAAAA" whiskerbars, \
# "" using ($2+0.15):FIT_PARAM_MED:FIT_PARAM_MED:FIT_PARAM_MED:FIT_PARAM_MED axis x1y2 notitle \
#      with candlesticks linecolor rgbcolor "#000000" whiskerbars \


#     g(x,a_An,b_An,c_An,d_An,s_An) axis x1y2,\
#     g(x,a_Co,b_Co,c_Co,d_An,s_Co) axis x1y2

# plot sprintf("%s.data", ARG1) using 3:1 axis x1y2 with dots,\
#  '' using 3:1:(1000.0) axis x1y2 smooth acsplines, \
#  sprintf("%s.cold", ARG1) using 3:1 axis x1y2 with dots,\
#  '' using 3:1:(1000.0) axis x1y2 smooth acsplines, \
