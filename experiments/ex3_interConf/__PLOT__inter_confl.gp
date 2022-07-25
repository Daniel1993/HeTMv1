set terminal dumb

# ARG1 --> location of the data
# ARG2 --> output file

set grid ytics
set fit results

# stats <FILENAME_HERE> using <COL_NB_HERE> nooutput
# available: STATS_max, STATS_min (check other measures)

if (ARG2[strlen(ARG2)-2:] eq 'tex') {
    set terminal cairolatex size 2.80,2.0
    set output sprintf("%s", ARG2)
} else { if (ARG2[strlen(ARG2)-2:] eq 'pdf') {
    set terminal pdf size 5,4
    set output sprintf("%s", ARG2)
} else { if (ARG2[strlen(ARG2)-2:] eq 'jpg') {
    set terminal jpeg enhanced large size 800,560
    set output sprintf("%s", ARG2)
    # set xtics rotate by 20
} else {
    set terminal pngcairo noenhanced size 800,560
    set output sprintf("%s", ARG2)
}}}

set ylabel "Throughput" font ",14" tc lt 0 #offset 2.7,-0.0
set xlabel "Probability deterministic inter-conflict (%)" font ",14" tc lt 0 #offset 2.7,-0.0

plot \
     sprintf("%s/BMAP_1GPU.avg", ARG1)              using 45:18 notitle with linespoints linecolor rgbcolor "#FF0000" pt 1 lw 2 ps 0.8, \
     sprintf("%s/BMAP_2GPU.avg", ARG1)              using 45:18 notitle with linespoints linecolor rgbcolor "#BA009A" pt 2 lw 2 ps 0.8, \
     sprintf("%s/CPUonly.avg", ARG1) using 45:18 notitle with linespoints linecolor rgbcolor "#13FF03" dashtype '-' pt 6 lw 1 ps 0.8, \
     sprintf("%s/GPUonly.avg", ARG1) using 45:18 notitle with linespoints linecolor rgbcolor "#13C3C3" dashtype '-' pt 4 lw 1 ps 0.8, \
     sprintf("%s/SHeTM_basic.avg", ARG1) using 37:18 notitle with linespoints linecolor rgbcolor "#13DD56" pt 3 lw 1 ps 0.8, \
     sprintf("%s/SHeTM_opt.avg", ARG1)   using 37:18 notitle with linespoints linecolor rgbcolor "#569913" pt 5 lw 1 ps 0.8, \
     1/0 with linespoints linecolor rgbcolor "#FF0000"  pt 1 lw 3 ps 1 ti "SHeTM 1GPU", \
     1/0 with linespoints linecolor rgbcolor "#BA009A"  pt 2 lw 3 ps 1 ti "SHeTM 2GPU", \
     1/0 with linespoints linecolor rgbcolor "#13FF03"  dashtype '-' pt 6 lw 3 ps 1 ti "CPU only", \
     1/0 with linespoints linecolor rgbcolor "#13C3C3"  dashtype '-' pt 4 lw 3 ps 1 ti "GPU only", \
     1/0 with linespoints linecolor rgbcolor "#13FF56"  pt 3 lw 3 ps 1 ti "SHeTM basic", \
     1/0 with linespoints linecolor rgbcolor "#56AA13"  pt 5 lw 3 ps 1 ti "SHeTM opt" \


