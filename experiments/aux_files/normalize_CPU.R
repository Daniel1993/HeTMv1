#!/usr/bin/env Rscript

# ##################################################
# ### NORMALIZE TO THEORETICAL #####################
# ##################################################
# Usage: script.R <CPU-only> <CPU-VERS> <output_file>
# normalize the HeTM throughput to the sum of
# CPU-only + GPU-only
# output:
# > <HeTM>.norm
# ###########################################
# Tab delimited format
# ###########################################

arguments <- commandArgs(TRUE)
argc <- length(arguments)
CPUonly <- arguments[1]
CPU_VERS <- arguments[2]
output <- arguments[3]

if (argc != 3) {
  print("Wrong number of parameters\n")
  print("Usage: script.R <CPU-only> <CPU-VERS> <output_file>\n")
}

CPUonlyTSV <- read.csv(CPUonly, sep='\t')
VERS_TSV <- read.csv(CPU_VERS, sep='\t')

theoCPU <- CPUonlyTSV[,c("X.18.HeTM_THROUGHPUT.18.")] + CPUonlyTSV[,c("X.55.HeTM_THROUGHPUT.18.")]

xAxis <- t(rbind(
  "PREC_WRITE"=VERS_TSV[,c("X.37.PREC_WRITE")],                          # 1
  "CPU_AVG"=VERS_TSV[,c("X.18.HeTM_THROUGHPUT.18.")] / theoCPU,     # 2
  "CPU_STDDEV"=VERS_TSV[,c("X.55.HeTM_THROUGHPUT.18.")] / theoCPU,  # 3
  "CPU_THROUGHPUT"=theoCPU #
))

write.table(xAxis, output, sep="\t", row.names=FALSE, col.names=TRUE)

