#!/usr/bin/env Rscript

# ##################################################
# ### NORMALIZE TO THEORETICAL #####################
# ##################################################
# Usage: script.R <CPU-only> <GPU-only> <sol> <output>
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
GPUonly <- arguments[2]
SOL <- arguments[3]
XAxisName <- arguments[4]
output <- arguments[5]

if (argc != 5) {
  print("Wrong number of parameters\n")
  print("Usage: script.R <CPU-only> <GPU-only> <SOL> <X-AXIS> <output_file>\n")
}

CPUonlyTSV <- read.csv(CPUonly, sep='\t')
GPUonlyTSV <- read.csv(GPUonly, sep='\t')
SOL_TSV <- read.csv(SOL, sep='\t')

theoCPU <- CPUonlyTSV[,c("X.18.HeTM_THROUGHPUT.18.")]
theoGPU <- GPUonlyTSV[,c("X.18.HeTM_THROUGHPUT.18.")]

xAxis <- t(rbind(
  "PREC_WRITE"=SOL_TSV[,c(XAxisName)],                          # 1
  "CPU_SOL_AVG"=SOL_TSV[,c("X.15.CPU_THROUGHPUT.15.")] / theoCPU,     # 2
  "CPU_SOL_STDDEV"=SOL_TSV[,c("X.60.CPU_THROUGHPUT.15.")] / theoCPU,  # 3
  "GPU_SOL_AVG"=SOL_TSV[,c("X.16.GPU_THROUGHPUT.16.")] / theoGPU,     # 2
  "GPU_SOL_STDDEV"=SOL_TSV[,c("X.61.GPU_THROUGHPUT.16.")] / theoGPU,  # 3
  "SUM_AVG"=SOL_TSV[,c("X.18.HeTM_THROUGHPUT.18.")] / (theoCPU+theoGPU),    # 4
  "SUM_STDDEV"=SOL_TSV[,c("X.63.HeTM_THROUGHPUT.18.")] / (theoCPU+theoGPU), # 5
  "CPU_THROUGHPUT"=theoCPU, #
  "GPU_THROUGHPUT"=theoGPU #
))

write.table(xAxis, output, sep="\t", row.names=FALSE, col.names=TRUE)

