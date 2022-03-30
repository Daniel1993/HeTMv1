#!/usr/bin/env Rscript

# ##################################################
# ### NORMALIZE TO THEORETICAL #####################
# ##################################################
# Usage: script.R <CPU-only> <GPU-only> <CPU-VERS> <GPU-BMAP> <output_file>
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
CPU_VERS <- arguments[3]
CPU_BMAP <- arguments[4]
GPU_BMAP <- arguments[5]
output <- arguments[6]

if (argc != 6) {
  print("Wrong number of parameters\n")
  print("Usage: script.R <CPU-only> <GPU-only> <CPU-VERS> <CPU-BMAP> <GPU-BMAP> <output_file>\n")
}

CPUonlyTSV <- read.csv(CPUonly, sep='\t')
GPUonlyTSV <- read.csv(GPUonly, sep='\t')
GPU_BMAP_TSV <- read.csv(GPU_BMAP, sep='\t')
VERS_TSV <- read.csv(CPU_VERS, sep='\t')
BMAP_TSV <- read.csv(CPU_BMAP, sep='\t')

theoCPU <- CPUonlyTSV[,c("X.18.HeTM_THROUGHPUT.18.")]
theoGPU <- GPUonlyTSV[,c("X.17.Kernel_THROUGHPUT.17.")]

xAxis <- t(rbind(
  "PREC_WRITE"=VERS_TSV[,c("X.37.PREC_WRITE")],                          # 1
  "CPU_BMAP_AVG"=BMAP_TSV[,c("X.18.HeTM_THROUGHPUT.18.")] / theoCPU,     # 2
  "CPU_BMAP_STDDEV"=BMAP_TSV[,c("X.55.HeTM_THROUGHPUT.18.")] / theoCPU,  # 3
  "CPU_VERS_AVG"=VERS_TSV[,c("X.18.HeTM_THROUGHPUT.18.")] / theoCPU,     # 2
  "CPU_VERS_STDDEV"=VERS_TSV[,c("X.55.HeTM_THROUGHPUT.18.")] / theoCPU,  # 3
  "GPU_BMAP_AVG"=GPU_BMAP_TSV[,c("X.17.Kernel_THROUGHPUT.17.")] / theoGPU,    # 4
  "GPU_BMAP_STDDEV"=GPU_BMAP_TSV[,c("X.54.Kernel_THROUGHPUT.17.")] / theoGPU, # 5
  "CPU_THROUGHPUT"=theoCPU, #
  "GPU_THROUGHPUT"=theoGPU #
))

write.table(xAxis, output, sep="\t", row.names=FALSE, col.names=TRUE)

