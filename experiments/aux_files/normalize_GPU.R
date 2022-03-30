#!/usr/bin/env Rscript

# ##################################################
# ### NORMALIZE TO THEORETICAL #####################
# ##################################################
# Usage: script.R <GPU-only> <GPU-BMAP> <GPU-BMAP-NO-RS> <GPU-BMAP-NO-WS> <output_file>
# normalize the HeTM throughput to the sum of
# CPU-only + GPU-only
# output:
# > <HeTM>.norm
# ###########################################
# Tab delimited format
# ###########################################

arguments <- commandArgs(TRUE)
argc <- length(arguments)
GPUonly <- arguments[1]
GPU_BMAP <- arguments[2]
GPU_BMAP_NO_RS <- arguments[3]
GPU_BMAP_NO_WS <- arguments[4]
output <- arguments[5]

if (argc != 5) {
  print("Wrong number of parameters\n")
  print("Usage: script.R <GPU-only> <GPU-BMAP> <GPU-BMAP-NO-RS> <GPU-BMAP-NO-WS> <output_file>\n")
}

GPUonlyTSV <- read.csv(GPUonly, sep='\t')
BMAP_TSV <- read.csv(GPU_BMAP, sep='\t')
BMAP_NO_RS_TSV <- read.csv(GPU_BMAP_NO_RS, sep='\t')
BMAP_NO_WS_TSV <- read.csv(GPU_BMAP_NO_WS, sep='\t')

theoGPU <- GPUonlyTSV[,c("X.18.HeTM_THROUGHPUT.18.")] 
theoGPU_k <- GPUonlyTSV[,c("X.17.Kernel_THROUGHPUT.17.")] + GPUonlyTSV[,c("X.54.Kernel_THROUGHPUT.17.")]

xAxis <- t(rbind(
  "PREC_WRITE"=BMAP_TSV[,c("X.37.PREC_WRITE")],                          # 1
  "GPU_BMAP_AVG"=BMAP_TSV[,c("X.18.HeTM_THROUGHPUT.18.")] / theoGPU,     # 2
  "GPU_BMAP_STDDEV"=BMAP_TSV[,c("X.55.HeTM_THROUGHPUT.18.")] / theoGPU,  # 3
  "GPU_BMAP_NO_RS_AVG"=BMAP_NO_RS_TSV[,c("X.18.HeTM_THROUGHPUT.18.")] / theoGPU,     # 4
  "GPU_BMAP_NO_RS_STDDEV"=BMAP_NO_RS_TSV[,c("X.55.HeTM_THROUGHPUT.18.")] / theoGPU,  # 5
  "GPU_BMAP_NO_WS_AVG"=BMAP_NO_WS_TSV[,c("X.18.HeTM_THROUGHPUT.18.")] / theoGPU,     # 6
  "GPU_BMAP_NO_WS_STDDEV"=BMAP_NO_WS_TSV[,c("X.55.HeTM_THROUGHPUT.18.")] / theoGPU,  # 7
  "GPU_k_BMAP_AVG"=BMAP_TSV[,c("X.17.Kernel_THROUGHPUT.17.")] / theoGPU_k,     # 8
  "GPU_k_BMAP_STDDEV"=BMAP_TSV[,c("X.54.Kernel_THROUGHPUT.17.")] / theoGPU_k,  # 9
  "GPU_k_BMAP_NO_RS_AVG"=BMAP_NO_RS_TSV[,c("X.17.Kernel_THROUGHPUT.17.")] / theoGPU_k,     # 10
  "GPU_k_BMAP_NO_RS_STDDEV"=BMAP_NO_RS_TSV[,c("X.54.Kernel_THROUGHPUT.17.")] / theoGPU_k,  # 11
  "GPU_k_BMAP_NO_WS_AVG"=BMAP_NO_WS_TSV[,c("X.17.Kernel_THROUGHPUT.17.")] / theoGPU_k,     # 12
  "GPU_k_BMAP_NO_WS_STDDEV"=BMAP_NO_WS_TSV[,c("X.54.Kernel_THROUGHPUT.17.")] / theoGPU_k,  # 13
  "GPU_THROUGHPUT"=theoGPU # 14
))

write.table(xAxis, output, sep="\t", row.names=FALSE, col.names=TRUE)

