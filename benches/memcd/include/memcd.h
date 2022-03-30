#ifndef MEMCD_H_GUARD
#define MEMCD_H_GUARD

#ifdef __cplusplus
extern "C" {
#endif
	void GPUbufferReadFromFile_NO_CONFLS();
	void GPUbufferReadFromFile_CONFLS();
  void GPUbufferReadFromFile_UNIF_RAND();

  void CPUbufferReadFromFile_NO_CONFLS();
  void CPUbufferReadFromFile_CONFLS();
  void CPUbufferReadFromFile_UNIF_RAND();

  void GPUbuffer_NO_CONFLS();
  void CPUbuffer_NO_CONFLS();

  void GPUbuffer_UNIF_2();
  void CPUbuffer_UNIF_2();

  void GPUbuffer_ZIPF_2();
  void CPUbuffer_ZIPF_2();
#ifdef __cplusplus
}
#endif

extern thread_data_t parsedData;
extern int isInterBatch;
extern size_t accountsSize;
extern size_t sizePool;
extern void* gpuMempool;

const static int NB_OF_GPU_BUFFERS = 64; // GPU receives some more space
const static int NB_CPU_TXS_PER_THREAD = 16384;

extern size_t currMaxCPUoutputBufferSize;
extern size_t currCPUoutputBufferPtr;
extern size_t maxGPUoutputBufferSize;
extern size_t size_of_GPU_input_buffer;
extern size_t size_of_CPU_input_buffer;
extern int lockOutputBuffer;

extern FILE *GPU_input_file;
extern FILE *CPU_input_file;

#endif /* MEMCD_H_GUARD */
