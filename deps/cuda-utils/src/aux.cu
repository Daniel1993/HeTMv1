#include "memman.hpp"
#include "cuda_util.h"
#include "stdio.h"
#include "stdlib.h"

static int nb_devs;
static void **dev_buffer;
static long buffer_size;

using namespace memman;

__global__ void apply_BMAP(char *bytes, char val, long bufferSize, int *dev_buffer_dst, int *dev_buffer_src)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    // if (id == 0)
    //     printf("bufferSize = %li  GPUwsPtr = %p\n", bufferSize, bytes);
    if (id >= bufferSize) return;
    if (bytes[id] == val)
    {
        // printf("apply_BMAP: pos=%i dstVal=%i srcVal=%i\n", id, (int)dev_buffer_dst[id], (int)dev_buffer_src[id]);
        dev_buffer_dst[id] = dev_buffer_src[id];
    }
}

void
memman_apply_BMAP_init(long bufferSize)
{
    buffer_size = (bufferSize / sizeof(int)) * sizeof(int);
    nb_devs = Config::GetInstance()->NbGPUs();
    dev_buffer = (void**)malloc(sizeof(void*)*nb_devs);
    for (int i = 0; i < nb_devs; ++i)
    {
        Config::GetInstance()->SelDev(i);
        CUDA_DEV_ALLOC(dev_buffer[i], buffer_size);
    }
}

void
memman_apply_BMAP_destroy()
{
    for (int i = 0; i < nb_devs; ++i)
    {
        Config::GetInstance()->SelDev(i);
        cudaFree(dev_buffer[i]);
    }
    free(dev_buffer);
}

long
memman_apply_BMAP_get_buffer_size()
{
    return buffer_size;
}

#define start_fn \
    int nbThrs = 256; \
    int nbBlcks = size / nbThrs + (size % nbThrs != 0 ? 1 : 0); \
    if (size > buffer_size) { \
        fprintf(stderr, "Overflow!\n"); return; \
    } \
//

void
memman_apply_BMAP_DtH(int devId, void* strm, long size, char *bytes, char val, int *hostPtr, int *devPtr)
{
    start_fn

    CUDA_CPY_TO_DEV_ASYNC(dev_buffer[devId], hostPtr, size, (cudaStream_t)strm);
    apply_BMAP<<<nbBlcks, nbThrs, 0, (cudaStream_t)strm>>>(bytes, val, size, (int*)dev_buffer[devId], devPtr);
    CUDA_CPY_TO_HOST_ASYNC(hostPtr, dev_buffer[devId], size, (cudaStream_t)strm);
}

void
memman_apply_BMAP_HtD(int devId, void* strm, long size, char *bytes, char val, int *devPtr, int *hostPtr)
{
    start_fn

    CUDA_CPY_TO_DEV_ASYNC(dev_buffer[devId], hostPtr, size, (cudaStream_t)strm);
    apply_BMAP<<<nbBlcks, nbThrs, 0, (cudaStream_t)strm>>>(bytes, val, size, devPtr, (int*)dev_buffer[devId]);
    // printf("memman_apply_BMAP_HtD\n");
}

void
memman_apply_BMAP_DtD(int devId, void* strm, long size, char *bytes, char val, int *devPtrDst, int *devPtrSrc)
{
    start_fn

    apply_BMAP<<<nbBlcks, nbThrs, 0, (cudaStream_t)strm>>>(bytes, val, size, devPtrDst, devPtrSrc);
}

void
memman_apply_BMAP_PtP(int devDst, int devSrc, void* strm, long size, char *bytes, char val, int *devPtrDst, int *devPtrSrc)
{
    int aDevDst = GetActualDev(devDst);
    int aDevSrc = GetActualDev(devSrc);
    if (aDevSrc == aDevDst)
    {
        memman_apply_BMAP_DtD(aDevSrc, strm, size, bytes, val, devPtrDst, devPtrSrc);
        return;
    }

    start_fn

    // copy all changes to remote
    CUDA_CPY_PtP_ASYNC(
        dev_buffer[devDst], aDevDst,
        devPtrSrc, aDevSrc,
        size,
        (cudaStream_t)strm
    );
    CUDA_CHECK_ERROR(cudaSetDevice(aDevDst), "");
    // apply fine-grained
    apply_BMAP<<<nbBlcks, nbThrs, 0, (cudaStream_t)strm>>>(bytes, val, size, devPtrDst, (int*)dev_buffer[devDst]);
}
