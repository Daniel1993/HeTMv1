#include "memman.hpp"

#include <map>
#include <stack>
#include <tuple>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda.h>
#include "cuda_util.h"

using namespace std;

static memman::MemObjOnDev *dev_buffer;
static memman::Config* m_config;
thread_local static int m_selGPU;

int
memman::Config::GetNbPhysicalGPUs()
{
  int nbDevices;
  CUDA_CHECK_ERROR(cudaGetDeviceCount(&nbDevices), "");
  return nbDevices;
}

int
memman::Config::SelDev(
  int id
) {
  if (id < 0 || id >= m_NbGPUs)
  {
    fprintf(stderr, "Invalid device!\n");
    return -1;
  }
  memman::SelDev(id);
  m_selGPU = id;
  return id;
}

int
memman::Config::SelDev()
{
  return m_selGPU;
}

void
memman::SetNbGPUs(
  int nbGPUs
) {
  Config::GetInstance(nbGPUs);
}

int
memman::GetNbGpus()
{
  return Config::GetInstance()->NbGPUs();
}

int
memman::SelDev(
  int id
) {
  CUDA_CHECK_ERROR(cudaSetDevice(GetActualDev(id)), "");
  return id;
}

int
memman::GetCurrDev()
{
  return Config::GetInstance()->SelDev();
}

int
memman::GetActualDev(
  int devId
) {
  int nbDevices, deviceId;
  CUDA_CHECK_ERROR(cudaGetDeviceCount(&nbDevices), "");
  deviceId = devId % nbDevices;
  return deviceId;
}

memman::MemObjBuilder*
memman::MemObjBuilder::AllocHostPtr()
{
  CUDA_HOST_ALLOC(m_host, m_size);
  return this;
}

memman::MemObjBuilder*
memman::MemObjBuilder::AllocDevPtr()
{
  CUDA_DEV_ALLOC(m_dev, m_size);
  return this;
}

memman::MemObjBuilder*
memman::MemObjBuilder::AllocUnifMem()
{
  CUDA_UNIF_MEM_ALLOC(m_dev, m_size);
  m_host = m_dev;  
  return this; 
}

void
memman::MemObj::CpyHtD(
  void *strm
) {
  cudaStream_t s = (cudaStream_t)strm;
  assert(nullptr != host && "host ptr is not defined");
  assert(nullptr != dev  &&  "dev ptr is not defined");
  if (host != dev)
    if (strm)
      CUDA_CPY_TO_DEV_ASYNC(dev, host, size, s);
    else
      CUDA_CPY_TO_DEV(dev, host, size);
}

void
memman::MemObj::CpyDtH(
  void *strm
) {
  cudaStream_t s = (cudaStream_t)strm;
  assert(nullptr != host && "host ptr is not defined");
  assert(nullptr != dev  &&  "dev ptr is not defined");
  if (host != dev)
    if (strm)
      CUDA_CPY_TO_HOST_ASYNC(dev, host, size, s);
    else
      CUDA_CPY_TO_HOST(dev, host, size);
}

void
memman::MemObj::CpyDtD(
  MemObj *other,
  void *strm
) {
  cudaStream_t s = (cudaStream_t)strm;
  assert(other->size >= size && "other size is too small");
  assert(nullptr != dev         &&        "dev ptr is not defined");
  assert(nullptr != other->dev  && "other->dev ptr is not defined");
  int oId = GetActualDev(other->devId);
  int mId = GetActualDev(devId);
  if ( oId == mId )
    if (strm)
        CUDA_CPY_DtD_ASYNC(other->dev, dev, size, s);
      else
        CUDA_CPY_DtD(other->dev, dev, size);
  else
    if (strm)
        CUDA_CPY_PtP_ASYNC(other->dev, oId, dev, mId, size, s);
      else
        CUDA_CPY_PtP(other->dev, oId, dev, mId, size);
}

void 
memman::MemObj::ZeroDev(
  void *strm
)
{
  cudaStream_t s = (cudaStream_t)strm;
  if (strm)
    CUDA_CHECK_ERROR(cudaMemsetAsync(dev, 0, size, s), "");
  else
    CUDA_CHECK_ERROR(cudaMemset(dev, 0, size), "");
}

void 
memman::MemObj::ZeroHost(
  void *strm
)
{
  memset(host, 0, size);
}

memman::Config::Config(const int nbGPUs, const size_t bufSize)
  : m_NbGPUs(nbGPUs),
    m_bufSize(bufSize)
{ }

memman::Config*
memman::Config::GetInstance(
  const int nbGPUs, const size_t buffer_size
) {
  if ( m_config == nullptr )
  {
    m_config = new Config(nbGPUs, buffer_size);
    dev_buffer = new MemObjOnDev();
    for (int i = 0; i < nbGPUs; ++i)
    {
      memman::SelDev(i);
      MemObjBuilder b;
      dev_buffer->AddMemObj(new MemObj(b
        .SetSize(m_config->m_bufSize)
        ->SetOptions(0)
        ->AllocDevPtr(), i));
    }
  }
  return m_config;
}

void
memman::Config::DestroyInstance()
{
  if ( m_config )
  {
    delete m_config;
    m_config = nullptr;
  }
}

memman::Config::~Config()
{
  for (int i = 0; i < m_NbGPUs; ++i)
  {
    m_config->SelDev(i);
    MemObj *m = dev_buffer->GetMemObj(i);
    CUDA_CHECK_ERROR(cudaFree(m->host), "");
    CUDA_CHECK_ERROR(cudaFree(m->dev), "");
    printf(" <<<<< Dealloc  dev_buffer  %i >>>> \n", i);
    delete m;
  }
  delete dev_buffer;
}

size_t
memman::MemObjCpy::Cpy()
{
  void *strm = strm1;
  size_t sz_chunk = size_chunk*gran_apply;
  size_t ret = 0;
  assert(nullptr != dst && "dst not set!");
  assert(nullptr != src && "src not set!");
  if (force_filter)
  {
    int nbThrs = 256;

    if (cache)
    {
      int nbBlcks = (size_chunk + (nbThrs-1)) / nbThrs;
      
      size_t offset = 0;
      for (int i = 0; offset < dst->size; i++, offset += sz_chunk)
        if (BMAP_CHECK_POS(cache->host, i, filter_val))
        {
          ret += CpyFilterTemplate(strm, nbThrs, nbBlcks, offset,
            offset+sz_chunk > dst->size ? sz_chunk - (offset+sz_chunk - dst->size) : sz_chunk);
          strm = strm == strm1 ? strm2 : strm1;
        }
    }
    else
    {
      int nbBlcks = (dst->size + (nbThrs-1)) / nbThrs;
      ret += CpyFilterTemplate(strm1, nbThrs, nbBlcks, 0, dst->size);
    }
  }
  else if (cache)
  {
    size_t offset = 0, sz = 0, end = (dst->size + (sz_chunk-1)) / sz_chunk;
    int sect = 0;
    for (int i = 0; i < end; i++)
    {
      size_t blSz = offset+sz+size_chunk > dst->size ? sz_chunk - (end*sz_chunk - dst->size) : size_chunk;
      if (BMAP_CHECK_POS(cache->host, i, filter_val))
        sz += blSz;
      else
      {
        sect = i;
        if (sz)
        {
          ret += CpyContiguousTemplate(strm, offset, sz);
          strm = strm == strm1 ? strm2 : strm1;
          offset += sz;
          sz = 0;
        }
        offset += blSz; // ignore this block
      }
    }
    if (sz) // the last block is also flagged
      ret += CpyContiguousTemplate(strm, offset, sz);
  }

  return ret;
}

__global__ void
apply_BMAP_1B(
  unsigned char *bytes,
  unsigned char val,
  long bufferSize,
  unsigned char *dev_buffer_dst,
  unsigned char *dev_buffer_src
) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    // if (id == 0)
    //     printf("bufferSize = %li  GPUwsPtr = %p\n", bufferSize, bytes);
    if (id >= bufferSize) return;
    if (BMAP_CHECK_POS(bytes, id, val))
    {
        // printf("apply_BMAP: pos=%i dstVal=%i srcVal=%i\n", id, (int)dev_buffer_dst[id], (int)dev_buffer_src[id]);
        dev_buffer_dst[id] = dev_buffer_src[id];
    }
}

__global__ void
apply_BMAP_4B(
  unsigned char *bytes,
  unsigned char val,
  long bufferSize,
  int *dev_buffer_dst,
  int *dev_buffer_src
) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    // if (id == 0)
    //     printf("bufferSize = %li  GPUwsPtr = %p\n", bufferSize, bytes);
    if (id >= (bufferSize >> 2)) return; // bufferSize is *sizeof(int)
    if (BMAP_CHECK_POS(bytes, id, val))
    {
    //     printf("apply_BMAP: pos=%i dstVal=%i srcVal=%i\n", id, (int)dev_buffer_dst[id], (int)dev_buffer_src[id]);
        dev_buffer_dst[id] = dev_buffer_src[id];
    }
}

size_t
memman::MemObjCpyHtD::CpyFilterTemplate(
  void *strm,
  int nbThrs,
  int nbBlcks,
  size_t offset,
  size_t sz
) {

  // TODO: the offset has to have in account if it is dataset or BMAP
  int devId = dst->devId;
  void *d = (int*)(((uintptr_t)(dst->dev)) + offset);  
  void *s = (int*)(((uintptr_t)(src->host)) + offset);
  memman::Config::GetInstance()->SelDev(devId);
  memman::MemObj *m = dev_buffer->GetMemObj(devId);
  int *buf = (int*)m->dev;
  unsigned char fVal = (unsigned char)filter_val;
  unsigned char *f = (unsigned char*)filter->dev;

  CUDA_CPY_TO_DEV_ASYNC(buf, s, sz, strm);
  if (gran_apply == 1)
    apply_BMAP_1B<<<nbBlcks, nbThrs, 0, (cudaStream_t)strm>>>(f, fVal, sz, (unsigned char*)d, (unsigned char*)buf);
  else if (gran_apply == 4)
    apply_BMAP_4B<<<nbBlcks, nbThrs, 0, (cudaStream_t)strm>>>(f, fVal, sz, (int*)d, (int*)buf);
  // CUDA_CHECK_ERROR(cudaStreamSynchronize((cudaStream_t)strm), "");
  return sz;
}

size_t
memman::MemObjCpyHtD::CpyContiguousTemplate(
  void *strm,
  size_t offset,
  size_t size
) {
  int devId = dst->devId;
  memman::Config::GetInstance()->SelDev(devId);
  void *d = (void*)(((uintptr_t)(dst->dev)) + offset);  
  void *s = (void*)(((uintptr_t)(src->host)) + offset);
  if (strm)
    CUDA_CPY_TO_DEV_ASYNC(d, s, size, strm);
  else   
    CUDA_CPY_TO_DEV(d, s, size);
  return size;
}

size_t
memman::MemObjCpyDtH::CpyFilterTemplate(
  void *strm,
  int nbThrs,
  int nbBlcks,
  size_t offset,
  size_t sz
) {
  int devId = src->devId;
  memman::Config::GetInstance()->SelDev(devId);
  // TODO: the offset has to have in account if it is dataset or BMAP
  void *d = (int*)(((uintptr_t)(dst->host)) + offset);  
  void *s = (int*)(((uintptr_t)(src->dev)) + offset);
  memman::MemObj *m = dev_buffer->GetMemObj(devId);
  int *buf = (int*)m->dev;
  unsigned char fVal = (unsigned char)filter_val;
  unsigned char *f = (unsigned char*)filter->dev;

  CUDA_CPY_TO_DEV_ASYNC(buf, d, sz, strm);
  if (gran_apply == 1)
    apply_BMAP_1B<<<nbBlcks, nbThrs, 0, (cudaStream_t)strm>>>(f, fVal, sz, (unsigned char*)buf, (unsigned char*)s);
  else if (gran_apply == 4)
    apply_BMAP_4B<<<nbBlcks, nbThrs, 0, (cudaStream_t)strm>>>(f, fVal, sz, (int*)buf, (int*)s);
  // CUDA_CHECK_ERROR(cudaStreamSynchronize((cudaStream_t)strm), "");
  CUDA_CPY_TO_HOST_ASYNC(d, buf, sz, strm);
  return sz;
}

size_t
memman::MemObjCpyDtH::CpyContiguousTemplate(
  void *strm,
  size_t offset,
  size_t size
) {
  int devId = src->devId;
  memman::Config::GetInstance()->SelDev(devId);
  void *d = (int*)(((uintptr_t)(dst->host)) + offset);  
  void *s = (int*)(((uintptr_t)(src->dev)) + offset);
  if (strm)
    CUDA_CPY_TO_HOST_ASYNC(d, s, size, strm);
  else   
    CUDA_CPY_TO_HOST(d, s, size);
  return size;
}

size_t 
memman::MemObjCpyDtD::CpyFilterTemplate(
  void *strm,
  int nbThrs,
  int nbBlcks,
  size_t offset,
  size_t sz
) {
  int srcId = src->devId;
  int dstId = dst->devId;
  memman::Config::GetInstance()->SelDev(dstId);
  void *d = (int*)(((uintptr_t)(dst->dev)) + offset);  
  void *s = (int*)(((uintptr_t)(src->dev)) + offset);
  memman::MemObj *m = dev_buffer->GetMemObj(dstId);
  int *buf = (int*)m->dev;
  unsigned char fVal = (unsigned char)filter_val;
  unsigned char *f = (unsigned char*)filter->dev;

  int aSrcId = GetActualDev(srcId);
  int aDstId = GetActualDev(dstId);

  if (aSrcId != aDstId)
    CUDA_CPY_PtP_ASYNC(buf, dstId, s, srcId, sz, strm);
  else
    CUDA_CPY_DtD_ASYNC(buf, s, sz, strm);

  if (gran_apply == 1)
    apply_BMAP_1B<<<nbBlcks, nbThrs, 0, (cudaStream_t)strm>>>(f, fVal, sz, (unsigned char*)d, (unsigned char*)buf);
  else if (gran_apply == 4)
    apply_BMAP_4B<<<nbBlcks, nbThrs, 0, (cudaStream_t)strm>>>(f, fVal, sz, (int*)d, (int*)buf);
  // CUDA_CHECK_ERROR(cudaStreamSynchronize((cudaStream_t)strm), "");
  return sz;
}

size_t
memman::MemObjCpyDtD::CpyContiguousTemplate(
  void *strm,
  size_t offset,
  size_t size
) {
  void *d = (void*)(((uintptr_t)(dst->dev)) + offset);  
  void *s = (void*)(((uintptr_t)(src->dev)) + offset);
  int srcId = src->devId;
  int dstId = dst->devId;
  int aSrcId = GetActualDev(srcId);
  int aDstId = GetActualDev(dstId);

  Config::GetInstance()->SelDev(dstId);

  if (strm)
    if (aSrcId == aDstId)
      CUDA_CPY_DtD_ASYNC(d, s, size, strm);
    else
      CUDA_CPY_PtP_ASYNC(d, aDstId, s, aSrcId, size, strm);
  else
    if (aSrcId == aDstId)
      CUDA_CPY_DtD(d, s, size);
    else
      CUDA_CPY_PtP(d, aDstId, s, aSrcId, size);
  return size;
}



