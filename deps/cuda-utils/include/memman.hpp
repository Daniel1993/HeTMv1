#ifndef MEMMAN_H_GUARD_
#define MEMMAN_H_GUARD_

#include <cstdlib>
#include <cstdint>
#include <string>
#include <cassert>
#include <vector>

#include "bitmap.hpp"
#include "cuda_util.h"

#ifndef HETM_NB_DEVICES
#define HETM_NB_DEVICES 2
#endif

#ifndef CHUNK_GRAN
#define CHUNK_GRAN (1<<20)
#endif

enum {
  MEMMAN_NONE  = 0b00000,
  MEMMAN_ASYNC = 0b00001,
  MEMMAN_THRLC = 0b00010, // thread local memory
  MEMMAN_COPY  = 0b00100,
  MEMMAN_UNIF  = 0b01000, // unified memory
  MEMMAN_BMAP  = 0b10000, // only copy some blocks (HtD)
  MEMMAN_BMAP_DtH = 0b100000, // same as MEMMAN_BMAP but DtH
  MEMMAN_FAKE = 0b100000 // same as MEMMAN_BMAP but DtH
};

namespace memman
{

class Config
{
  protected:
    Config(const int NbGPUs, const size_t bufSize);
    ~Config();
    int m_NbGPUs;
    size_t m_bufSize;

  public:
    Config(Config &other) = delete;
    void operator=(const Config &) = delete;
    static Config *GetInstance(const int NbGPUs = HETM_NB_DEVICES, const size_t buffer_size = CHUNK_GRAN);
    static void DestroyInstance();
    int NbGPUs() const { return m_NbGPUs; }
    int GetNbPhysicalGPUs();
    int SelDev(int id); // set
    int SelDev(); // get
};

void SetNbGPUs(int nbGPUs);
int GetNbGpus();
int SelDev(int id); // returns actual device
int GetCurrDev();
int GetActualDev(int devId);

class MemObjBuilder
{
  public:
    MemObjBuilder() : m_size(0), m_options(0), m_host(NULL), m_dev(NULL) { }
    // MemObjBuilder(MemObjBuilder &other); // implicit?

    MemObjBuilder *SetSize(size_t size){ m_size    = size;                return this; }
    MemObjBuilder *SetOptions(int opt) { m_options = opt;                 return this; }
    MemObjBuilder *SetHostPtr(void *h) { m_host    = h;                   return this; }
    MemObjBuilder *SetDevPtr(void *d)  { m_dev     = d;                   return this; }
    MemObjBuilder *AllocHostPtr();
    MemObjBuilder *AllocDevPtr();
    MemObjBuilder *AllocUnifMem();

  private:
    size_t m_size;
    int m_options;
    void *m_host;
    void *m_dev;

    friend class MemObj;
};

class MemObj
{
  public:
    MemObj(MemObjBuilder *b, int dId)
      : devId(dId),
        size(b->m_size),
        options(b->m_options),
        host(b->m_host),
        dev(b->m_dev)
    { }

    void CpyHtD(void *strm = nullptr);
    void CpyDtH(void *strm = nullptr);
    void CpyDtD(MemObj *other, void *strm = nullptr);

    void ZeroDev(void *strm = nullptr);
    void ZeroHost(void *strm = nullptr);

    int devId;
    size_t size;
    int options;
    void *host;
    void *dev;
};

class MemObjOnDev
{
  public:
    MemObjOnDev() { m_PerDev.resize(HETM_NB_DEVICES); }

    void AddMemObj(MemObj *mem)
    {
      assert(nullptr != mem && "mem not defined");
      m_PerDev.insert(m_PerDev.begin() + mem->devId, mem);
    }

    MemObj *GetMemObj(int devId = 0)
    {
      return m_PerDev[devId];
    }

  private:
    std::vector<MemObj*> m_PerDev;
};

class MemObjCpyBuilder
{
  public:
    MemObjCpyBuilder()
      : m_gran_filter(8),
        m_gran_apply(4),
        m_force_filter(1),
        m_filter_val(1),
        m_cache(nullptr),
        m_filter(nullptr),
        m_dst(nullptr),
        m_src(nullptr),
        m_size_chunk(CHUNK_GRAN),
        m_strm1(nullptr),
        m_strm2(nullptr)
    { }

    MemObjCpyBuilder *SetGranFilter (int size) { m_gran_filter = size; return this; }
    MemObjCpyBuilder *SetGranApply  (int size) { m_gran_apply  = size; return this; }
    MemObjCpyBuilder *SetForceFilter(int fltr) { m_force_filter= fltr; return this; }
    MemObjCpyBuilder *SetFilterVal  (int val)  { m_filter_val  = val;  return this; }
    MemObjCpyBuilder *SetCache      (MemObj*c) { m_cache       = c;    return this; }
    MemObjCpyBuilder *SetFilter     (MemObj*f) { m_filter      = f;    return this; }
    MemObjCpyBuilder *SetDst        (MemObj*d) { m_dst         = d;    return this; }
    MemObjCpyBuilder *SetSrc        (MemObj*s) { m_src         = s;    return this; }
    MemObjCpyBuilder *SetSizeChunk  (size_t s) { m_size_chunk  = s;    return this; }
    MemObjCpyBuilder *SetStrm1      (void *s1) { m_strm1       = s1;   return this; }
    MemObjCpyBuilder *SetStrm2      (void *s2) { m_strm2       = s2;   return this; }

  private:
    int m_gran_filter; // BMAP uses 8 bits, set to 1 to use single bit
    int m_gran_apply;  // Number of bytes to apply (default 4)
    int m_force_filter;// force apply ONLY changed regions
    int m_filter_val;  // value to find in the filter in order to apply
    MemObj *m_cache;   // cache of the filter in host memory
    MemObj *m_filter;  // location of the filter 
    MemObj *m_dst;     // destiny buffer
    MemObj *m_src;     // source buffer
    size_t m_size_chunk;// granularity of the cache
    void *m_strm1;     // optional stream to interleave copies
    void *m_strm2;     // optional stream to interleave copies

  friend class MemObjCpy;
};

class MemObjCpy
{
  public:
    MemObjCpy(MemObjCpyBuilder *b)
    : gran_filter(b->m_gran_filter),
      gran_apply(b->m_gran_apply),
      force_filter(b->m_force_filter),
      filter_val(b->m_filter_val),
      cache(b->m_cache),
      filter(b->m_filter),
      dst(b->m_dst),
      src(b->m_src),
      size_chunk(b->m_size_chunk),
      strm1(b->m_strm1),
      strm2(b->m_strm2)
    { }

    size_t Cpy();

  protected:
    virtual size_t CpyFilterTemplate(void *strm, int nbThrs, int nbBlcks, size_t offset, size_t size)
      { assert(0 && "Cannot go through here!"); return 0; }
    virtual size_t CpyContiguousTemplate(void *strm, size_t offset, size_t size)
      { assert(0 && "Cannot go through here!"); return 0; }

    int gran_filter;
    int gran_apply;
    int force_filter;
    int filter_val;
    MemObj *cache;
    MemObj *filter;
    MemObj *dst;
    MemObj *src;
    size_t size_chunk;
    void *strm1;
    void *strm2;
};

class MemObjCpyHtD : public MemObjCpy
{
  public:
    MemObjCpyHtD(MemObjCpyBuilder *b) : MemObjCpy(b) { };

  protected:
    size_t CpyFilterTemplate(void *strm, int nbThrs, int nbBlcks, size_t offset, size_t size);
    size_t CpyContiguousTemplate(void *strm, size_t offset, size_t size);
};

class MemObjCpyDtH : public MemObjCpy
{
  public:
    MemObjCpyDtH(MemObjCpyBuilder *b) : MemObjCpy(b) { };

  protected:
    size_t CpyFilterTemplate(void *strm, int nbThrs, int nbBlcks, size_t offset, size_t size);
    size_t CpyContiguousTemplate(void *strm, size_t offset, size_t size);
};

class MemObjCpyDtD : public MemObjCpy
{
  public:
    MemObjCpyDtD(MemObjCpyBuilder *b) : MemObjCpy(b) { };

  protected:
    size_t CpyFilterTemplate(void *strm, int nbThrs, int nbBlcks, size_t offset, size_t size);
    size_t CpyContiguousTemplate(void *strm, size_t offset, size_t size);
};

};

#endif /* MEMMAN_H_GUARD_ */
