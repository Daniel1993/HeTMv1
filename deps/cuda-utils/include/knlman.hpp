#ifndef KNLMAN_H_GUARD_
#define KNLMAN_H_GUARD_

#include "memman.hpp"

#define DIM3(x_, y_, z_) (knlman_dim3_s){.x = x_,  .y = y_, .z = z_}

typedef struct knlman_dim3_ {
  int x, y, z;
} knlman_dim3_s;

typedef struct knlman_callback_params_ {
  knlman_dim3_s blocks;
  knlman_dim3_s threads;
  int devId;
  void *stream;
  memman::MemObj *entryObj;
  int currDev;
} knlman_callback_params_s;

// callback type
typedef void(*knlman_callback)(knlman_callback_params_s);
// MemObj* --> knlman_callback_params_s

enum {
  KNLMAN_NONE  = 0b0000,
  KNLMAN_COPY  = 0b0001,
  KNLMAN_DtoH  = 0b0010,
  KNLMAN_HtoD  = 0b0100
};

namespace knlman
{

using namespace memman;

class KnlObjBuilder
{
  public:
    KnlObjBuilder()
      : m_blocks ((knlman_dim3_s){ .x = 0, .y = 0, .z = 0}),
        m_threads((knlman_dim3_s){ .x = 0, .y = 0, .z = 0}),
        m_callback(nullptr),
        m_entryObj(nullptr)
    { }
    // MemObjBuilder(MemObjBuilder &other); // implicit?

    KnlObjBuilder *SetBlocks  (knlman_dim3_s blks) { m_blocks   = blks; return this; }
    KnlObjBuilder *SetThreads (knlman_dim3_s thrs) { m_threads  = thrs; return this; }
    KnlObjBuilder *SetCallback(knlman_callback  c) { m_callback = c;    return this; }
    KnlObjBuilder *SetEntryObj(MemObjOnDev   *mem) { m_entryObj = mem;  return this; }

  private:
    knlman_dim3_s m_blocks;
    knlman_dim3_s m_threads;
    knlman_callback m_callback;
    MemObjOnDev *m_entryObj;

    friend class KnlObj;
};

class KnlObj
{
  public:
    KnlObj(KnlObjBuilder *b)
      : blocks(b->m_blocks),
        threads(b->m_threads),
        callback(b->m_callback),
        entryObj(b->m_entryObj)
    { }

    void Run(int devId, void *strm = nullptr);

    knlman_dim3_s blocks;
    knlman_dim3_s threads;
    knlman_callback callback;
    MemObjOnDev *entryObj;
};

};

#endif