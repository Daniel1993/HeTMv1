#include "prod-cons.h"

#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

#define BUFFER_EVENTS        0x80
#define MAX_FREE_NODES       0x100

// -------------------- // TODO: organize them
// Functions
static void* offloadThread(void*);
static unsigned long freeNodesPtr, freeNodesEndPtr;
static prod_cons_async_req_s reqsBuffer[MAX_FREE_NODES];
// --------------------

// --------------------
// Variables
static pthread_t async_thread;
static int is_stop, is_created;
static prod_cons_s *pc;
// --------------------

int prod_cons_start_thread()
{
  is_stop = 0;
  pc = prod_cons_init(BUFFER_EVENTS);
  if (!is_created) {
    thread_create_or_die(&async_thread, NULL, offloadThread, NULL);
    is_created = 1;
  }
  return 0;
}

int prod_cons_join_thread()
{
  is_stop = 1;
  thread_join_or_die(async_thread, NULL);
  is_created = 0;
  return 0;
}

void prod_cons_async_request(prod_cons_async_req_s req)
{
  prod_cons_async_req_s *m_req;
  long idx = PROD_CONS_ATOMIC_INC_PTR(freeNodesPtr, MAX_FREE_NODES);
  // wait release space (else use a malloc/free solution)
  reqsBuffer[idx].fn   = req.fn;
  reqsBuffer[idx].args = req.args;
  m_req = &reqsBuffer[idx];
  prod_cons_produce(pc, m_req);
}

// Producer-Consumer thread: consumes requests from the other ones
static void* offloadThread(void *arg)
{
  prod_cons_async_req_s *req;

  while (!is_stop || !prod_cons_is_empty(pc)) {
    prod_cons_consume(pc, (void**)&req);
    PROD_CONS_ATOMIC_INC_PTR(freeNodesEndPtr, MAX_FREE_NODES);
    req->fn(req->args);
    // if malloc'ing the request free it here
  }
  return NULL;
}
