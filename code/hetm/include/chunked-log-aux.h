#ifndef CHUNKED_LOG_AUX_H_GUARD_
#define CHUNKED_LOG_AUX_H_GUARD_

#define CHUNKED_LOG_INIT_NODE(ptr) \
  (ptr)->first = NULL; \
  (ptr)->curr = NULL; \
  (ptr)->last = NULL; \
  (ptr)->trunc = NULL; \
  (ptr)->size = 0; \
  (ptr)->pos = 0 \
//

#define CHUNKED_LOG_LOCAL_INST(var) \
  chunked_log_s var; \
  CHUNKED_LOG_INIT_NODE(&var) \
//

/**
 * Extends the log with a new chunk.
 */
#define CHUNKED_LOG_EXTEND(log, node) ({ \
  (node)->next = (node)->prev = NULL; \
  if ((log)->first == NULL || (log)->last == NULL) { \
    (log)->first = (log)->curr = (log)->last = node; \
    (log)->size = 1; \
    (log)->pos = 0; \
  } else { \
    (log)->last->next = node; \
    (node)->prev = (log)->last; \
    (log)->last = node; \
    (log)->size++; \
  } \
})

/**
 * Extends the truncated log with the next chunk.
 */
#define CHUNKED_LOG_TRUNC_EXTEND(devId, log) ({ \
  int _idx = devId; \
  if ((log)->trunc[_idx] != NULL && (log)->trunc[_idx]->curr != NULL && (log)->trunc[_idx]->curr->next != NULL) { \
    truncated_chunked_log_node_s *_tn = (truncated_chunked_log_node_s*)malloc(sizeof(truncated_chunked_log_node_s)); \
    (log)->truncLast[_idx]->next = _tn; \
    _tn->next = NULL; \
    _tn->prev = (log)->truncLast[_idx]; \
    _tn->curr = (log)->truncLast[_idx]->curr->next; \
    (log)->truncLast[_idx] = _tn; \
  } \
  (log)->nextTruncated[devId] = ((log)->truncLast[_idx] != NULL && (log)->truncLast[_idx]->curr != NULL) ? (log)->truncLast[_idx]->curr->next : NULL; \
})

#define CHUNKED_LOG_REMOVE_FRONT(log) ({ \
  if ((log)->first == (log)->last) { \
    (log)->last = (log)->first = NULL; \
    (log)->size = 0; \
  } else { \
    (log)->first->next->prev = NULL; \
    (log)->first = (log)->first->next; \
  } \
})

#define CHUNKED_LOG_TRUNC_PREPARE(devId, log) ({ \
  chunked_log_node_s *_node = NULL; \
  int _idx = devId; \
  if ((log)->trunc[_idx] != NULL) { \
    _node = (log)->truncLast[_idx]->curr->next; \
    while ((log)->trunc[_idx] != (log)->truncLast[_idx]) { \
      truncated_chunked_log_node_s *_nt; \
      _nt = (log)->trunc[_idx]->next; \
      free((log)->trunc[_idx]); \
      (log)->trunc[_idx] = _nt; \
    } \
    (log)->trunc[_idx]->curr = _node; \
    (log)->trunc[_idx]->prev = (log)->trunc[_idx]->next = NULL; \
  } else { \
    _node = (log)->nextTruncated[devId] == NULL ? (log)->first : (log)->nextTruncated[devId]; \
    (log)->trunc[_idx] = (truncated_chunked_log_node_s*)malloc(sizeof(truncated_chunked_log_node_s)); \
    (log)->truncLast[_idx] = (log)->trunc[_idx]; \
    (log)->trunc[_idx]->curr = _node; \
    (log)->trunc[_idx]->prev = (log)->trunc[_idx]->next = NULL; \
  } \
  if (_node != NULL && _node->next != NULL) { \
    (log)->nextTruncated[devId] = _node->next; \
  } \
  if ((log)->nextTruncated[devId] == NULL) { \
    (log)->nextTruncated[devId] = (log)->first; \
  } \
  (log)->truncLast[_idx] = (log)->trunc[_idx]; \
  _node; \
})

// in case of warp --> i > chunked_log_free_ptr + SIZE_OF_FREE_NODES
#define CHUNKED_LOG_FIND_FREE(sizeNode, nbBuckets) ({ \
  chunked_log_node_s *res_ = NULL; \
  unsigned long i = chunked_log_alloc_ptr; \
  if (i < chunked_log_free_ptr) { \
    res_ = chunked_log_node_recycled[i % SIZE_OF_FREE_NODES]; /* barrier needed? */ \
    chunked_log_alloc_ptr++; \
  } \
  res_; \
})
#endif /* CHUNKED_LOG_AUX_H_GUARD_ */
