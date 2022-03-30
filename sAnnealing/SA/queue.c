#include <stdlib.h>
#include <string.h>

#include "queue.h"

struct queue{
  int *A; /* Array of elements */
  int front;
  int back;
  int n; /* size of the queue */
};

/* Create a queue that supports n elements */
queue
allocQ(int n
       )
{
  queue q = malloc(sizeof(struct queue));

  q->n = n+1;
  q->A = (int*) malloc(q->n*sizeof(int));
  q->front = 0;
  q->back = 0;

  return q;
}

queue
dupQ(queue q
    )
{
  queue dup = calloc(1, sizeof(struct queue));
  return cpyQ(dup, q);
}

queue
cpyQ(queue dst,
     queue src
    )
{
  if (!dst) return dupQ(src);
  dst->n = src->n;
  dst->back = src->back;
  dst->front = src->front;
  dst->A = (int*)realloc(dst->A, src->n*sizeof(int));
  memcpy(dst->A, src->A, src->n*sizeof(int));
  return dst;
}

void
freeQ(queue q
      )
{
  free(q->A);
  free(q);
}

/* True iff queue is empty */
int
emptyQ(queue q
       )
{
  return q->front == q->back;
}

/* Current element in front */
int
frontQ(queue q
       )
{
  return q->A[q->front];
}

/* Put this element at the end of the queue */
void
pushQ(queue q,
      int e /* element to push to queue */
      )
{
  q->A[q->back] = e;
  q->back++;
  q->back %= q->n;
}

/* Remove the element in front of the queue */
void
popQ(queue q
     )
{
  q->front++;
  q->front %= q->n;
}

