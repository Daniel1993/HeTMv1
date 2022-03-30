#ifndef _QUEUE_H
#define _QUEUE_H

typedef struct queue *queue;

/* Create a queue that supports n elements */
queue
allocQ(int n
       );

queue
dupQ(queue q
    );

queue
cpyQ(queue dst,
     queue src
    );

void
freeQ(queue q
      );

/* True iff queue is empty */
int
emptyQ(queue q
       );


/* Current element in front */
int
frontQ(queue q
       );

/* Put this element at the end of the queue */
void
pushQ(queue q,
      int e /* element to push to queue */
      );

/* Remove the element in front of the queue */
void
popQ(queue q
     );

#endif /* _QUEUE_H */
