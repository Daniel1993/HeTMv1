#ifndef TIMER_H_GUARD_
#define TIMER_H_GUARD_

#include <sys/time.h>

#define TIMER_T                         struct timeval

#define TIMER_READ(time)                gettimeofday(&(time), NULL)

#define TIMER_DIFF_SECONDS(start, stop) \
    (((double)(stop.tv_sec)  + (double)(stop.tv_usec / 1000000.0)) - \
     ((double)(start.tv_sec) + (double)(start.tv_usec / 1000000.0)))

#define HETM_TIMER_KEY_SIZE  128

// TODO:
typedef struct HeTM_timeline_  HeTM_timeline_s;

typedef struct HeTM_timeline_point_ {
  TIMER_T ts;
  char key[HETM_TIMER_KEY_SIZE]; // small description
} HeTM_timeline_point_s;

HeTM_timeline_s* HeTM_timeline_init(char *name);
void HeTM_timeline_destroy(HeTM_timeline_s*);
void HeTM_timeline_add_point(HeTM_timeline_s*, HeTM_timeline_point_s);
void HeTM_timeline_to_file(HeTM_timeline_s*, char *fileName);

#endif /* TIMER_H_GUARD_ */
