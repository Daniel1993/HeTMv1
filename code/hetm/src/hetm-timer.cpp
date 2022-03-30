#include "hetm-timer.h"

#include <list>

typedef struct HeTM_timeline_ {
  list<HeTM_timeline_point_s> points;
} HeTM_timeline_s;

HeTM_timeline_s* HeTM_timeline_init()
{
  return new HeTM_timeline_s();
}

void HeTM_timeline_destroy(HeTM_timeline_s *timeline)
{
  delete timeline;
}

void HeTM_timeline_add_point(HeTM_timeline_s *timeline, HeTM_timeline_point_s point)
{
  timeline->points.push_back(point);
}

void HeTM_timeline_to_file(HeTM_timeline_s *timeline, char *fileName)
{
 // TODO: would be nice to print the profiling information
}
