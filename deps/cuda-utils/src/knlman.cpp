#include "knlman.hpp"

#include <map>
#include <tuple>
#include <list>
#include <vector>

#include "cuda_util.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "helper_timer.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda.h>

using namespace std;
using namespace memman;

void
knlman::KnlObj::Run(
  int devId,
  void *strm
) {
  Config::GetInstance()->SelDev(devId);
  int currDev = GetActualDev(devId);

  callback({
    .blocks   = blocks,
    .threads  = threads,
    .devId    = devId,
    .stream   = (cudaStream_t)strm,
    .entryObj = entryObj->GetMemObj(devId),
    .currDev  = currDev
  });
}
