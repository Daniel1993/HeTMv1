#!/usr/bin/env python

import sys
import math
import numpy as np

arg_datafile_loc = 1
outfile = "profSCC.time"
sample_header_set = 0
samples = {}
graph_order = []
i = arg_datafile_loc
min_field_len = 9999

while i < len(sys.argv):
  # print("proc:", sys.argv[i])
  with open(sys.argv[i]) as f:
    lines = f.readlines()

    for l in lines:
      if l.startswith("#"):
        # ignore
        # print("Finished processing sample")
        continue
      else:
        fields = l.strip().split("\t")
        s = []
        s += [float(f) for f in fields]
        if fields[0] not in samples:
          samples[fields[0]] = []
        samples[fields[0]] += [s[1]]
  i += 1

# diffScore = np.percentile(np.transpose(s)[10], [0,10,50,90,100])
proc_data = {}
i = 1
for k in sorted(samples.keys(), key=lambda k: float(k)):
  proc_data[k] = [i, k]
  proc_data[k] += [p for p in np.percentile(np.transpose(samples[k]), [0,10,50,90,100])]
  proc_data[k] += [np.average(np.array(samples[k]))]
  proc_data[k] += [np.std(np.array(samples[k]))]
  # print("Min time:", proc_data[k][5])
  i += 1

with open(outfile, "w+") as f:
  f.write("# idx size 0-time 10-time 50-time 90-time 100-time avg stdev\n")
  [f.write("\t".join(map(str, [s for s in proc_data[k]]))+"\n") for k in sorted(proc_data.keys(), key=lambda k: float(k))]
