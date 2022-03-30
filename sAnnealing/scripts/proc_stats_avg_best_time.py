#!/usr/bin/env python

import sys
import math
import numpy as np

### argv is an array of files to average

outfile = "multigraph.avg.time"
outfile_std = "multigraph.std.time"
sample_header_set = 0
samples = {}
graph_order = []
i = 1

while i < len(sys.argv):
  graph_name = sys.argv[i].split(".")[0].strip()
  # TODO: use the hint in the name of the file?
  # graph_family = graph_name[:graph_name.rfind("_")]

  with open(sys.argv[i]) as f:
    lines = f.readlines()
    in_header = 1

    for l in lines:
      # line is info of the graph
      if in_header:
        # first line is the headers
        in_header = 0
        continue
      else:
        fields = l.strip().split("\t")
        graph_family = fields[0][:fields[0].rfind("_")]
        if graph_family not in samples:
          samples[graph_family] = []
          graph_order += [graph_family]
        s = [float(f) for f in fields[1:]]
        samples[graph_family] += [s]
  i += 1

# diffScore = np.percentile(np.transpose(s)[10], [0,10,50,90,100])
proc_data = {}
proc_data_std = {}
for k in samples:
  proc_data[k] = []
  proc_data_std[k] = []
  # min: 1 6 11
  # 10%: 2 7 12
  # med: 3 8 13
  # 90%: 4 9 14
  # max: 5 10 15
  for i in range(1, len(np.transpose(samples[k]))):
    proc_data_std[k] += [np.std(np.transpose(samples[k])[i])]
    if i in [1, 6, 11]:
      proc_data[k] += [np.percentile(np.transpose(samples[k])[i], 0)]
    elif i in [2, 7, 12]:
      proc_data[k] += [np.percentile(np.transpose(samples[k])[i], 10)]
    elif i in [3, 8, 13]:
      proc_data[k] += [np.percentile(np.transpose(samples[k])[i], 50)]
    elif i in [4, 9, 14]:
      proc_data[k] += [np.percentile(np.transpose(samples[k])[i], 90)]
    elif i in [5, 10, 15]:
      proc_data[k] += [np.percentile(np.transpose(samples[k])[i], 100)]

with open(outfile, "w+") as f:
  f.write("# graph line 0-scr 10-scr 50-scr 90-scr 100-scr \
    0-iter 10-iter 50-iter 90-iter 100-iter \
    0-time 10-time 50-time 90-time 100-time \n")
  i = 0
  for k in graph_order:
    # f.write("\t".join(map(str, [s for s in range(22)]))+"\n")
    f.write(k+"\t"+str(i)+"\t"+"\t".join(map(str, [s for s in proc_data[k]]))+"\n")
    i += 1

with open(outfile_std, "w+") as f:
  f.write("# graph line 0-scr 10-scr 50-scr 90-scr 100-scr \
    0-iter 10-iter 50-iter 90-iter 100-iter \
    0-time 10-time 50-time 90-time 100-time \n")
  i = 0
  for k in graph_order:
    # f.write("\t".join(map(str, [s for s in range(22)]))+"\n")
    f.write(k+"\t"+str(i)+"\t"+"\t".join(map(str, [s for s in proc_data_std[k]]))+"\n")
    i += 1
