#!/usr/bin/env python

import sys
import math
import numpy as np

arg_datafile_loc = 1
outfile = "multigraph.time"
sample_header_set = 0
samples = {}
graph_order = []
i = arg_datafile_loc
min_field_len = 9999
while i < len(sys.argv):
  graph_name = sys.argv[i].split(".")[0].strip()
  solution = 0
  samples[graph_name] = []
  graph_order += [graph_name]
  with open(graph_name + ".guess") as f:
    lines = f.readlines()
    for l in lines:
      solution = int(l)
      break
  with open(sys.argv[i]) as f:
    lines = f.readlines()
    new_sample = 1
    new_header = 1
    sample_done = 0
    lastline = 0

    for l in lines:
      # line is info of the graph
      if new_sample:
        # first line
        new_sample = 0
      elif not sample_header_set and new_header:
        # print("Second line")
        sample_header_set = 1
        new_header = 0
        fields = l.split("#")[1].strip().split("\t")
      elif new_header < 3:
        # pass three lines
        new_header += 1
      elif l.startswith("#"):
        # next sample
        # print("Finished processing sample")
        if sample_done == 0:
          # found none
          fields = lastline.strip().split("\t")
          s = []
          s += [float(f) for f in fields]
          # print("len(s):", len(s))
          if len(s) < min_field_len:
            min_field_len = len(s)
          samples[graph_name] += [s]
        new_header = 1
        sample_done = 0
      elif sample_done == 1:
        continue
      else:
        fields = l.strip().split("\t")
        if sample_done == 0 and int(fields[3]) == solution:
          # s = [graph_name]
          s = []
          s += [float(f) for f in fields]
          # print("len(s):", len(s))
          if len(s) < min_field_len:
            min_field_len = len(s)
          samples[graph_name] += [s]
          sample_done = 1
      lastline = l
  i += 1

# diffScore = np.percentile(np.transpose(s)[10], [0,10,50,90,100])
proc_data = {}
for k in samples:
  proc_data[k] = []
  new_s = [s[:min_field_len] for s in samples[k]]
  proc_data[k] += [s for s in np.percentile(np.transpose(new_s)[4], [0,10,50,90,100])]
  proc_data[k] += [s for s in np.percentile(np.transpose(new_s)[0], [0,10,50,90,100])]
  proc_data[k] += [s for s in np.percentile(np.transpose(new_s)[1], [0,10,50,90,100])]
  # print("Min time:", proc_data[k][5])

with open(outfile, "w+") as f:
  f.write("# graph line 0-scr 10-scr 50-scr 90-scr 100-scr \
    0-iter 10-iter 50-iter 90-iter 100-iter \
    0-time 10-time 50-time 90-time 100-time \n")
  i = 0
  for k in graph_order:
    # f.write("\t".join(map(str, [s for s in range(22)]))+"\n")
    f.write(k+"\t"+str(i)+"\t"+"\t".join(map(str, [s for s in proc_data[k]]))+"\n")
    i += 1
