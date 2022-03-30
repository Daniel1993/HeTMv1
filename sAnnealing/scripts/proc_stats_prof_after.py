#!/usr/bin/env python

import sys
import math
import numpy as np

arg_datafile_loc = 1
outfile = ".after"
sample_header_set = 0
samples = {}
samples_order = {}
i = arg_datafile_loc
min_field_len = 9999

nbFiles = len(sys.argv) - 1

while i < len(sys.argv):
  # print("proc:", sys.argv[i])
  data_name = sys.argv[i]
  samples[data_name] = {}
  samples_order[data_name] = []
  with open(data_name) as f:
    lines = f.readlines()

    for l in lines:
      if l.startswith("#"):
        # ignore
        # print("Finished processing sample")
        continue
      else:
        fields = l.strip().split("\t")
        s = fields[1:]
        samples_order[data_name] += [int(fields[1])]
        samples[data_name][int(fields[1])] = s
  i += 1

map_scc_size = {}

final = [{} for i in range(nbFiles)]
f_i = [0 for i in range(nbFiles)]

idx = 1
keys = samples.keys()
keys_i = [samples_order[k] for k in keys]
while any([len(samples[keys[k]]) > f_i[k] for k in range(nbFiles)]):
  ### TODO: do not process f_i[k] == len(keys_i[k])
  f = []
  for k in range(nbFiles):
    if f_i[k] == len(keys_i[k]):
      f += [9999999] # very large number
    else: 
      f += [int(samples[keys[k]][keys_i[k][f_i[k]]][0])]
  m = [f[k] == min(f) for k in range(nbFiles)]
  for k in range(nbFiles):
    if m[k]:
      samples[keys[k]][keys_i[k][f_i[k]]].insert(0, idx)
      if f_i[k] < len(keys_i[k]):
        f_i[k] += 1
  if all([len(keys_i[k]) - f_i[k] == 1 for k in range(nbFiles)]):
    break
  idx += 1

i = arg_datafile_loc
while i < len(sys.argv):
  data_name = sys.argv[i]
  with open(data_name + ".after", "w+") as f:
    f.write("# idx size 0-time 10-time 50-time 90-time 100-time avg stdev\n")
    f.write("\n".join(["\t".join(
      map(str, [s for s in samples[data_name][k]])
    ) for k in sorted(samples[data_name].keys())]))
  i += 1
