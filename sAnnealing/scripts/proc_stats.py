#!/usr/bin/env python

import sys
import math
import numpy as np

arg_datafile_loc  = 1
arg_nbbuckets_loc = 2
arg_kbp_local_max_loc = 3 # only for kbp
arg_kbp_global_max_loc = 4
arg_min_bucket_loc = 5
arg_max_bucket_loc = 6

datafile = sys.argv[arg_datafile_loc]
nbbuckets = int(sys.argv[arg_nbbuckets_loc])
is_kbp = False
kbp_local_max = 0
kbp_global_max = 0
use_min_bucket = False
min_bucket = 0
if len(sys.argv) > 3:
  is_kbp = True
  kbp_local_max = int(sys.argv[arg_kbp_local_max_loc])
  kbp_global_max = int(sys.argv[arg_kbp_global_max_loc])
if len(sys.argv) > 5:
  use_min_bucket = True
  min_bucket = float(sys.argv[arg_min_bucket_loc])
  max_bucket = float(sys.argv[arg_max_bucket_loc])

outfile_time = datafile + ".btime"
outfile_iter = datafile + ".biter"
outfile_vert = datafile + ".max_cost"

sample_header_set = 0
samples = []
row_names = {}
vert  = 0
edge  = 0
d_v   = 0
log_v = 0
is_bb = False
prof_SCC_exec_time = {}

### TODO: open .guess to get the target solution

graph_name = datafile.split(".")[0].strip()
solution = 0
# if field[3] at solution, then stop
with open(graph_name + ".guess") as f:
  lines = f.readlines()
  for l in lines:
    solution = int(l)
    break

with open(datafile) as f:
  lines = f.readlines()
  new_sample = 1
  new_header = 1
  sample = []
  lastSCCsize = -1
  lastSCCid = -1
  lastSCCts = -1
  bug_found = False
  processing_prof = True
  sccInfo = {}
  sccIdSize = {}

  for l in lines:
    # line is info of the graph
    if new_sample:
      vert = l.split("#")[1].strip().split(",")[0].split("=")[1]
      edge = l.split("#")[1].strip().split(",")[1].split("=")[1]
      d_v = l.split("#")[1].strip().split(",")[2].split("=")[1]
      log_v = l.split("#")[1].strip().split(",")[3].split("=")[1]
      with open(outfile_vert, "w+") as f_vert:
        f_vert.write(str(float(d_v) * float(log_v)))
      new_sample = 0
    elif not sample_header_set and new_header:
      # print("Second line")
      if l.startswith("#BB"):
        is_bb = True
        continue
      sample_header_set = 1
      new_header = 0
      fields = l.split("#")[1].strip().split("\t")
      i = 0
      for f in fields:
        row_names[f] = i
        i += 1
      if i == 5:
        is_bb = True
    elif new_header < 3:
      # pass three lines
      new_header += 1
    elif l.startswith("#"):
      # next sample
      # print("Finished processing sample")
      samples += [sample]
      lastSCCid = -1
      sample = []
      new_header = 1
      if is_bb:
        # just 1 sample in BB
        break
    else:
      fields = l.strip().split("\t")
      s = [float(f) for f in fields]
      if len(fields) < 8:
        is_bb = True
      if len(sample) > 0:
        lastSample = len(sample) - 1
        s += [float(fields[0])-sample[lastSample][0] if sample[lastSample][0] > 0 else 0] # diff iter
        s += [float(fields[1])-sample[lastSample][1] if sample[lastSample][1] > 0 else 0] # diff time
        s += [float(fields[4])-sample[lastSample][4] if sample[lastSample][4] > 0 else 0] # diff accScore
        if not is_bb and len(fields) > 7:
          s += [float(fields[7])-sample[lastSample][7] if sample[lastSample][7] > 0 else 0] # diff vertPerIt
          s += [float(fields[9])-sample[lastSample][9] if sample[lastSample][9] > 0 else 0] # diff nbDfs
      else:
        if not is_bb and len(fields) > 7:
          s += [0, 0, 0, 0, 0]
        else:
          s += [0, 0, 0]
      if len(fields) >= 7 and int(fields[3]) < solution: 
        if int(fields[5]) != -1 and int(fields[5]) not in sccIdSize:
          sccIdSize[int(fields[5])] = int(fields[6])
        if lastSCCid == -1 and lastSCCid != int(fields[5]):
          lastSCCsize = int(fields[6])
          lastSCCid = int(fields[5])
          lastSCCts = float(fields[1])
        elif lastSCCid != int(fields[5]):
          execTime = float(sample[-1][1]) - lastSCCts
          if bug_found:
            if lastSCCid not in sccInfo:
              sccInfo[lastSCCid] = []
            sccInfo[lastSCCid] += [execTime]
            lastSCCsize = int(fields[6])
            lastSCCid = int(fields[5])
            lastSCCts = float(fields[1])
          else:
            if lastSCCsize > 0 and execTime > 0 and lastSCCid not in sccInfo:
              sccInfo[lastSCCid] = []
            if lastSCCsize > 0 and execTime > 0:
              sccInfo[lastSCCid] += [execTime]
              lastSCCsize = int(fields[6])
              lastSCCid = int(fields[5])
              lastSCCts = float(fields[1])
        bug_found = False
      elif int(fields[3]) == solution and processing_prof:
        processing_prof = False
        if lastSCCid != int(fields[5]):
          execTime = float(sample[-1][1]) - lastSCCts
        else:
          execTime = float(fields[1]) - lastSCCts
        if lastSCCid not in sccInfo:
          sccInfo[lastSCCid] = []
        sccInfo[lastSCCid] += [execTime]
        lastSCCsize = int(fields[6])
        lastSCCid = int(fields[5])
        lastSCCts = float(fields[1])
      else: # work around
        bug_found = True
      sample += [s]
        
  # aggregate all the latencies
  aux_size_scc = {}
  for scc_id in sccInfo:
    if sccIdSize[scc_id] not in aux_size_scc:
      aux_size_scc[sccIdSize[scc_id]] = []
    aux_size_scc[sccIdSize[scc_id]] += [sum(sccInfo[scc_id])]
  for scc_size in aux_size_scc: # average this seems more correct
    prof_SCC_exec_time[scc_size] = np.average(aux_size_scc[scc_size])

buckets_per_iter = []
buckets_per_time = []
merge_samples_iter = []
merge_samples_time = []

max_time = -1
min_time = 999999

for s in samples:
  len_s = len(s)
  samples_per_bucket = math.ceil(len_s / nbbuckets)
  bucket = []
  for i in range(nbbuckets-1):
    bucket += [s[int(i*samples_per_bucket):int((i+1)*samples_per_bucket)]]
  bucket += [s[int((nbbuckets-1)*samples_per_bucket):len(s)]]
  if float(bucket[-1][-1][0]) > max_time:
    max_time = float(bucket[-1][-1][0])
  if float(bucket[-1][0][0]) < min_time:
    min_time = float(bucket[-1][0][0])
  buckets_per_iter += [bucket]

for i in range(nbbuckets):
  m = []
  for s in range(len(buckets_per_iter)):
    m += buckets_per_iter[s][i]
  merge_samples_iter += [m]

for s in samples:
  len_s = len(s)
  min_samples = len_s / (2.0 ** (nbbuckets*nbbuckets))
  if use_min_bucket:
    delta = float(max_bucket - min_bucket) / (2.0 ** nbbuckets)
    time_threshold_per_bucket = min_bucket
  else:
    delta = float(max_time - min_time) / (2.0 ** nbbuckets)
    time_threshold_per_bucket = min_time + delta
  i = 0
  j = 0
  s_bucket = []
  while j < len_s:
    bucket = []
    l = s[j]
    # minimum number of samples in bucket is 10
    while float(l[1]) <= time_threshold_per_bucket or len(bucket) < min_samples:
      bucket += [l]
      j += 1
      if j >= len_s:
        break
      l = s[j]
    i += 1
    delta *= 2.0
    if use_min_bucket:
      time_threshold_per_bucket += delta
    else:
      time_threshold_per_bucket = float(bucket[-1][1]) + delta
    if i == nbbuckets:
      bucket += s[j+1:len_s]
      j = len_s
    s_bucket += [bucket]
  bucket = [bucket[-1]]
  while i < nbbuckets:
    s_bucket += [bucket]
    i += 1
  buckets_per_time += [s_bucket]

for i in range(nbbuckets):
  m = []
  for s in range(len(buckets_per_time)):
    m += buckets_per_time[s][i]
  merge_samples_time += [m]

data_iter = []
data_time = []

for s in merge_samples_iter:
  l = str(int(s[0][0])) if s[0][0] < 1000.0 else str(int(s[0][0]/1000.0))+"k" if s[0][0]/1000.0 < 1000.0 else "{:.2f}".format(s[0][0]/1000000.0)+"M"
  h = str(int(s[-1][0])) if s[-1][0] < 1000.0 else str(int(s[-1][0]/1000.0))+"k" if s[-1][0]/1000.0 < 1000.0 else "{:.2f}".format(s[-1][0]/1000000.0)+"M"
  label = "[" + l + "," + h + "]"
  score = np.percentile(np.transpose(s)[4], [0,10,50,90,100])
  if not is_bb:
    diffScore = np.percentile(np.transpose(s)[10], [0,10,50,90,100])
    verts = np.percentile(np.transpose(s)[11], [0,10,50,90,100])
    nbDfs = np.percentile(np.transpose(s)[12], [0,10,50,90,100])
  else:
    diffScore = [0, 0, 0, 0, 0]
    verts = [0, 0, 0, 0, 0]
    nbDfs = [0, 0, 0, 0, 0]
  times = np.percentile(np.transpose(s)[1], [0,10,50,90,100])
  kbp_stats = [0, 0, 0, 0]
  if is_kbp:
    len_s4 = float(len(np.transpose(s)[4]))
    kbp_stats[0] += float(np.count_nonzero(np.transpose(s)[4] < kbp_local_max)) / len_s4
    kbp_stats[1] += float(np.count_nonzero(np.transpose(s)[4] == kbp_local_max)) / len_s4
    kbp_stats[3] += float(np.count_nonzero(np.transpose(s)[4] == kbp_global_max)) / len_s4
    kbp_stats[2] += float(np.count_nonzero(np.transpose(s)[4] > kbp_local_max) - kbp_stats[3]) / len_s4
  line = [label]
  line += [i for i in score]
  line += [i for i in diffScore]
  line += [i for i in verts]
  line += [i for i in nbDfs]
  line += [i for i in times]
  line += [i for i in kbp_stats]
  data_iter += [line]

for s in merge_samples_time:
  l = "{:.3f}".format(s[0][1]/1000.0) if s[0][1] < 1000.0 else "{:.1f}".format(s[0][1]/1000.0) if s[0][1]/1000.0 < 60.0 else "{:.1f}".format(s[0][1]/60000.0) + "min"
  h = "{:.3f}".format(s[-1][1]/1000.0) if s[-1][1] < 1000.0 else "{:.1f}".format(s[-1][1]/1000.0) if s[-1][1]/1000.0 < 60.0 else "{:.1f}".format(s[-1][1]/60000.0) + "min"
  label = "[" + l + "," + h + "]"
  score = np.percentile(np.transpose(s)[4], [0,10,50,90,100])
  if not is_bb:
    diffScore = np.percentile(np.transpose(s)[10], [0,10,50,90,100])
    verts = np.percentile(np.transpose(s)[11], [0,10,50,90,100])
    nbDfs = np.percentile(np.transpose(s)[12], [0,10,50,90,100])
  else:
    diffScore = [0, 0, 0, 0, 0]
    verts = [0, 0, 0, 0, 0]
    nbDfs = [0, 0, 0, 0, 0]
  times = np.percentile(np.transpose(s)[1], [0,10,50,90,100])
  kbp_stats = [0, 0, 0, 0]
  if is_kbp:
    len_s4 = float(len(np.transpose(s)[4]))
    kbp_stats[0] += float(np.count_nonzero(np.transpose(s)[4] < kbp_local_max)) / len_s4
    kbp_stats[1] += float(np.count_nonzero(np.transpose(s)[4] == kbp_local_max)) / len_s4
    kbp_stats[3] += float(np.count_nonzero(np.transpose(s)[4] == kbp_global_max)) / len_s4
    kbp_stats[2] += (float(np.count_nonzero(np.transpose(s)[4] > kbp_local_max)) / len_s4) - kbp_stats[3]
    # print("Sum kbp stats:", sum(kbp_stats),"len_s4:",len_s4,"0:",np.count_nonzero(np.transpose(s)[4] < kbp_local_max),
    #   "1:",np.count_nonzero(np.transpose(s)[4] == kbp_local_max),"2:",
    #   np.count_nonzero(np.transpose(s)[4] > kbp_local_max)-np.count_nonzero(np.transpose(s)[4] == kbp_global_max),
    #   "3:",np.count_nonzero(np.transpose(s)[4] == kbp_global_max))
    print("Sum kbp stats:", sum(kbp_stats), "0:",kbp_stats[0], "1:",kbp_stats[1], "2:",kbp_stats[2], "3:",kbp_stats[3])
  line = [label]
  line += [i for i in score]
  line += [i for i in diffScore]
  line += [i for i in verts]
  line += [i for i in nbDfs]
  line += [i for i in times]
  line += [i for i in kbp_stats]
  data_time += [line]

with open(outfile_iter, "w+") as f:
  f.write("# line iter \
    0-scr 10-scr 50-scr 90-scr 100-scr \
    0-dScr 10-dScr 50-dScr 90-dScr 100-dScr \
    0-vert 10-vert 50-vert 90-vert 100-vert \
    0-dfs 10-dfs 50-dfs 90-dfs 100-dfs \
    0-time 10-time 50-time 90-time 100-time \
    lt-lmax at-lmax gt-lmax at-gmax\n")
  # f.write("\t".join(map(str, [s for s in range(22)]))+"\n")
  [f.write(str(s)+"\t"+"\t".join(map(str, data_iter[s]))+"\n") for s in range(len(data_iter))]

with open(datafile+".profSCCtime", "w+") as f:
  f.write("# scc_size scc_time\n")
  # f.write("\t".join(map(str, [s for s in range(22)]))+"\n")
  for s in prof_SCC_exec_time:
    f.write(str(s) + "\t" + str(prof_SCC_exec_time[s]) + "\n")

with open(outfile_time, "w+") as f:
  f.write("# line time \
    0-scr 10-scr 50-scr 90-scr 100-scr \
    0-dScr 10-dScr 50-dScr 90-dScr 100-dScr \
    0-vert 10-vert 50-vert 90-vert 100-vert \
    0-dfs 10-dfs 50-dfs 90-dfs 100-dfs \
    0-time 10-time 50-time 90-time 100-time \
    lt-lmax at-lmax gt-lmax at-gmax\n")
  # f.write("\t".join(map(str, [s for s in range(22)]))+"\n")
  [f.write(str(s)+"\t"+"\t".join(map(str, data_time[s]))+"\n") for s in range(len(data_time))]
