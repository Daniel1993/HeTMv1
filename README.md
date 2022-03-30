# HeTM

HeTM: Transactional Memory for Heterogeneous Systems.
To be presented at PACT'19.

---

## Sumary

HeTM is an attempt of bringing TM into the heterogenous system world, i.e.,
allowing a program that executes in different processing unit (GPU and CPU) to
maintain a consistent shared dataset.

## Current status

Currently, HeTM is implemented in a synthetic benchmark, we are trying to make
it more generic and export into other applications (besides MemcachedGPU).

## Dependencies

### Hardware

The experiments presented in PACT'19 paper were performed in the following system:
- CPU: **dual socket Intel Xeon E5-2648L v4 @ 1.80GHz**
- GPU: **MSI GTX 1080 8GB XDDR5**
- RAM: **32GB DDR4 ECC@2.4GHz**

Further software information:
```
$ lsb_release -a
Distributor ID:	Ubuntu
Description:	Ubuntu 16.04.6 LTS
Release:	16.04
Codename:	xenial
```
```
$ uname -a
Linux intel14_v1 4.4.0-142-generic #168-Ubuntu SMP Wed Jan 16 21:00:45 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux
```

### Tools

- nvcc: **9.1.85**
- gcc: **5.4.0 20160609**

### Folders

- [cuda-utils](cuda-utils): utility code
- [pr-stm](pr-stm): the implementation of PR-STM instrumented with HeTM
- [tinyMOD](tinyMOD): the implementation of TinySTM instrumented with HeTM
- [tsxMOD](tsxMOD): the implementation of HTM+SGL instrumented with HeTM
- [hetm](hetm): the HeTM library
- [bank](bank): synthetic benchmark
- [memcd](memcd): MemcachedGPU benchmark

## Experiences in the paper (Work In Progress)

In order to reproduce the results in the paper we are currently collecting the
scripts used (the plot for Figure 2 is almost finished, execute ./ex1_inst_cost.sh
to produce the plot).
