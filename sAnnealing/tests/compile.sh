#!/bin/bash

set -x

gcc torus.c -lbsd -o torus

gcc k-regular.c -lbsd -o k-regular

gcc bipartite-k-regular.c -lbsd -o bipartite-k-regular
#gcc bipartite-k-regular.c -o bipartite-k-regular

set +x

exit 0;x
