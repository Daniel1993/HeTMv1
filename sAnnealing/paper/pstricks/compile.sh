#!/bin/bash

latex stack.tex
dvips stack.dvi -o stack.ps

exit 0

# Init process with entr
#
# echo stack.tex | entr bash ./compile.sh
#
# Also start gv in another shell
#
# gv -watch stack.ps
#
