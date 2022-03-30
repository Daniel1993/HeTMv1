#!/bin/bash

# Run as :
#
#  bash ./getbib.sh $(xclip -o)

set -x

wget -O biblio.tmp "http://api.crossref.org/works/$1/transform/application/x-bibtex"

echo >> biblio.bib
cat biblio.tmp >> biblio.bib
echo >> biblio.bib
rm biblio.tmp

set +x

exit 0
