#! /bin/bash

wget https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nips.txt.gz
gunzip docword.nips.txt.gz
wget https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab.nips.txt

python gen_nips.py docword.nips.txt nips-train
cp vocab.nips.txt nips-vocab
