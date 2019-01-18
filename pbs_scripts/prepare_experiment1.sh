#!/bin/bash -x

set -x
source activate allen
corpusname=exp1_mini
rootpath=/Users/pablo/Dropbox/workspace/darth_linguo
# Split the corpus
cd $rootpath/python
python corpus_tools.py split_corpus $corpusname-base \
    --named_corpus $corpusname --piece_names LM GT GV CT CV \
    --piece_ratio 0.2 0.32 0.08 0.32 0.08
# Generate the word salads
python generateWS.py $corpusname-LM --named_corpus $corpusname --mix
#Generate the corputed sentences
