#!/bin/bash -x

# Running example
# -on mbp bash prepare_experiment1.sh exp1 /Users/pablo/Dropbox/workspace/darth_linguo
# -on lab bash prepare_experiment1.sh exp1 /home/lab/Pablo/darth_linguo 

set -x
source activate allennlp
corpusname=$1
rootpath=$2
corpusdir=$rootpath/Data/$corpusname
#rootpath=/Users/pablo/Dropbox/workspace/darth_linguo
# Split the corpus
cd $rootpath/python
python corpus_tools.py split_corpus $corpusname-base \
    --named_corpus $corpusname --piece_names LM GT GV CT CV \
    --piece_ratio 0.2 0.32 0.08 0.32 0.08
# Generate the word salads
python generateWS.py $corpusname-LM --named_corpus $corpusname --mix --suf train
#Generate the corputed sentences
python corrupt_sentences.py $corpusdir/$corpusname-CT
python corrupt_sentences.py $corpusdir/$corpusname-CV

#label the corresponding training files
python corpus_tools.py label_corpus $corpusname-CT_corrupted-by_verbRM 0 \
    --named_corpus $corpusname --ungramType RV --outfile $corpusname-CTRV-label

python corpus_tools.py label_corpus $corpusname-CT_corrupted-by_verbInfl 0 \
    --named_corpus $corpusname --ungramType VA --outfile $corpusname-CTVA-label

python corpus_tools.py label_corpus $corpusname-CT_corrupted-by_adjInfl 0 \
    --named_corpus $corpusname --ungramType AA --outfile $corpusname-CTAA-label

python corpus_tools.py label_corpus $corpusname-GT 1 \
    --outfile $corpusname-GT-label --named_corpus $corpusname

# label the testing (validation) files
python corpus_tools.py label_corpus $corpusname-CV_corrupted-by_verbRM 0 \
    --named_corpus $corpusname --ungramType RV --outfile $corpusname-CVRV-label

python corpus_tools.py label_corpus $corpusname-CV_corrupted-by_verbInfl 0 \
    --named_corpus $corpusname --ungramType VA --outfile $corpusname-CVVA-label

python corpus_tools.py label_corpus $corpusname-CV_corrupted-by_adjInfl 0 \
    --named_corpus $corpusname --ungramType AA --outfile $corpusname-CVAA-label

python corpus_tools.py label_corpus $corpusname-GV 1 \
    --outfile $corpusname-GV-label --named_corpus $corpusname



# build the training files
cd $rootpath/Data/$corpusname

cat  $corpusname-CTRV-label $corpusname-CTVA-label $corpusname-CTAA-label \
    $corpusname-GT-label $corpusname-LM-mix-gramWS-train > $corpusname-fullws-train

cat  $corpusname-CTRV-label $corpusname-CTVA-label $corpusname-CTAA-label \
    $corpusname-GT-label > $corpusname-nows-train


numws=$(wc -l < $corpusname-LM-mix-gramWS-train)
let half=numws/2
let quarter=numws/4

head -n $half $corpusname-LM-mix-gramWS-train | cat - $corpusname-CTRV-label \
    $corpusname-CTVA-label $corpusname-CTAA-label \
    $corpusname-GT-label > $corpusname-halfws-train

head -n $quarter $corpusname-LM-mix-gramWS-train | cat - $corpusname-CTRV-label \
    $corpusname-CTVA-label $corpusname-CTAA-label \
    $corpusname-GT-label > $corpusname-quarterws-train

# build the test file

cat  $corpusname-CVRV-label $corpusname-CVVA-label $corpusname-CVAA-label \
    $corpusname-GV-label  > $corpusname-validation

#Run the experiments

allennlp train /home/lab/Pablo/darth_linguo/experiments/exp_1_all-WS_350-1000.json \
    -s /home/lab/Pablo/darth_linguo/results/exp_1_all-WS_350-1000_lab \
    --include-package allen_linguo

allennlp train /home/lab/Pablo/darth_linguo/experiments/exp_1_half-WS_350-1000.json \
    -s /home/lab/Pablo/darth_linguo/results/exp_1_half-WS_350-1000_lab \
    --include-package allen_linguo

allennlp train /home/lab/Pablo/darth_linguo/experiments/exp_1_quarter-WS_350-1000.json \
    -s /home/lab/Pablo/darth_linguo/results/exp_1_quarter-WS_350-1000_lab \
    --include-package allen_linguo

allennlp train /home/lab/Pablo/darth_linguo/experiments/exp_1_no-WS_350-1000.json \
    -s /home/lab/Pablo/darth_linguo/results/exp_1_no-WS_350-1000_lab \
    --include-package allen_linguo
