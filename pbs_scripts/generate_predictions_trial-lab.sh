#!/bin/bash -x

set -x 
source activate allen 

allennlp predict /home/lab/home/lab/Pablo/darth_linguo/results/$1/model.tar.gz \
/home/lab/home/lab/Pablo/darth_linguo/Data/pairtest-mini/pairtest-mini_AA \
--include-package allen_linguo \
--cuda-device 0 \
--weights-file /home/lab/home/lab/Pablo/darth_linguo/results/$1/best.th \
--output-file /home/lab/home/lab/Pablo/darth_linguo/results/pairtest/classifier_results/$1_AA_results \
--use-dataset-reader --predictor linguo-predictor  --silent

allennlp predict /home/lab/home/lab/Pablo/darth_linguo/results/$1/model.tar.gz \
/home/lab/home/lab/Pablo/darth_linguo/Data/pairtest-mini/pairtest-mini_VA \
--include-package allen_linguo \
--cuda-device 0 \
--weights-file /home/lab/home/lab/Pablo/darth_linguo/results/$1/best.th \
--output-file /home/lab/home/lab/Pablo/darth_linguo/results/pairtest/classifier_results/$1_VA_results \
--use-dataset-reader --predictor linguo-predictor --silent
 
allennlp predict /home/lab/home/lab/Pablo/darth_linguo/results/$1/model.tar.gz \
/home/lab/home/lab/Pablo/darth_linguo/Data/pairtest-mini/pairtest-mini_RV \
--include-package allen_linguo \
--cuda-device 0 \
--weights-file /home/lab/home/lab/Pablo/darth_linguo/results/$1/best.th \
--output-file /home/lab/home/lab/Pablo/darth_linguo/results/pairtest/classifier_results/$1_RV_results \
--use-dataset-reader --predictor linguo-predictor --silent