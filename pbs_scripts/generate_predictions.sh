#!/bin/bash -x

set -x 
source activate allennlp

allennlp predict /home/lab/Pablo/darth_linguo/results/$1/model.tar.gz \
/home/lab/Pablo/darth_linguo/Data/exp3-pairtest/exp3-pairtest_AA \
--include-package allen_linguo \
--cuda-device 0 \
--weights-file /home/lab/Pablo/darth_linguo/results/$1/best.th \
--output-file /home/lab/Pablo/darth_linguo/results/pairtest/$1_AA_results \
--use-dataset-reader

allennlp predict /home/lab/Pablo/darth_linguo/results/$1/model.tar.gz \
/home/lab/Pablo/darth_linguo/Data/exp3-pairtest/exp3-pairtest_VA \
--include-package allen_linguo \
--cuda-device 0 \
--weights-file /home/lab/Pablo/darth_linguo/results/$1/best.th \
--output-file /home/lab/Pablo/darth_linguo/results/pairtest/$1_VA_results \
--use-dataset-reader

allennlp predict /home/lab/Pablo/darth_linguo/results/$1/model.tar.gz \
/home/lab/Pablo/darth_linguo/Data/exp3-pairtest/exp3-pairtest_RV \
--include-package allen_linguo \
--cuda-device 0 \
--weights-file /home/lab/Pablo/darth_linguo/results/$1/best.th \
--output-file /home/lab/Pablo/darth_linguo/results/pairtest/$1_RV_results \
--use-dataset-reader

######

allennlp predict /home/lab/Pablo/darth_linguo/results/exp-2.1_1uni-1-layer/model.tar.gz \
/home/lab/Pablo/darth_linguo/Data/exp3-pairtest/exp3-pairtest_AA \
--include-package allen_linguo \
--cuda-device 0 \
--weights-file /home/lab/Pablo/darth_linguo/results/exp-2.1_1uni-1-layer/best.th \
--output-file /home/lab/Pablo/darth_linguo/results/pairtest/exp-2.1_1uni-1-layer_AA_results \
--use-dataset-reader --predictor predictor

allennlp predict /home/lab/Pablo/darth_linguo/results/exp-2.1_1uni-1-layer/model.tar.gz \
/home/lab/Pablo/darth_linguo/Data/pairtest-mini/pairtest-mini_AA \
--include-package allen_linguo \
--cuda-device 0 \
--weights-file /home/lab/Pablo/darth_linguo/results/exp-2.1_1uni-1-layer/best.th \
--output-file /home/lab/Pablo/darth_linguo/results/mini-pairtest/exp-2.1_1uni-1-layer_AA_results \
--use-dataset-reader --predictor linguo-predictor