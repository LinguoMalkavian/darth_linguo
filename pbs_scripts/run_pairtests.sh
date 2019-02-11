#!/bin/bash -x

set -x 
source activate allennlp

allennlp evaluate /home/lab/Pablo/darth_linguo/results/$1/model.tar.gz \
/home/lab/Pablo/darth_linguo/Data/exp3-pairtest/exp3-pairtest_AA \
--include-package allen_linguo \
--cuda-device 0 \
--weights-file /home/lab/Pablo/darth_linguo/results/$1/best.th \
--output-file /home/lab/Pablo/darth_linguo/results/pairtest/$1_AA

allennlp evaluate /home/lab/Pablo/darth_linguo/results/$1/model.tar.gz \
/home/lab/Pablo/darth_linguo/Data/exp3-pairtest/exp3-pairtest_VA \
--include-package allen_linguo \
--cuda-device 0 \
--weights-file /home/lab/Pablo/darth_linguo/results/$1/best.th \
--output-file /home/lab/Pablo/darth_linguo/results/pairtest/$1_VA

allennlp evaluate /home/lab/Pablo/darth_linguo/results/$1/model.tar.gz \
/home/lab/Pablo/darth_linguo/Data/exp3-pairtest/exp3-pairtest_RV \
--include-package allen_linguo \
--cuda-device 0 \
--weights-file /home/lab/Pablo/darth_linguo/results/$1/best.th \
--output-file /home/lab/Pablo/darth_linguo/results/pairtest/$1_RV
