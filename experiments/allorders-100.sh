#!/bin/bash

source activate allennlp


allennlp train /home/lab/Pablo/darth_linguo/experiments/full_G_vs_3gWS32dim.json -s /home/lab/Pablo/darth_linguo/results/fullGVS3g32 --include-package allen_linguo
allennlp train /home/lab/Pablo/darth_linguo/experiments/full_G_vs_4gWS32dim.json -s /home/lab/Pablo/darth_linguo/results/fullGVS4g32 --include-package allen_linguo
allennlp train /home/lab/Pablo/darth_linguo/experiments/full_G_vs_5gWS32dim.json -s /home/lab/Pablo/darth_linguo/results/fullGVS5g32 --include-package allen_linguo

allennlp train /home/lab/Pablo/darth_linguo/experiments/full_G_vs_2gWS100dim.json -s /home/lab/Pablo/darth_linguo/results/fullGVS2g100 --include-package allen_linguo
allennlp train /home/lab/Pablo/darth_linguo/experiments/full_G_vs_3gWS100dim.json -s /home/lab/Pablo/darth_linguo/results/fullGVS3g100 --include-package allen_linguo
allennlp train /home/lab/Pablo/darth_linguo/experiments/full_G_vs_4gWS100dim.json -s /home/lab/Pablo/darth_linguo/results/fullGVS4g100 --include-package allen_linguo
allennlp train /home/lab/Pablo/darth_linguo/experiments/full_G_vs_5gWS100dim.json -s /home/lab/Pablo/darth_linguo/results/fullGVS5g100 --include-package allen_linguo
allennlp train /home/lab/Pablo/darth_linguo/experiments/full_G_vs_6gWS100dim.json -s /home/lab/Pablo/darth_linguo/results/fullGVS6g100 --include-package allen_linguo
allennlp train /home/lab/Pablo/darth_linguo/experiments/full_G_vs_mixgWS100dim.json -s /home/lab/Pablo/darth_linguo/results/fullGVSmixg100 --include-package allen_linguo
