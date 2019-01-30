#!/bin/bash

source activate allennlp




allennlp train /home/lab/Pablo/darth_linguo/experiments/full_G_vs_2gWS350-1000dim.json -s /home/lab/Pablo/darth_linguo/results/fullGVS2g350-1000 --include-package allen_linguo
allennlp train /home/lab/Pablo/darth_linguo/experiments/full_G_vs_3gWS350-1000dim.json -s /home/lab/Pablo/darth_linguo/results/fullGVS3g350-1000 --include-package allen_linguo
allennlp train /home/lab/Pablo/darth_linguo/experiments/full_G_vs_4gWS350-1000dim.json -s /home/lab/Pablo/darth_linguo/results/fullGVS4g350-1000 --include-package allen_linguo
allennlp train /home/lab/Pablo/darth_linguo/experiments/full_G_vs_5gWS350-1000dim.json -s /home/lab/Pablo/darth_linguo/results/fullGVS5g350-1000 --include-package allen_linguo
allennlp train /home/lab/Pablo/darth_linguo/experiments/full_G_vs_6gWS350-1000dim.json -s /home/lab/Pablo/darth_linguo/results/fullGVS6g350-1000 --include-package allen_linguo
allennlp train /home/lab/Pablo/darth_linguo/experiments/full_G_vs_mixgWS350-1000dim.json -s /home/lab/Pablo/darth_linguo/results/fullGVSmixg350-1000 --include-package allen_linguo
