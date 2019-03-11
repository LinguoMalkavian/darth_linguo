#Run the experiments
    set -x
    source activate allennlp
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_uni-1-layer32-32.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_uni-1-layer32-32 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_uni-2-layer32-32.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_uni-2-layer32-32 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_uni-3-layer32-32.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_uni-3-layer32-32 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_bi-1-layer32-32.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_bi-1-layer32-32 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_bi-2-layer32-32.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_bi-2-layer32-32 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_bi-3-layer32-32.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_bi-3-layer32-32 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_uni-1-layer32-64.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_uni-1-layer32-64 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_uni-2-layer32-64.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_uni-2-layer32-64 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_uni-3-layer32-64.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_uni-3-layer32-64 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_bi-1-layer32-64.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_bi-1-layer32-64 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_bi-2-layer32-64.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_bi-2-layer32-64 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_bi-3-layer32-64.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_bi-3-layer32-64 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_uni-1-layer64-64.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_uni-1-layer64-64 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_uni-2-layer64-64.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_uni-2-layer64-64 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_uni-3-layer64-64.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_uni-3-layer64-64 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_bi-1-layer64-64.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_bi-1-layer64-64 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_bi-2-layer64-64.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_bi-2-layer64-64 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_bi-3-layer64-64.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_bi-3-layer64-64 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_uni-1-layer64-128.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_uni-1-layer64-128 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_uni-2-layer64-128.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_uni-2-layer64-128 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_uni-3-layer64-128.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_uni-3-layer64-128 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_bi-1-layer64-128.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_bi-1-layer64-128 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_bi-2-layer64-128.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_bi-2-layer64-128 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_bi-3-layer64-128.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_bi-3-layer64-128 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_uni-1-layer128-128.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_uni-1-layer128-128 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_uni-2-layer128-128.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_uni-2-layer128-128 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_uni-3-layer128-128.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_uni-3-layer128-128 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_bi-1-layer128-128.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_bi-1-layer128-128 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_bi-2-layer128-128.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_bi-2-layer128-128 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_bi-3-layer128-128.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_bi-3-layer128-128 --include-package allen_linguo
    