#Run the experiments
    set -x
    source activate allennlp
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_1uni-1-layer.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_1uni-1-layer --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_2uni-2-layer.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_2uni-2-layer --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_3uni-3-layer.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_3uni-3-layer --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_4bi-1-layer.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_4bi-1-layer --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_5bi-2-layer.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_5bi-2-layer --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-2.1_6bi-3-layer.json -s /home/lab/Pablo/darth_linguo/results/exp-2.1_6bi-3-layer --include-package allen_linguo
    