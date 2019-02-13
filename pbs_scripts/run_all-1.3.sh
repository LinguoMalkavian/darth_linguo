#Run the experiments
    set -x
    source activate allennlp
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-1.3_16-512.json -s /home/lab/Pablo/darth_linguo/results/exp-1.3_16-512 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-1.3_32-512.json -s /home/lab/Pablo/darth_linguo/results/exp-1.3_32-512 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-1.3_64-512.json -s /home/lab/Pablo/darth_linguo/results/exp-1.3_64-512 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-1.3_128-512.json -s /home/lab/Pablo/darth_linguo/results/exp-1.3_128-512 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-1.3_256-512.json -s /home/lab/Pablo/darth_linguo/results/exp-1.3_256-512 --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-1.3_512-512.json -s /home/lab/Pablo/darth_linguo/results/exp-1.3_512-512 --include-package allen_linguo
    