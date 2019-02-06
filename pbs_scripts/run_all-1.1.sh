#Run the experiments
set -x
source activate allennlp
allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-1.1_no-WS_32-32.json -s /home/lab/Pablo/darth_linguo/results/exp-1.1_no-WS_32-32 --include-package allen_linguo
allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-1.1_no-WS_64-64.json -s /home/lab/Pablo/darth_linguo/results/exp-1.1_no-WS_64-64 --include-package allen_linguo
allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-1.1_no-WS_128-128.json -s /home/lab/Pablo/darth_linguo/results/exp-1.1_no-WS_128-128 --include-package allen_linguo
allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-1.1_no-WS_256-256.json -s /home/lab/Pablo/darth_linguo/results/exp-1.1_no-WS_256-256 --include-package allen_linguo
allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-1.1_no-WS_512-512.json -s /home/lab/Pablo/darth_linguo/results/exp-1.1_no-WS_512-512 --include-package allen_linguo
allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-1.1_no-WS_512-1024.json -s /home/lab/Pablo/darth_linguo/results/exp-1.1_no-WS_512-1024 --include-package allen_linguo
