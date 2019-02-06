#Run the experiments
    set -x
    source activate allennlp
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-1.2_random.json -s /home/lab/Pablo/darth_linguo/results/exp-1.2_random --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-1.2_pretrained-cont.json -s /home/lab/Pablo/darth_linguo/results/exp-1.2_pretrained-cont --include-package allen_linguo
    allennlp train /home/lab/Pablo/darth_linguo/experiments/exp-1.2_pretrained-freeze.json -s /home/lab/Pablo/darth_linguo/results/exp-1.2_pretrained-freeze --include-package allen_linguo
    