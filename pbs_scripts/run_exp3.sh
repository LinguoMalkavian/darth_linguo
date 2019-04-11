#Run the experiments
set -x
source activate allen

experiments=(exp-3_uni-1-layer_32-32 exp-3_uni-2-layer_32-32 exp-3_uni-3-layer_32-32 exp-3_bi-1-layer_32-32 exp-3_bi-2-layer_32-32 exp-3_bi-3-layer_32-32 exp-3_uni-1-layer_32-64 exp-3_uni-2-layer_32-64 exp-3_uni-3-layer_32-64 exp-3_bi-1-layer_32-64 exp-3_bi-2-layer_32-64 exp-3_bi-3-layer_32-64 exp-3_uni-1-layer_64-64 exp-3_uni-2-layer_64-64 exp-3_uni-3-layer_64-64 exp-3_bi-1-layer_64-64 exp-3_bi-2-layer_64-64 exp-3_bi-3-layer_64-64 exp-3_uni-1-layer_64-128 exp-3_uni-2-layer_64-128 exp-3_uni-3-layer_64-128 exp-3_bi-1-layer_64-128 exp-3_bi-2-layer_64-128 exp-3_bi-3-layer_64-128 exp-3_uni-2-layer_128-128 exp-3_uni-3-layer_128-128 exp-3_uni-1-layer_128-128 exp-3_bi-1-layer_128-128 exp-3_bi-2-layer_128-128 exp-3_bi-3-layer_128-128)

for exp_name in "${experiments[@]}"; 
do
    allennlp train /home/lab/home/lab/Pablo/darth_linguo/experiments/$exp_name.json -s /home/lab/home/lab/Pablo/darth_linguo/results/$exp_name --include-package allen_linguo
    bash /home/lab/home/lab/Pablo/darth_linguo/pbs_scripts/generate_predictions.sh $exp_name
done