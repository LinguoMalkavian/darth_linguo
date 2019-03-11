#! /bin/bash
#Runs predicitions on all the experiments 

experiments=(exp-2.2_uni-1-layer_32-32 exp-2.2_uni-2-layer_32-32 exp-2.2_uni-3-layer_32-32 exp-2.2_bi-1-layer_32-32 exp-2.2_bi-2-layer_32-32 exp-2.2_bi-3-layer_32-32 exp-2.2_uni-1-layer_32-64 exp-2.2_uni-2-layer_32-64 exp-2.2_uni-3-layer_32-64 exp-2.2_bi-1-layer_32-64 exp-2.2_bi-2-layer_32-64 exp-2.2_bi-3-layer_32-64 exp-2.2_uni-1-layer_64-64 exp-2.2_uni-2-layer_64-64 exp-2.2_uni-3-layer_64-64 exp-2.2_bi-1-layer_64-64 exp-2.2_bi-2-layer_64-64 exp-2.2_bi-3-layer_64-64 exp-2.2_uni-1-layer_64-128 exp-2.2_uni-2-layer_64-128 exp-2.2_uni-3-layer_64-128 exp-2.2_bi-1-layer_64-128 exp-2.2_bi-2-layer_64-128 exp-2.2_bi-3-layer_64-128 exp-2.2_uni-2-layer_128-128 exp-2.2_uni-3-layer_128-128 exp-2.2_uni-1-layer_128-128 exp-2.2_bi-1-layer_128-128 exp-2.2_bi-2-layer_128-128 exp-2.2_bi-3-layer_128-128)

for exp_name in "${experiments[@]}"; 
do
    bash /home/lab/Pablo/darth_linguo/pbs_scripts/generate_predictions.sh $exp_name
done