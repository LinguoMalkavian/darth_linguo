#! /bin/bash
#Runs predicitions on all the experiments 

experiments=(exp-1.2_pretrained-cont exp-1_quarter-WS_350-1000_lab exp-1.3_64-512 exp-1.3_512-512 exp-1_all-WS_350-1000_lab exp-2.1_3uni-3-layer exp-1.1_256-256 exp-1.1_512-1024 exp-2.1_6bi-3-layer exp-1.3_256-512 exp-2.1_4bi-1-layer exp-2.1_2uni-2-layer exp-1.1_512-512 exp-1.1_128-128 exp-1.1_32-32 exp-2.1_1uni-1-layer exp-1_half-WS_350-1000_lab exp-2.1_5bi-2-layer exp-1.2_pretrained-freeze exp-1.1_64-64 exp-1.3_128-512 exp-1.2_random exp-1_no-WS_350-1000_lab exp-1.3_32-512 exp-1.3_16-512)

for exp_name in "${experiments[@]}"; 
do
    bash /home/lab/Pablo/darth_linguo/pbs_scripts/generate_predictions.sh $exp_name
done