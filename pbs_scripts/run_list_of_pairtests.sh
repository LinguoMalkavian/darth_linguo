#! /bin/bash
#Takes a list of experiment names and runs the pairtest on them

experiments=$1

for exp_name in "${experiments[@]}"; 
do
    bash /home/lab/Pablo/darth_linguo/pbs_scripts/run_pairtests.sh $exp_name
done