#Run the experiments
#set -x
source activate allen

cd ~/Dropbox/workspace/darth_linguo
echo "Started experiments" > /home/lab/home/lab/Pablo/darth_linguo/results/exp3-2progress
experiments=(exp-3.2_uni-1-layer64-16 exp-3.2_uni-2-layer64-12, exp-3.2_uni-3-layer64-11 exp-3.2_uni-4-layer64-9 exp-3.2_bi-1-layer64-9 exp-3.2_bi-2-layer64-7 exp-3.2_bi-3-layer64-6 exp-3.2_bi-4-layer64-5)
iterations=(1 2 3 4 5 6)
let num_runs=${#experiments[@]}*${#iterations[@]}
echo $exp_num
let "count=0"

for exp_name in "${experiments[@]}"; 
do
    for iter in "${iterations[@]}";
    do
        echo $exp_name.json
        echo "/home/lab/home/lab/Pablo/darth_linguo/results/$exp_name-run$iter"
        echo "/home/lab/home/lab/Pablo/darth_linguo/pbs_scripts/generate_predictions.sh $exp_name-run$iter"

        echo "Finished experiment $exp_name"        
        let 'count+=1'  
        now=$(date +"%c")
        echo "$count/$num_runs done at $now"         
    done
done