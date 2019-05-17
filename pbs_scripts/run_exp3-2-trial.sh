#Run the experiments
set -x
source activate allen

echo "Started experiments" > /home/lab/home/lab/Pablo/darth_linguo/results/exp3-2progress
experiments=(exp-3.2_uni-1-layer64-9-trial exp-3.2_uni-1-layer64-17-trial)
length="${#experiments[@]}"
let "count=0"

for exp_name in "${experiments[@]}"; 
do
    allennlp train /home/lab/home/lab/Pablo/darth_linguo/experiments/$exp_name.json -s /home/lab/home/lab/Pablo/darth_linguo/results/$exp_name --include-package allen_linguo
    bash /home/lab/home/lab/Pablo/darth_linguo/pbs_scripts/generate_predictions.sh $exp_name

    echo "Finished experiment $exp_name" >> /home/lab/home/lab/Pablo/darth_linguo/results/exp3-2progress
    let 'count+=1'  
    now=$(date +"%c")
    echo "$count/$length done at $now" >> /home/lab/home/lab/Pablo/darth_linguo/results/exp3-2progress
    
    aws s3 sync /home/lab/home/lab/Pablo/darth_linguo/results s3://linguo-death-star/darth_linguo/results

done