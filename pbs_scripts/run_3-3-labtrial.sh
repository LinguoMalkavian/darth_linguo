#Run the experiments
#set -x
conda init bash 
conda activate allen

cd /home/lab/home/lab/Pablo/darth_linguo

resfile="/home/lab/home/lab/Pablo/darth_linguo/results/consolidated/random_init_trial"
progfile='/home/lab/home/lab/Pablo/darth_linguo/results/exp3-3-trial-progress'
echo "exp_name,VA,AA,RV,total" > $resfile
echo "Started experiments" > $progfile
experiments=(exp-3.3_uni-1-layer_64-182_run1 exp-3.3_uni-1-layer_64-182_run2)
length="${#experiments[@]}"

echo $exp_num
let "count=0"

for exp_name in "${experiments[@]}"; 
do

    allennlp train /home/lab/home/lab/Pablo/darth_linguo/experiments/$exp_name.json -s /home/lab/home/lab/Pablo/darth_linguo/results/$exp_name --include-package allen_linguo
    bash /home/lab/home/lab/Pablo/darth_linguo/pbs_scripts/generate_predictions_trial-lab.sh $exp_name

    echo "Finished experiment $exp_name" >> $progfile
    let 'count+=1'  
    now=$(date +"%c")
    echo "$count/$length done at $now" >> $progfile
    
    python python/extract_results.py $exp_name >> $resfile
    aws s3 cp $resfile s3://linguo-death-star/darth_linguo/results
    aws s3 cp $progfile s3://linguo-death-star/darth_linguo/results

done