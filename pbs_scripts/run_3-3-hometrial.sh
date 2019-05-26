#Run the experiments
set -x
source activate allen

cd ~/Dropbox/workspace/darth_linguo

resfile="/Users/pablo/Dropbox/workspace/darth_linguo/results/consolidated/trial"
progress_file="/Users/pablo/Dropbox/workspace/darth_linguo/results/exp3-3progress"
rootdir="/Users/pablo/Dropbox/workspace/darth_linguo/"
echo "exp_name,VA,AA,RV,total" > $resfile
echo "Started experiments" > $progress_file
experiments=(trial-home_bi-1-layer_64-9 trial-home_uni-1-layer_64-16)
length="${#experiments[@]}"
iterations=(1 2)
let num_runs=${#experiments[@]}*${#iterations[@]}
echo $exp_num
let "count=0"

for exp_name in "${experiments[@]}"; 
do
    for iter in "${iterations[@]}";
    do
        allennlp train $rootdir/experiments/$exp_name.json -s $rootdir/results/$exp_name-run$iter --include-package allen_linguo
        bash $rootdir/darth_linguo/pbs_scripts/generate_predictions_trial.sh $exp_name-run$iter

        echo "Finished experiment $exp_name" >> $progress_file
        let 'count+=1'  
        now=$(date +"%c")
        echo "$count/$num_runs done at $now" >> $progress_file
        
        python python/extract_results.py $exp_name-run$iter >> $resfile
        aws s3 cp $resfile s3://linguo-death-star/darth_linguo/results/consolidated/
        aws s3 cp $progress_file s3://linguo-death-star/progress/
    done
done