#Run the experiments

source activate allen

cd /home/lab/home/lab/Pablo/darth_linguo

resfile="/home/lab/home/lab/Pablo/darth_linguo/results/consolidated/exp-3.3-multirun_resultsV1"
echo "exp_name,VA,AA,RV,total" > $resfile
echo "Started experiments" > /home/lab/home/lab/Pablo/darth_linguo/results/exp3-3progress
experiments=(exp-3.2_uni-1-layer_64-182 exp-3.2_uni-2-layer_64-112, exp-3.2_uni-3-layer_64-88 exp-3.2_uni-4-layer_64-75 exp-3.2_bi-1-layer_64-121 exp-3.2_bi-2-layer_64-67 exp-3.2_bi-3-layer_64-52 exp-3.2_bi-4-layer_64-44)
let num_runs=${#experiments[@]}*${#iterations[@]}
echo $exp_num
let "count=0"
iterations=(1 2 3 4 5 6 7 8 9 10)
let "count=0"

for exp_name in "${experiments[@]}"; 
do
    for iter in "${iterations[@]}";
    do
        allennlp train /home/lab/home/lab/Pablo/darth_linguo/experiments/$exp_name.json -s /home/lab/home/lab/Pablo/darth_linguo/results/$exp_name-run$iter --include-package allen_linguo
        bash /home/lab/home/lab/Pablo/darth_linguo/pbs_scripts/generate_predictions.sh $exp_name-run$iter

        echo "Finished experiment $exp_name-run$iter" >> /home/lab/home/lab/Pablo/darth_linguo/results/exp3-3progress
        let 'count+=1'  
        now=$(date +"%c")
        echo "$count/$num_runs done at $now" >> /home/lab/home/lab/Pablo/darth_linguo/results/exp3-3progress
        
        python python/extract_results.py $exp_name-run$iter >> $resfile
        aws s3 cp $resfile s3://linguo-death-star/darth_linguo/results
    done
done