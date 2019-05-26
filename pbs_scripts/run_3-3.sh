#Run the experiments
set -x
source activate allen

cd /home/lab/home/lab/Pablo/darth_linguo

echo "Started experiments" > /home/lab/home/lab/Pablo/darth_linguo/results/exp3-2progress
experiments=(exp-3.2_uni-1-layer64-182 exp-3.2_uni-2-layer64-112, exp-3.2_uni-3-layer64-88 exp-3.2_uni-4-layer64-75 exp-3.2_bi-1-layer64-121 exp-3.2_bi-2-layer64-67 exp-3.2_bi-3-layer64-52 exp-3.2_bi-4-layer64-44)
length="${#experiments[@]}"
iterations=(1 2 3 4 5 6 7 8 9 10)
let "count=0"

for exp_name in "${experiments[@]}"; 
do
    for iter in "${iterations[@]}";
    do
        allennlp train /home/lab/home/lab/Pablo/darth_linguo/experiments/$exp_name.json -s /home/lab/home/lab/Pablo/darth_linguo/results/$exp_name-run$iter --include-package allen_linguo
        bash /home/lab/home/lab/Pablo/darth_linguo/pbs_scripts/generate_predictions.sh $exp_name-run$iter

        echo "Finished experiment $exp_name" >> /home/lab/home/lab/Pablo/darth_linguo/results/exp3-2progress
        let 'count+=1'  
        now=$(date +"%c")
        echo "$count/$length done at $now" >> /home/lab/home/lab/Pablo/darth_linguo/results/exp3-2progress
        
        aws s3 sync /home/lab/home/lab/Pablo/darth_linguo/results s3://linguo-death-star/darth_linguo/results
    done
done