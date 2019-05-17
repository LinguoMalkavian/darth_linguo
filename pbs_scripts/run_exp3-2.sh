#Run the experiments
set -x
source activate allen

echo "Started experiments" > /home/lab/home/lab/Pablo/darth_linguo/results/exp3-2progress
experiments=(exp-3.2_uni-1-layer64-16 exp-3.2_uni-1-layer64-32 exp-3.2_uni-1-layer64-64 exp-3.2_uni-1-layer64-96 exp-3.2_uni-1-layer64-122 exp-3.2_uni-1-layer64-144 exp-3.2_uni-1-layer64-164 exp-3.2_uni-1-layer64-182 exp-3.2_uni-1-layer64-200 exp-3.2_uni-1-layer64-216 exp-3.2_uni-2-layer64-12 exp-3.2_uni-2-layer64-23 exp-3.2_uni-2-layer64-43 exp-3.2_uni-2-layer64-62 exp-3.2_uni-2-layer64-77 exp-3.2_uni-2-layer64-90 exp-3.2_uni-2-layer64-101 exp-3.2_uni-2-layer64-112 exp-3.2_uni-2-layer64-122 exp-3.2_uni-2-layer64-132 exp-3.2_uni-3-layer64-11 exp-3.2_uni-3-layer64-19 exp-3.2_uni-3-layer64-34 exp-3.2_uni-3-layer64-49 exp-3.2_uni-3-layer64-61 exp-3.2_uni-3-layer64-71 exp-3.2_uni-3-layer64-80 exp-3.2_uni-3-layer64-88 exp-3.2_uni-3-layer64-96 exp-3.2_uni-3-layer64-104 exp-3.2_uni-4-layer64-9 exp-3.2_uni-4-layer64-17 exp-3.2_uni-4-layer64-30 exp-3.2_uni-4-layer64-42 exp-3.2_uni-4-layer64-52 exp-3.2_uni-4-layer64-61 exp-3.2_uni-4-layer64-68 exp-3.2_uni-4-layer64-75 exp-3.2_uni-4-layer64-82 exp-3.2_uni-4-layer64-88 exp-3.2_bi-1-layer64-9 exp-3.2_bi-1-layer64-19 exp-3.2_bi-1-layer64-39 exp-3.2_bi-1-layer64-61 exp-3.2_bi-1-layer64-79 exp-3.2_bi-1-layer64-94 exp-3.2_bi-1-layer64-108 exp-3.2_bi-1-layer64-121 exp-3.2_bi-1-layer64-133 exp-3.2_bi-1-layer64-145 exp-3.2_bi-2-layer64-7 exp-3.2_bi-2-layer64-13 exp-3.2_bi-2-layer64-25 exp-3.2_bi-2-layer64-36 exp-3.2_bi-2-layer64-46 exp-3.2_bi-2-layer64-54 exp-3.2_bi-2-layer64-61 exp-3.2_bi-2-layer64-67 exp-3.2_bi-2-layer64-74 exp-3.2_bi-2-layer64-79 exp-3.2_bi-3-layer64-6 exp-3.2_bi-3-layer64-11 exp-3.2_bi-3-layer64-20 exp-3.2_bi-3-layer64-29 exp-3.2_bi-3-layer64-36 exp-3.2_bi-3-layer64-42 exp-3.2_bi-3-layer64-47 exp-3.2_bi-3-layer64-52 exp-3.2_bi-3-layer64-57 exp-3.2_bi-3-layer64-61 exp-3.2_bi-4-layer64-5 exp-3.2_bi-4-layer64-9 exp-3.2_bi-4-layer64-17 exp-3.2_bi-4-layer64-25 exp-3.2_bi-4-layer64-30 exp-3.2_bi-4-layer64-35 exp-3.2_bi-4-layer64-40 exp-3.2_bi-4-layer64-44 exp-3.2_bi-4-layer64-48 exp-3.2_bi-4-layer64-52)
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