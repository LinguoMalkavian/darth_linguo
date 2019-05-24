#Run the experiments
set -x
source activate allen

echo "Started experiments" > /home/lab/home/lab/Pablo/darth_linguo/results/exp3-2progress
experiments=(exp-3.2_uni-1-layer_64-16 exp-3.2_uni-1-layer_64-32 exp-3.2_uni-1-layer_64-64 exp-3.2_uni-1-layer_64-96 exp-3.2_uni-1-layer_64-122 exp-3.2_uni-1-layer_64-144 exp-3.2_uni-1-layer_64-164 exp-3.2_uni-1-layer_64-182 exp-3.2_uni-1-layer_64-200 exp-3.2_uni-1-layer_64-216 exp-3.2_uni-2-layer_64-12 exp-3.2_uni-2-layer_64-23 exp-3.2_uni-2-layer_64-43 exp-3.2_uni-2-layer_64-62 exp-3.2_uni-2-layer_64-77 exp-3.2_uni-2-layer_64-90 exp-3.2_uni-2-layer_64-101 exp-3.2_uni-2-layer_64-112 exp-3.2_uni-2-layer_64-122 exp-3.2_uni-2-layer_64-132 exp-3.2_uni-3-layer_64-11 exp-3.2_uni-3-layer_64-19 exp-3.2_uni-3-layer_64-34 exp-3.2_uni-3-layer_64-49 exp-3.2_uni-3-layer_64-61 exp-3.2_uni-3-layer_64-71 exp-3.2_uni-3-layer_64-80 exp-3.2_uni-3-layer_64-88 exp-3.2_uni-3-layer_64-96 exp-3.2_uni-3-layer_64-104 exp-3.2_uni-4-layer_64-9 exp-3.2_uni-4-layer_64-17 exp-3.2_uni-4-layer_64-30 exp-3.2_uni-4-layer_64-42 exp-3.2_uni-4-layer_64-52 exp-3.2_uni-4-layer_64-61 exp-3.2_uni-4-layer_64-68 exp-3.2_uni-4-layer_64-75 exp-3.2_uni-4-layer_64-82 exp-3.2_uni-4-layer_64-88 exp-3.2_bi-1-layer_64-9 exp-3.2_bi-1-layer_64-19 exp-3.2_bi-1-layer_64-39 exp-3.2_bi-1-layer_64-61 exp-3.2_bi-1-layer_64-79 exp-3.2_bi-1-layer_64-94 exp-3.2_bi-1-layer_64-108 exp-3.2_bi-1-layer_64-121 exp-3.2_bi-1-layer_64-133 exp-3.2_bi-1-layer_64-145 exp-3.2_bi-2-layer_64-7 exp-3.2_bi-2-layer_64-13 exp-3.2_bi-2-layer_64-25 exp-3.2_bi-2-layer_64-36 exp-3.2_bi-2-layer_64-46 exp-3.2_bi-2-layer_64-54 exp-3.2_bi-2-layer_64-61 exp-3.2_bi-2-layer_64-67 exp-3.2_bi-2-layer_64-74 exp-3.2_bi-2-layer_64-79 exp-3.2_bi-3-layer_64-6 exp-3.2_bi-3-layer_64-11 exp-3.2_bi-3-layer_64-20 exp-3.2_bi-3-layer_64-29 exp-3.2_bi-3-layer_64-36 exp-3.2_bi-3-layer_64-42 exp-3.2_bi-3-layer_64-47 exp-3.2_bi-3-layer_64-52 exp-3.2_bi-3-layer_64-57 exp-3.2_bi-3-layer_64-61 exp-3.2_bi-4-layer_64-5 exp-3.2_bi-4-layer_64-9 exp-3.2_bi-4-layer_64-17 exp-3.2_bi-4-layer_64-25 exp-3.2_bi-4-layer_64-30 exp-3.2_bi-4-layer_64-35 exp-3.2_bi-4-layer_64-40 exp-3.2_bi-4-layer_64-44 exp-3.2_bi-4-layer_64-48 exp-3.2_bi-4-layer_64-52)
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