#Run the experiments

source activate allen

cd /home/lab/home/lab/Pablo/darth_linguo

resfile="/home/lab/home/lab/Pablo/darth_linguo/results/consolidated/exp-3.3-multirun_resultsV1"
echo "exp_name,VA,AA,RV,total" > $resfile
prog_file="/home/lab/home/lab/Pablo/darth_linguo/results/consolidated/exp-3.3-multirun_progress"
echo "Started experiments" > $prog_file
experiments=(exp-3.3_uni-1-layer64-182_run0 exp-3.3_uni-1-layer64-182_run1 exp-3.3_uni-1-layer64-182_run2 exp-3.3_uni-1-layer64-182_run3 exp-3.3_uni-1-layer64-182_run4 exp-3.3_uni-1-layer64-182_run5 exp-3.3_uni-1-layer64-182_run6 exp-3.3_uni-1-layer64-182_run7 exp-3.3_uni-1-layer64-182_run8 exp-3.3_uni-1-layer64-182_run9 exp-3.3_uni-2-layer64-112_run0 exp-3.3_uni-2-layer64-112_run1 exp-3.3_uni-2-layer64-112_run2 exp-3.3_uni-2-layer64-112_run3 exp-3.3_uni-2-layer64-112_run4 exp-3.3_uni-2-layer64-112_run5 exp-3.3_uni-2-layer64-112_run6 exp-3.3_uni-2-layer64-112_run7 exp-3.3_uni-2-layer64-112_run8 exp-3.3_uni-2-layer64-112_run9 exp-3.3_uni-3-layer64-88_run0 exp-3.3_uni-3-layer64-88_run1 exp-3.3_uni-3-layer64-88_run2 exp-3.3_uni-3-layer64-88_run3 exp-3.3_uni-3-layer64-88_run4 exp-3.3_uni-3-layer64-88_run5 exp-3.3_uni-3-layer64-88_run6 exp-3.3_uni-3-layer64-88_run7 exp-3.3_uni-3-layer64-88_run8 exp-3.3_uni-3-layer64-88_run9 exp-3.3_uni-4-layer64-75_run0 exp-3.3_uni-4-layer64-75_run1 exp-3.3_uni-4-layer64-75_run2 exp-3.3_uni-4-layer64-75_run3 exp-3.3_uni-4-layer64-75_run4 exp-3.3_uni-4-layer64-75_run5 exp-3.3_uni-4-layer64-75_run6 exp-3.3_uni-4-layer64-75_run7 exp-3.3_uni-4-layer64-75_run8 exp-3.3_uni-4-layer64-75_run9 exp-3.3_bi-1-layer_64-121_run0 exp-3.3_bi-1-layer_64-121_run1 exp-3.3_bi-1-layer_64-121_run2 exp-3.3_bi-1-layer_64-121_run3 exp-3.3_bi-1-layer_64-121_run4 exp-3.3_bi-1-layer_64-121_run5 exp-3.3_bi-1-layer_64-121_run6 exp-3.3_bi-1-layer_64-121_run7 exp-3.3_bi-1-layer_64-121_run8 exp-3.3_bi-1-layer_64-121_run9 exp-3.3_bi-2-layer_64-67_run0 exp-3.3_bi-2-layer_64-67_run1 exp-3.3_bi-2-layer_64-67_run2 exp-3.3_bi-2-layer_64-67_run3 exp-3.3_bi-2-layer_64-67_run4 exp-3.3_bi-2-layer_64-67_run5 exp-3.3_bi-2-layer_64-67_run6 exp-3.3_bi-2-layer_64-67_run7 exp-3.3_bi-2-layer_64-67_run8 exp-3.3_bi-2-layer_64-67_run9 exp-3.3_bi-3-layer_64-52_run0 exp-3.3_bi-3-layer_64-52_run1 exp-3.3_bi-3-layer_64-52_run2 exp-3.3_bi-3-layer_64-52_run3 exp-3.3_bi-3-layer_64-52_run4 exp-3.3_bi-3-layer_64-52_run5 exp-3.3_bi-3-layer_64-52_run6 exp-3.3_bi-3-layer_64-52_run7 exp-3.3_bi-3-layer_64-52_run8 exp-3.3_bi-3-layer_64-52_run9 exp-3.3_bi-4-layer_64-44_run0 exp-3.3_bi-4-layer_64-44_run1 exp-3.3_bi-4-layer_64-44_run2 exp-3.3_bi-4-layer_64-44_run3 exp-3.3_bi-4-layer_64-44_run4 exp-3.3_bi-4-layer_64-44_run5 exp-3.3_bi-4-layer_64-44_run6 exp-3.3_bi-4-layer_64-44_run7 exp-3.3_bi-4-layer_64-44_run8 exp-3.3_bi-4-layer_64-44_run9)
let exp_num=${#experiments[@]}
echo $exp_num
let "count=0"


for exp_name in "${experiments[@]}"; 
do
    allennlp train /home/lab/home/lab/Pablo/darth_linguo/experiments/$exp_name.json -s /home/lab/home/lab/Pablo/darth_linguo/results/$exp_name --include-package allen_linguo
    bash /home/lab/home/lab/Pablo/darth_linguo/pbs_scripts/generate_predictions.sh $exp_name

    echo "Finished experiment $exp_name" >> prog_file
    let 'count+=1'  
    now=$(date +"%c")
    echo "$count/$exp_num done at $now" >> prog_file
    
    python python/extract_results.py $exp_name >> $resfile
    aws s3 cp $resfile s3://linguo-death-star/darth_linguo/results
    aws s3 cp $prog_file s3://linguo-death-star/darth_linguo/results

done