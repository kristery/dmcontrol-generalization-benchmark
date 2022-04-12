for iters in 1 3 5 10 20
do
    for seed in 100 200
    do
       cuda=0
       CUDA_VISIBLE_DEVICES=$cuda python src/feat_train.py --algorithm sac_bc --domain_name cheetah --task_name run --seed $seed --iters $iters
    done
done
