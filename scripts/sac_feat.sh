for iters in 1 3 5
do
    for seed in 100 200 300
    do
       cuda=0
       CUDA_VISIBLE_DEVICES=$cuda python src/feat_train.py --algorithm sac_feat_exp --domain_name finger --task_name spin --seed $seed --iters $iters &
       CUDA_VISIBLE_DEVICES=$cuda python src/feat_train.py --algorithm sac_feat --domain_name finger --task_name spin --seed $seed &
    done

    for seed in 100 200 300
    do
       cuda=2
       CUDA_VISIBLE_DEVICES=$cuda python src/feat_train.py --algorithm sac_feat_exp --domain_name walker --task_name stand --seed $seed --iters $iters &
       CUDA_VISIBLE_DEVICES=$cuda python src/feat_train.py --algorithm sac_feat --domain_name walker --task_name stand --seed $seed 
    done
done
