for iters in 1 3 5 10
do

    for seed in 100 200 300
    do
       domain=cheetah
       task=run
       cuda=0
       CUDA_VISIBLE_DEVICES=${cuda} python src/feat_train.py --algorithm sac_feat_exp --domain_name $domain --task_name $task --seed $seed --iters $iters &
       CUDA_VISIBLE_DEVICES=${cuda} python src/feat_train.py --algorithm sac_feat --domain_name $domain --task_name $task --seed $seed 
    done

    for seed in 100 200 300
    do
       domain=quadruped
       task=run
       cuda=0
       CUDA_VISIBLE_DEVICES=${cuda} python src/feat_train.py --algorithm sac_feat_exp --domain_name ${domain} --task_name ${task} --seed $seed --iters $iters &
       CUDA_VISIBLE_DEVICES=${cuda} python src/feat_train.py --algorithm sac_feat --domain_name ${domain} --task_name ${task} --seed $seed
    done

done
