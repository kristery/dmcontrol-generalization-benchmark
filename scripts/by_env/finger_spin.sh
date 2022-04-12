for iters in 3 10
do
    for seed in 100 200 300
    do
       for lam in 0.1 1 5 10
       do
          cuda=0
          CUDA_VISIBLE_DEVICES=$cuda python src/feat_train.py --algorithm sac_rev --domain_name finger --task_name spin --seed $seed --iters $iters --lam $lam
       done 
   done
done
