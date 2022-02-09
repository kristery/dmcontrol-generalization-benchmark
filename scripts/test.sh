for seed in 500
do
   domain=quadruped
   task=run
   cuda=4
   CUDA_VISIBLE_DEVICES=${cuda} python src/feat_train.py --algorithm sac_feat --domain_name ${domain} --task_name ${task} --seed $seed &
done
