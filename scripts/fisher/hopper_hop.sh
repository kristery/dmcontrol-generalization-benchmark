for seed in 100 200 300
do
	CUDA_VISIBLE_DEVICES=0 python src/feat_train.py --algorithm sac_fisher --domain_name hopper --task_name hop --f_reg 0.5 --tau_ratio 100.0 --value_w 0.1 --ov_actorupdate --seed $seed --allow_ow &
done