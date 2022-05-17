for seed in 100 200 300
do
	CUDA_VISIBLE_DEVICES=0 python src/feat_train.py --algorithm sac_feat --domain_name walker --task_name run --seed $seed
done