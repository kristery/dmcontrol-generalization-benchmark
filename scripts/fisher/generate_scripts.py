import os


def write_text(domain, task, fname, cuda=0):
    print(f"writing to {fname}")
    msg = "for seed in 100 200 300\ndo\n"
    # msg += f"\tCUDA_VISIBLE_DEVICES={cuda} python src/feat_train.py --algorithm sac_fisher --domain_name {domain} --task_name {task} --f_reg 0.5 --tau_ratio 100.0 --value_w 0.1 --seed $seed &\n"
    # msg += f"\tCUDA_VISIBLE_DEVICES={cuda} python src/feat_train.py --algorithm sac_fisher --domain_name {domain} --task_name {task} --f_reg 1.0 --tau_ratio 100.0 --value_w 0.1 --seed $seed &\n"
    # msg += f"\tCUDA_VISIBLE_DEVICES={cuda} python src/feat_train.py --algorithm sac_fisher --domain_name {domain} --task_name {task} --f_reg 2.0 --tau_ratio 100.0 --value_w 0.1 --seed $seed &\n"
    # msg += f"\tCUDA_VISIBLE_DEVICES={cuda} python src/feat_train.py --algorithm sac_fisher --domain_name {domain} --task_name {task} --f_reg 1.0 --tau_ratio 1.0 --value_w 0.1 --seed $seed\n"
    msg += f"\tCUDA_VISIBLE_DEVICES={cuda} python src/feat_train.py --algorithm sac_fisher --domain_name {domain} --task_name {task} --f_reg 0.5 --tau_ratio 100.0 --value_w 0.1 --ov_actorupdate --seed $seed --allow_ow &\n"
    msg += "done"
    #print(msg)
    with open(fname, 'w') as f:
        f.write(msg)

files = [f for f in os.listdir('.') if os.path.isfile(f) and 'sh' in f and f[0] != '.']

for fname in files:
    f = fname.split('.')[0]
    f = f.split('_')

    domain = f[0]
    task = f[1]

    write_text(domain, task, fname)
