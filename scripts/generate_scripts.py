import os


def write_text(domain, task, fname, cuda=0):
    print(f"writing to {fname}")
    msg = "for seed in 100 200 300\ndo\n"
    msg += f"\tCUDA_VISIBLE_DEVICES={cuda} python src/feat_train.py --algorithm sac_fisher --domain_name {domain} --task_name {task} --seed $seed\n"
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