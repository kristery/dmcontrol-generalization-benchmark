import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

fname = f'walker_stand_feat/sac_bc_1_2.5/100/eval.log'

iters = [3, 10]
lams = [0.1, 1.0, 2.5, 5.0, 10.0]
seeds = [100, 200, 300]

#envs = ['cheetah_run_feat', 'finger_spin_feat', 'walker_stand_feat']
envs = ['finger_spin_feat', 'walker_stand_feat']

def readfile(fname, values_dict):
    episodes = []
    rewards = []
    with open(fname, 'r') as f:
        for line in f:
            data = json.loads(line)
            episodes.append(float(data["episode"]))
            rewards.append(float(data["episode_reward_test_env"]))
            try:
                values_dict[float(data["episode"])].append(float(data["episode_reward_test_env"]))
            except:
                values_dict[float(data["episode"])] = [float(data["episode_reward_test_env"])]

    return np.array(episodes), np.array(rewards)


def plot_summary(env, iters, seeds):
    for it in iters:
        for lam in lams:
            eps_summary = []
            rtn_summary = []
            values_dict = {}
            eps = []
            rtn_mean = []
            rtn_std = []
            for seed in seeds:
                try:
                    episodes, rewards = readfile(f"{env}/sac_rev_{it}_{lam}/{seed}/eval.log", values_dict)
                    eps_summary.append(episodes)
                    rtn_summary.append(rewards)
                except:
                    print(f"{it}-{lam} does not exist")
            # summaries
            for key in values_dict:
                eps.append(key)
                rtn_mean.append(np.mean(values_dict[key]))
                rtn_std.append(np.std(values_dict[key]))
            rtn_mean = np.array(rtn_mean)
            rtn_std = np.array(rtn_std)
            plt.plot(eps, rtn_mean, label=f'{it}_{lam}')
            plt.fill_between(eps, rtn_mean - rtn_std, rtn_mean + rtn_std, alpha=0.3)
    plt.legend()
    plt.title(env)
    plt.show()



for env in envs:
    plot_summary(env, iters, seeds)
