from env_rescue import EnvRescue
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random

env = EnvRescue(13, 2, 4, random.randint(0, 10000))

max_MC_iter = 1000
save_fig = False

for MC_iter in range(max_MC_iter):
    if MC_iter%100 ==0 and save_fig:
        fig = plt.figure(figsize=(5, 5))
        gs = GridSpec(3, 2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        plt.xticks([])
        plt.yticks([])
        ax2 = fig.add_subplot(gs[2, 0:1])
        plt.xticks([])
        plt.yticks([])
        ax3 = fig.add_subplot(gs[2, 1:2])
        plt.xticks([])
        plt.yticks([])

        ax1.imshow(env.get_global_obs())
        ax2.imshow(env.get_agt_obs(0))
        ax3.imshow(env.get_agt_obs(1))
        
        fig.savefig(f'./marl-gridworlds/figures/snapshot_{MC_iter}.png')

    # obs_list = env.get_obs()
    action1 = [random.random()-0.5, random.random()-0.5, 20*(random.random()-0.5), 0]
    action2 = [random.random()-0.5, random.random()-0.5, 20*(random.random()-0.5), 0]
    action_list = [action1, action2]
    env.step(action_list)
    
    print('iter', MC_iter)
    print('agent 0 is at',env.get_agent_pos(0))
    print('agent 1 is at',env.get_agent_pos(1))

