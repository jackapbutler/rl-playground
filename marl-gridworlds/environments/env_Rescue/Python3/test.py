import random
from typing import List

import numpy as np
from env_rescue import EnvRescue


def save_video(images: List, out_directory, fps=1):
    import imageio

    imageio.mimsave(out_directory, [np.array(img) for img in images], fps=fps)


env = EnvRescue(13, 2, 4, random.randint(0, 10000))

max_MC_iter = 1000
verbose = False
images = []

for MC_iter in range(max_MC_iter):
    if MC_iter % 10 == 0:
        og = env.get_global_obs()
        images.append((og * 255).astype(np.uint8))

    action1 = [
        random.random() - 0.5,
        random.random() - 0.5,
        20 * (random.random() - 0.5),
        0,
    ]
    action2 = [
        random.random() - 0.5,
        random.random() - 0.5,
        20 * (random.random() - 0.5),
        0,
    ]
    action_list = [action1, action2]
    env.step(action_list)

    if verbose:
        print("iter", MC_iter)
        print("agent 0 is at", env.get_agent_pos(0))
        print("agent 1 is at", env.get_agent_pos(1))

save_video(images, "./marl-gridworlds/videos/test.mp4")
