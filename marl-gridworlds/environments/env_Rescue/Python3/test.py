import random
from typing import List

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from env_rescue import EnvRescue

matplotlib.use("TkAgg")

env = EnvRescue(13, 2, 4, random.randint(0, 10000))

max_MC_iter = 1000
verbose = False
save_video_replay = True
snapshot_interval = 5
images = []


def save_video(images: List[np.ndarray]):
    frames = []  # for storing the generated images
    fig = plt.figure()
    for image in images:
        frames.append([plt.imshow(image, animated=True)])
    ani = animation.ArtistAnimation(
        fig, frames, interval=50, blit=True, repeat_delay=1000
    )
    ani.save("./marl-gridworlds/videos/replay.mp4")


for MC_iter in range(max_MC_iter):
    if MC_iter % 2 == 0 and save_video_replay:
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

if save_video_replay:
    save_video(images)
