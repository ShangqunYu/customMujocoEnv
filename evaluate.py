'''
Helpful docs
https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi
Getting eyes for the robot
https://www.roboti.us/forum/index.php?threads/add-vision-to-the-model.4207/
Can MUJOCO dynamically generate models？
https://www.roboti.us/forum/index.php?threads/can-mujoco-dynamically-generate-models%EF%BC%9F.4224/

'''
from box_pushing_env import BoxPushingEnv
import gym
import time
import numpy as np
import math
import sys
sys.path.append('/home/simon/Downloads/stable-baselines3')
from stable_baselines3 import SAC

number_eval = 10
env = BoxPushingEnv()

env.reset_task(0)

obs = env.reset()
model = SAC.load("logs/best_model")

all_reward = []

for i in range(number_eval):
    s, G, done, t = env.reset(), 0, False, 0
    accum_reward = 0
    step =0
    while done == False:
        env.render()
        action, _states = model.predict(s, deterministic=True)
        #print("action:", action)
        #action = np.array([0.1,2])
        s, reward, done, info = env.step(action)
        #print("reward:", reward)
        #print("control_cost", info['reward_ctrl'])
        accum_reward += reward
        step += 1
    print("total step:", step)
    print("total reward:", accum_reward)


    all_reward.append(accum_reward)

print("average reward:", np.mean(all_reward))
