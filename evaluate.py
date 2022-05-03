'''
Helpful docs
https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi
Getting eyes for the robot
https://www.roboti.us/forum/index.php?threads/add-vision-to-the-model.4207/
Can MUJOCO dynamically generate models？
https://www.roboti.us/forum/index.php?threads/can-mujoco-dynamically-generate-models%EF%BC%9F.4224/

'''
from ant_mixed_long_dense import AntMixLongEnv
from ant_mixed_new_dense import AntMixEnv
import gym
import time
import numpy as np
import math
import sys
from ant_box import AntBoxEnv
#sys.path.append('/home/simon/Downloads/stable-baselines3')
from stable_baselines3 import SAC
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--env",
                    required=True,
                    choices=['antmix', 'antmixlong'],
                    type=str)
args, unknown = parser.parse_known_args()

if args.env =="antmix":
    env = AntMixEnv()
else:
    env = AntMixLongEnv()

obs = env.reset()
model = SAC.load("logs/antmix_dense_seed1/best_model")
number_eval = 50
all_reward = []
no_success = 0
for i in range(number_eval):
    #env.reset_task(i)
    s, G, done, t = env.reset(), 0, False, 0
    accum_reward = 0
    step =0
    while done == False:
        #print("obs", s)
        env.render()
        action, _states = model.predict(s, deterministic=True)

        #print("action:", action)
        #action = np.array([0.1,2])
        s, reward, done, info = env.step(action)
        #print("reward:", reward)
        #print("control_cost", info['reward_ctrl'])
        accum_reward += reward
        step += 1
        if info['subtask_success']:
            print("success")
            #breakpoint()
    print("total step:", step)
    print("total reward:", accum_reward)


    all_reward.append(accum_reward)
print("success rate:", no_success/number_eval)
print("average reward:", np.mean(all_reward))
