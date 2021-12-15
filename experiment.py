'''
Helpful docs
https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi
Getting eyes for the robot 
https://www.roboti.us/forum/index.php?threads/add-vision-to-the-model.4207/
Can MUJOCO dynamically generate models？
https://www.roboti.us/forum/index.php?threads/can-mujoco-dynamically-generate-models%EF%BC%9F.4224/

'''
from reactive_env_ant import CustomMujocoEnv
import gym 
import time
import numpy as np

import sys
sys.path.append('/home/simon/Downloads/stable-baselines3')
from stable_baselines3 import SAC


env = CustomMujocoEnv(1)

obs = env.reset()

model = SAC("MlpPolicy",  env, learning_starts=10000, verbose=1)
model.learn(total_timesteps=300000, eval_env=env, eval_freq= 10000, n_eval_episodes=10,log_interval=4, eval_log_path="./logs")
model.save("sac_reactive_control")