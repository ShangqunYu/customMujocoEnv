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


env = CustomMujocoEnv(0.5)


for i in range(10):
    env.reset()
    done = False
    steps = 0 
    while not done:
        a = env.action_space.sample()

        a = np.array((0,2))
        #breakpoint()
        
        obs, reward, done, info = env.step(a)

        env.render()

        steps += 1
        print("total step:", steps)