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

OBSTACLE_VELOCITY = 500
FIRED_BALL = False

env = CustomMujocoEnv()

env.reset()



steps = 0 

while True:

    a = env.action_space.sample()
    print(a)
    env.step(a)
    env.render()

    steps += 1