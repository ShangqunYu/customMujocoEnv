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
import argparse
from stable_baselines3.common.logger import configure
import sys
from stable_baselines3 import SAC


parser = argparse.ArgumentParser()
parser.add_argument("--logdir",
                    required=False,
                    default="run0",
                    type=str)
parser.add_argument("--env",
                    required=True,
                    choices=['antmix', 'antmixlong'],
                    type=str)

args, unknown = parser.parse_known_args()
logpath = "./logs/" + args.logdir
new_logger = configure(logpath, ["stdout","csv"])
new_logger.info()
if args.env =="antmix":
    env = AntMixEnv()
else:
    env = AntMixLongEnv()


obs = env.reset()


model = SAC("MlpPolicy",  env, learning_starts=10000, verbose=2)
model.set_logger(new_logger)
model.learn(total_timesteps=2000000, eval_env=env, eval_freq= 10000,
            n_eval_episodes=10,log_interval=4, eval_log_path=logpath)
model.save("sac_reactive_control")
