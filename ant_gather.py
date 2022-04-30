import torch
import numpy as np
import os
import gym
from . import register_env
from gym import utils
from gym.envs.mujoco import mujoco_env
import math
import random

@register_env('ant-gather')
class AntCoinCollectionEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, task={}, n_tasks=1, env_type='train', randomize_tasks=True):

        self._task = task
        self.env_type = env_type
        self.tasks = self.sample_tasks(n_tasks)
        self.random_steps = 5
        # these will get overiden when we call reset_task from the outside.
        self.max_step = 400
        self.outside_reward = -5
        self.floor_width = 10
        self.floor_backAndfront_width = 10
        self.get_first_coin = False
        self.get_second_coin = False
        self._init_first_coin_position = (0, 0)
        self._init_second_coin_position = (0, 0)

        self.distanceToGoalWeight = 30

        self.first_coin_reward = 10
        self.second_coin_reward = 10
        self.distanceToFirstCoinWeight = 10
        self.distanceToSecondCoinWeight = 20
        self.goal_reward = 20

        self.current_step = 0
        self._outside = False
        self.ob_shape = {"joint": [24]}
        self.ob_type = self.ob_shape.keys()
        xml_path = os.path.join(os.getcwd(), "rlkit/envs/assets/ant-gather.xml")
        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        # self.render()
        distanceToFirstCoinBefore, distanceToSecondCoinBefore = self.distance_to_coins()
        distanceToGoalBefore = self.distance_to_goal()

        self.do_simulation(a, self.frame_skip)
        distanceToFirstCoinAfter, distanceToSecondCoinAfter = self.distance_to_coins()
        distanceToGoalAfter = self.distance_to_goal()
        get_coin_reward = 0
        if not self.get_first_coin and (distanceToFirstCoinAfter <= 1.2):
            self.get_first_coin = True
            get_coin_reward += self.first_coin_reward
            self.put_firstCoin_away()

        if not self.get_second_coin and (distanceToSecondCoinAfter <= 1.2):
            self.get_second_coin = True
            get_coin_reward += self.second_coin_reward
            self.put_secondCoin_away()

        # increment 1 step
        self.current_step += 1
        agent_xpos, agent_ypos, agent_zpos = self.get_body_com("agent_torso")
        distance_reward = 0
        if not self.get_first_coin:
            distance_reward += (distanceToFirstCoinBefore - distanceToFirstCoinAfter) * self.distanceToFirstCoinWeight
        if self.get_first_coin and not self.get_second_coin:
            distance_reward += (
                                           distanceToSecondCoinBefore - distanceToSecondCoinAfter) * self.distanceToSecondCoinWeight
        if self.get_first_coin and self.get_second_coin:
            distance_reward += (distanceToGoalBefore - distanceToGoalAfter) * self.distanceToGoalWeight

        # if passed the cliff, will give it some dense reward to motivate the agent to get to the goal

        outside_reward = 0
        # went outside or fall off the cliff
        tipped_over = self.get_body_com("agent_torso")[2] <= 0.3
        if abs(agent_xpos) >= self.floor_width or agent_zpos < 0 or tipped_over:
            self._outside = True
            outside_reward = self.outside_reward

        # check if the agent got tipped over
        # tipped_over = self.get_body_com("agent_torso")[2] <= 0.3  # or self.get_body_com("agent_torso")[2]>=1

        # control cost for the agent, I don't think we need it because the ball will just move, leave it for now with a smaller weight.
        ctrl_cost = - 0.005 * np.square(a).sum()
        # I don't think we will have a contact cost ever.

        contact_cost = (
                0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )

        # check when we need to finish the current episode
        done = False
        if self._outside or self.current_step >= self.max_step or tipped_over:
            done = True

        ob = self._get_obs()

        goal_reward = 0
        success = False
        if distanceToGoalAfter < 1.2:
            done = True
            success = True
            goal_reward = self.goal_reward
        reward = outside_reward + goal_reward + distance_reward + get_coin_reward
        # print("ctrl_cost:", ctrl_cost, "goal_reward", goal_reward, "distance_reward", distance_reward, "coin", get_coin_reward)
        return (
            ob,
            reward,
            done,
            dict(
                reward_ctrl=-ctrl_cost,
                success=success,
            ),
        )

    # the new observation is [agent position, curb1 y axis position, agent velocity]

    def _get_obs(self):

        obs = np.concatenate(
            [
                [self.sim.data.qpos.flat[0] / 6],
                np.array([8 - self.sim.data.get_body_xpos('agent_torso')[1]]) / 8,
                [self.get_first_coin],
                [self.get_second_coin],
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:] / 5,

            ]
        )

        return obs

    def reset_model(self):
        id = random.randint(0, len(self.tasks) - 1)
        self._task = self.tasks[id]
        self.current_step = 0
        self._outside = False
        self.get_first_coin = False
        self.get_second_coin = False

        self.set_coins()
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1

        self.set_state(qpos, qvel)

        for _ in range(int(self.random_steps)):
            self.do_simulation(self.unwrapped.action_space.sample(), self.frame_skip)
        return self._get_obs()

    def put_firstCoin_away(self):
        idx = self.model.geom_name2id('coin_geom1')
        self.sim.model.geom_pos[idx][0] = 0
        self.sim.model.geom_pos[idx][1] = -20

    def put_secondCoin_away(self):
        idx = self.model.geom_name2id('coin_geom2')
        self.sim.model.geom_pos[idx][0] = 0
        self.sim.model.geom_pos[idx][1] = -22

    def set_coins(self):
        idx1 = self.model.geom_name2id('coin_geom1')
        idx2 = self.model.geom_name2id('coin_geom2')
        self.sim.model.geom_pos[idx1][0] = self._init_first_coin_position[0]
        self.sim.model.geom_pos[idx1][1] = self._init_first_coin_position[1]
        self.sim.model.geom_pos[idx2][0] = self._init_second_coin_position[0]
        self.sim.model.geom_pos[idx2][1] = self._init_second_coin_position[1]

    def distance_to_coins(self):
        agent_x = self.get_body_com('agent_torso')[0]
        agent_y = self.get_body_com('agent_torso')[1]
        # box_x = self.get_body_com("box")[0]
        # box_y = self.get_body_com("box")[1] - 2
        first_coin_x, first_coin_y = self._init_first_coin_position
        second_coin_x, second_coin_y = self._init_second_coin_position
        first_coint_distance = math.sqrt((agent_x - first_coin_x) ** 2 + (agent_y - first_coin_y) ** 2)
        second_coint_distance = math.sqrt((agent_x - second_coin_x) ** 2 + (agent_y - second_coin_y) ** 2)
        return first_coint_distance, second_coint_distance

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def collision_detection(self, ref_name=None, body_name=None):
        assert ref_name is not None
        mjcontacts = self.data.contact

        ncon = self.data.ncon
        collision = False
        for i in range(ncon):
            ct = mjcontacts[i]
            g1, g2 = ct.geom1, ct.geom2
            g1 = self.model.geom_names[g1]
            g2 = self.model.geom_names[g2]

            if body_name is not None:
                if (g1.find(ref_name) >= 0 or g2.find(ref_name) >= 0) and \
                        (g1.find(body_name) >= 0 or g2.find(body_name) >= 0):
                    collision = True
                    break
            else:
                if (g1.find(ref_name) >= 0 or g2.find(ref_name) >= 0):
                    collision = True
                    break
        return collision

    def distance_to_goal(self):
        goal_x = 0  # self.sim.data.get_geom_xpos('goal')[0]
        goal_y = 16  # self.sim.data.get_geom_xpos('goal')[1]
        agent_x = self.get_body_com("agent_torso")[0]
        agent_y = self.get_body_com("agent_torso")[1]
        distance = math.sqrt((goal_x - agent_x) ** 2 + (goal_y - agent_y) ** 2)
        return distance

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def get_box_joint_pos(self):

        # joint_OBJTx = self.sim.model.get_joint_qpos_addr('OBJTx')
        joint_OBJTy = self.sim.model.get_joint_qpos_addr('OBJTy')
        joint_OBJTz = self.sim.model.get_joint_qpos_addr('OBJTz')
        return joint_OBJTy, joint_OBJTz
        # return joint_OBJTx, joint_OBJTy, joint_OBJTz

    def get_box_joint_vel(self):
        # joint_OBJTx = self.sim.model.get_joint_qvel_addr('OBJTx')
        joint_OBJTy = self.sim.model.get_joint_qvel_addr('OBJTy')
        joint_OBJTz = self.sim.model.get_joint_qvel_addr('OBJTz')
        return joint_OBJTy, joint_OBJTz
        # return joint_OBJTx, joint_OBJTy, joint_OBJTz

    def sample_tasks(self, num_tasks):

        if self.env_type == 'test':
            first_coin_x_positions = np.linspace(-4.5, 4.5, num_tasks)
            second_coin_x_positions = -first_coin_x_positions
        else:
            first_coin_x_positions = np.random.uniform(-6, 6, size=(num_tasks,))
            second_coin_x_positions = -first_coin_x_positions

        coins_y_positions = np.random.uniform(4.9999, 5.0001, size=(num_tasks,))
        tasks = [{'first_coin_x_position': first_coin_x_position, 'second_coin_x_position': second_coin_x_position,
                  "coins_y_position": coins_y_position}
                 for first_coin_x_position, second_coin_x_position, coins_y_position in
                 zip(first_coin_x_positions, second_coin_x_positions, coins_y_positions)]

        return tasks

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._init_first_coin_position = (self._task['first_coin_x_position'], self._task['coins_y_position'])
        self._init_second_coin_position = (self._task['second_coin_x_position'], self._task['coins_y_position'] + 6)

        self.reset()
