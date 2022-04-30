import torch
import numpy as np
import os
import gym
from . import register_env
from gym import utils
from gym.envs.mujoco import mujoco_env
import math


@register_env('ant-bridge')
class AntBridgeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, task={}, n_tasks=1, env_type='train', randomize_tasks=True):

        self._task = task
        self.env_type = env_type
        self.tasks = self.sample_tasks(n_tasks)
        self.random_steps = 5
        # these will get overiden when we call reset_task from the outside.
        self.max_step = 400
        self.goal_reward = 20
        self.outside_reward = -5
        self.floor_width = 1.5
        self.floor_backAndfront_width = 10
        self.colliding_reward = 0
        self.survive_reward = 0
        self.distanceToGoalWeight = 20
        self.wind_force_coeffecient = 0.5

        self.current_step = 0
        self._outside = False
        self.ob_shape = {"joint": [29]}
        self.ob_type = self.ob_shape.keys()
        xml_path = os.path.join(os.getcwd(), "rlkit/envs/assets/ant-bridge.xml")
        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        # self.render()

        yposbefore = self.get_body_com("agent_torso")[1]

        distanceToGoalBefore = self.distance_to_goal()
        self.do_simulation(a, self.frame_skip)
        # increment 1 step
        self.current_step += 1
        yposafter = self.get_body_com("agent_torso")[1]
        agent_xpos = self.get_body_com("agent_torso")[0]

        # forward_reward = (yposafter - yposbefore) / self.dt

        # if collide with the wall or went outside, then we shall stop and give agent a big penalty.
        outside_reward = 0
        if yposafter >= 5 and yposafter <= 21:
            if agent_xpos < -self.floor_width or agent_xpos >= self.floor_width:
                self._outside = True
                outside_reward = self.outside_reward
        else:
            if agent_xpos < -self.floor_backAndfront_width or agent_xpos >= self.floor_backAndfront_width:
                self._outside = True
                outside_reward = self.outside_reward

        # check if the agent got tipped over
        tipped_over = self.get_body_com("agent_torso")[2] <= 0.3  # or self.get_body_com("agent_torso")[2]>=1

        distanceToGoalAfter = self.distance_to_goal()
        distanceToGoalReward = 0
        distanceDifference = distanceToGoalBefore - distanceToGoalAfter
        distanceToGoalReward = distanceDifference * self.distanceToGoalWeight

        # control cost for the agent, I don't think we need it because the ball will just move, leave it for now with a smaller weight.
        ctrl_cost = 0  # 0.5 * np.square(a).sum()
        # I don't think we will have a contact cost ever.

        contact_cost = (
                0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = self.survive_reward

        state = self.state_vector()
        # check when we need to finish the current episode
        done = False
        if self._outside or self.current_step >= self.max_step or tipped_over:
            done = True
        ob = self._get_obs()
        #print("observation:", ob)
        goal_reward = 0
        success = False
        if distanceToGoalAfter < 1.2:
            done = True
            success = True
            goal_reward = self.goal_reward
        reward = outside_reward + goal_reward + distanceToGoalReward - ctrl_cost

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

        return np.concatenate(
            [
                [self.sim.data.qpos.flat[0] / 10],
                np.array([10 - self.sim.data.get_body_xpos('agent_torso')[1]]) / 10,
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:] / 5,

            ]
        )

    def reset_model(self):

        self.current_step = 0
        self.wind_force()
        self._outside = False

        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1

        self.set_state(qpos, qvel)
        for _ in range(int(self.random_steps)):
            self.do_simulation(self.unwrapped.action_space.sample(), self.frame_skip)
        return self._get_obs()

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

    def wind_force(self):
        torso_index = self.sim.model._body_name2id["agent_torso"]

        self.sim.data.xfrc_applied[torso_index][0] = self.wind_force_coeffecient

    def distance_to_goal(self):
        goal_x = 0  # self.sim.data.get_geom_xpos('goal')[0]
        goal_y = 26  # self.sim.data.get_geom_xpos('goal')[1]
        agent_x = self.get_body_com("agent_torso")[0]
        agent_y = self.get_body_com("agent_torso")[1]
        distance = math.sqrt((goal_x - agent_x) ** 2 + (goal_y - agent_y) ** 2)
        return distance

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def sample_tasks(self, num_tasks):

        if self.env_type == 'test':
            wind_force_coeffecients = np.linspace(-2.5, 2.6, num_tasks)

            tasks = [{'wind_force_coeffecient': wind_force_coeffecient} for wind_force_coeffecient in
                     wind_force_coeffecients]
        else:
            wind_force_coeffecients = np.random.uniform(-3.0, 3.0, size=(num_tasks,))

            tasks = [{'wind_force_coeffecient': wind_force_coeffecient} for wind_force_coeffecient in
                     wind_force_coeffecients]
        return tasks

    def reset_task(self, idx):

        self._task = self.tasks[idx]
        self.wind_force_coeffecient = self._task['wind_force_coeffecient']

        self.reset()
