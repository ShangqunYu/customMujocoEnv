import torch
import numpy as np
import os
import gym
from gym import utils
from gym.envs.mujoco import mujoco_env
import math


class BoxPushingEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, task={}, n_tasks=10, env_type='train', randomize_tasks=True):

        self._task = task
        self.env_type = env_type
        self.tasks = self.sample_tasks(n_tasks)
        self.max_step = 500
        self.goal_reward = 10
        self.outside_reward = -100
        self.floor_width = 10
        self.current_step = 0
        self._goal_x_pos = 0
        self._goal_y_pos = 8
        self._outside = False
        self.distanceToBoxWeight = 2

        self.ob_shape = {"joint": [7]}
        self.ob_type = self.ob_shape.keys()
        xml_path = os.path.join(os.getcwd(), "assets/box-pushing.xml")
        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self.render()
        #do the simulation and check how much we moved on the y axis
        dist_boxToGoalBefore = self.box_distance_to_goal()
        dist_ballToBoxBefore = self.distance_BallToBox()
        
        self.do_simulation(a, self.frame_skip)
        #increment 1 step
        self.current_step += 1
        dist_boxToGoalAfter = self.box_distance_to_goal()
        dist_ballToBoxAfter = self.distance_BallToBox()

        #if collide with the wall or went outside, then we shall stop and give agent a big penalty.
        outside_reward = 0
        agent_xpos = self.get_body_com("agent_ball_body")[0]
        if agent_xpos<-self.floor_width or agent_xpos >=self.floor_width:
            self._outside = True
            outside_reward = self.outside_reward

        #control cost for the agent, I don't think we need it because the ball will just move, leave it for now with a smaller weight.
        ctrl_cost = 0.5 * np.square(a).sum() * 0.1
        #I don't think we will have a contact cost ever.
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )

        distanceToGoal = dist_boxToGoalBefore - dist_boxToGoalAfter
        distanceToBox =  dist_ballToBoxBefore - dist_ballToBoxAfter

        state = self.state_vector()
        #check when we need to finish the current episode
        done = False
        if self._outside or self.current_step >= self.max_step:
            done = True
        ob = self._get_obs()

        goal_reward = 0
        success = False
        if dist_boxToGoalAfter <=1.2:
            done = True
            success = True
            goal_reward = self.goal_reward
        reward =  outside_reward  + goal_reward  + distanceToGoal + distanceToBox * self.distanceToBoxWeight

        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=0,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                success = success,
            ),
        )
    #the new observation is [agent position, curb1 y axis position, agent velocity]
    def _get_obs(self):

        return np.concatenate(
            [
                #abs position of the ball
                self.sim.data.get_body_xpos('agent_ball_body'),
                #abs position of the box 
                self.sim.data.get_body_xpos('box'),
                #relative positon from agent to box
                #self.sim.data.get_body_xpos('box')-self.sim.data.get_body_xpos('agent_ball_body'),
                #relative position from box to goal
                #self.sim.data.get_geom_xpos('goal') - self.sim.data.get_body_xpos('box'),
                #velocity of the ball
                self.sim.data.qvel[:3]
            ]
        )

    def reset_model(self):
        self.current_step = 0
        self._outside = False

        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self._put_goal()
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _put_goal(self):
        idxGoal = self.model.geom_name2id('goal')
        self.sim.model.geom_pos[idxGoal][0] = self._goal_x_pos
        self.sim.model.geom_pos[idxGoal][1] = self._goal_y_pos

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


    def box_distance_to_goal(self):
        goal_x = self.sim.data.get_geom_xpos('goal')[0]
        goal_y = self.sim.data.get_geom_xpos('goal')[1]
        box_x = self.get_body_com("box")[0]
        box_y = self.get_body_com("box")[1]
        distance =  math.sqrt((goal_x - box_x) **2 + (goal_y - box_y) **2)
        return distance

    def distance_BallToBox(self):
        ball_x = self.sim.data.get_geom_xpos('agent_ball_geom')[0]
        ball_y = self.sim.data.get_geom_xpos('agent_ball_geom')[1]
        box_x = self.get_body_com("box")[0]
        box_y = self.get_body_com("box")[1]
        distance =  math.sqrt((ball_x - box_x) **2 + (ball_y - box_y) **2)
        return distance

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def sample_tasks(self, num_tasks):

        if self.env_type == 'test':
            goal_x_pos_samplers = np.linspace(-6,6,num_tasks)
            goal_y_pos_samplers = np.random.uniform(9.9, 10.1, size=(num_tasks,))
            tasks = [{'goal_x_pos': goal_x_pos, 'goal_y_pos': goal_y_pos} for goal_x_pos, goal_y_pos in zip(goal_x_pos_samplers, goal_y_pos_samplers)]
        else:
            goal_x_pos_samplers = np.random.uniform(-6, 6, size=(num_tasks,))
            goal_y_pos_samplers = np.random.uniform(9.9, 10.1, size=(num_tasks,))
            tasks = [{'goal_x_pos': goal_x_pos, 'goal_y_pos': goal_y_pos} for goal_x_pos, goal_y_pos in zip(goal_x_pos_samplers, goal_y_pos_samplers)]
        return tasks

    def reset_task(self, idx):


        self._task = self.tasks[idx]
        self._goal_x_pos =  self._task['goal_x_pos']
        self._goal_y_pos = self._task['goal_y_pos']

        self.reset()
