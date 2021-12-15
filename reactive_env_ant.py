import torch
import numpy as np
import os
import gym
from gym import utils
from gym.envs.mujoco import mujoco_env
import math

class CustomMujocoEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, x_pos):
        self.max_step = 500
        self.passing_reward = 50
        self.goal_reward = 200
        self.outside_reward = -100
        self.floor_width = 10
        self.colliding_reward = 0
        self.survive_reward = 0.2
        self.distanceToDoorRewardWeight = 10
        self.current_step = 0
        self.x_pos = (x_pos *0.8 + 0.1 )* 20 - 10
        self._outside = False
        self._passingDoor = False
        xml_path = os.path.join(os.getcwd(), "assets/reactive-ant.xml")
        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)   
        utils.EzPickle.__init__(self)


    def step(self, a):
        #do the simulation and check how much we moved on the y axis
        yposbefore = self.get_body_com("agent_ball_body")[1]
        distanceToDoorBefore = self.distance_to_door()
        self.do_simulation(a, self.frame_skip)
        #increment 1 step
        self.current_step += 1
        yposafter = self.get_body_com("agent_ball_body")[1]
        agent_xpos = self.get_body_com("agent_ball_body")[0]
        forward_reward = (yposafter - yposbefore) / self.dt
        distanceToDoorAfter = self.distance_to_door()
        distanceToDoorReward = 0
        if not self._passingDoor:
            distanceDifference = distanceToDoorBefore-distanceToDoorAfter
            distanceToDoorReward = max(0, distanceDifference*self.distanceToDoorRewardWeight)
            #print(distanceToDoorReward)
        #if collide with the wall or went outside, then we shall stop and give agent a big penalty. 
        outside_reward = 0
        if agent_xpos<-self.floor_width or agent_xpos >=self.floor_width:
            self._outside = True
            outside_reward = self.outside_reward

        colliding_reward = 0
        if self.collision_detection("curbleft") or self.collision_detection("curbright"):
            colliding_reward = self.colliding_reward

        # if we haven't passed the door, then we can get reward when pass the door. 
        passing_reward = 0
        if yposafter >= self.sim.data.get_geom_xpos('curbleft')[1] and not self._passingDoor:
            self._passingDoor = True
            passing_reward = self.passing_reward

        #control cost for the agent, I don't think we need it because the ball will just move, leave it for now with a smaller weight. 
        ctrl_cost = 0.5 * np.square(a).sum() * 0.1 
        #I don't think we will have a contact cost ever. 
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = self.survive_reward 
        
       
        state = self.state_vector()
        #check when we need to finish the current episode
        done = False
        if self._outside or self.current_step >= self.max_step:
            done = True
        ob = self._get_obs()

        goal_reward = 0
        if self.collision_detection('goal'):
            done = True
            goal_reward = self.goal_reward
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward + outside_reward + passing_reward + goal_reward + colliding_reward + distanceToDoorReward

        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )
    #the new observation is [agent position, curb1 y axis position, agent velocity]
    def _get_obs(self):

        return np.concatenate(
            [
                self.sim.data.get_body_xpos('agent_ball_body'),
                np.array([self.sim.data.get_geom_xpos('curbleft')[1]-self.sim.data.get_body_xpos('agent_ball_body')[1]]),
                self.sim.data.qvel
            ]
        )

    def reset_model(self):
        self.current_step = 0
        self._outside = False
        self._passingDoor = False
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        
        self.set_state(qpos, qvel)

        self._put_curbs()
        
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def _put_curbs(self):
        idxleft = self.model.geom_name2id('curbleft')
        idxright = self.model.geom_name2id('curbright')

        right_curb_leftend_pos = self.x_pos + 1
        right_curb_length = 10 - right_curb_leftend_pos
        right_curb_pos = right_curb_leftend_pos + right_curb_length / 2


        self.sim.model.geom_pos[idxright][0] =   right_curb_pos
        self.sim.model.geom_size[idxright][0] =  right_curb_length / 2   

        left_curb_rightend_pos = self.x_pos - 1
        left_curb_length = left_curb_rightend_pos + 10
        left_curb_pos = left_curb_rightend_pos - left_curb_length / 2
        
        self.sim.model.geom_pos[idxleft][0] =  left_curb_pos
        self.sim.model.geom_size[idxleft][0] = left_curb_length / 2
        


    
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

    def distance_to_door(self):
        door_x = self.x_pos
        door_y = self.sim.data.get_geom_xpos('curbleft')[1]
        agent_x = self.get_body_com("agent_ball_body")[0]
        agent_y = self.get_body_com("agent_ball_body")[1]
        distance =  math.sqrt((door_x - agent_x) **2 + (door_y - agent_y) **2)

        return distance