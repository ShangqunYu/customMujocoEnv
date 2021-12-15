import torch
import numpy as np
import os
import gym
from gym import utils
from gym.envs.mujoco import mujoco_env

class CustomMujocoEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, x_pos):
        self.max_step = 500
        self.passing_reward = 50
        self.current_step = 0
        self.x_pos = (x_pos *0.8 + 0.1 )* 20 - 10
        self._collide = False
        self._passingDoor = False
        xml_path = os.path.join(os.getcwd(), "assets/reactive-ant.xml")
        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)   
        utils.EzPickle.__init__(self)


    def step(self, a):
        yposbefore = self.get_body_com("agent_ball_body")[1]
        self.do_simulation(a, self.frame_skip)
        self.current_step += 1
        yposafter = self.get_body_com("agent_ball_body")[1]
        agent_xpos = self.get_body_com("agent_ball_body")[0]

        #if collide with the wall or went outside, then we shall stop and give agent a big penalty. 
        colliding_reward = 0
        if self.collision_detection('curbleft') or self.collision_detection('curbright') or agent_xpos<-10 or agent_xpos >=10:
            self._collide = True
            colliding_reward = -100

        # if we haven't passed the door, then we can get reward when pass the door. 
        passing_reward = 0
        if yposafter >= self.sim.data.get_geom_xpos('curbleft')[1] and not self._passingDoor:
            self._passingDoor = True
            passing_reward = self.passing_reward


            
        forward_reward = (yposafter - yposbefore) / self.dt
        ctrl_cost = 0.5 * np.square(a).sum() * 0.1 #simon: make the control cost smaller for now
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward + colliding_reward + passing_reward
        #("contact_cost:",contact_cost,"forward_reward", forward_reward, "ctrl_cost:", ctrl_cost, "survive_reward", survive_reward)

        state = self.state_vector()
        #check when we need to finish the current episode
        done = False
        if self._collide or self.current_step >= self.max_step:
            done = True
        ob = self._get_obs()


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
                # self.sim.data.qpos.flat[2:],
                # self.sim.data.qvel.flat,
                # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
                self.sim.data.get_body_xpos('agent_ball_body'),
                np.array([self.sim.data.get_geom_xpos('curbleft')[1]]),
                self.sim.data.qvel
            ]
        )

    def reset_model(self):
        self.current_step = 0
        self._collide = False
        self._passingDoor = False
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self._put_curbs()
        self.set_state(qpos, qvel)
        
        

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def _put_curbs(self):
        idxleft = self.model.geom_name2id('curbleft')
        idxright = self.model.geom_name2id('curbright')
        right_curb_leftend_pos = self.x_pos + 1
        right_curb_length = 10 - right_curb_leftend_pos
        right_curb_pos = right_curb_leftend_pos + right_curb_length / 2

        left_curb_rightend_pos = self.x_pos - 1
        left_curb_length = left_curb_rightend_pos + 10
        left_curb_pos = left_curb_rightend_pos - left_curb_length / 2
        
        self.model.geom_pos[idxleft][0] = left_curb_pos
        self.model.geom_size[idxleft][0] = left_curb_length / 2
        
        self.model.geom_pos[idxright][0] = right_curb_pos
        self.model.geom_size[idxright][0] = right_curb_length / 2   



        #self.model.geom_pos[idx][0] = x_pos
        # self.model.geom_pos[idx][2] = h
        # self.model.geom_size[idx][2] = h

        # pos = self.model.geom_pos[idx][0]
        # size = self.model.geom_size[idx][0]
        # self._curbs = {'pos': pos, 'size': size}

    
    def collision_detection(self, ref_name=None, body_name=None):
        assert ref_name is not None
        mjcontacts = self.data.contact
        #breakpoint()
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