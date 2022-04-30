import torch
import numpy as np
import os
import gym
#from . import register_env
from gym import utils
from gym.envs.mujoco import mujoco_env
import math

#@register_env('ant-goal')
class CustomMujocoEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, task={}, n_tasks=1, env_type='train', randomize_tasks=True):

        self._task = task
        self.subtaskid = 0
        self.env_type = env_type
        #self.tasks = self.sample_tasks(n_tasks)
        #these will get overiden when we call reset_task from the outside.
        self._x_pos_sampler = 0.5
        
        self._curb_y_pos = 10
        self.random_steps = 1
        self.max_step = 5000
        self.passing_reward = 10
        self.goal_reward = 20
        self.outside_reward = -5
        self.floor_width = 10
        self.corridor_width = 3
        self.survive_reward = 0
        self.current_step = 0
        self.coin_reward_weight = 5
        self.substask_succeed_weight = 5
        self.success_reward_weight = 100
        self.goals_position_y = [25,51,67,92,118,134,159,185,201,226]
        


        self.offset_y = [0+10, 25+10, 51+8, 67+10, 92+10, 118+8, 134+10, 159+10, 185+8, 201+10]
        self.x_pos = (self._x_pos_sampler *0.8 + 0.1)* 20 - 10
        

        self.first_coins_get = [0,0,0]
        self.second_coins_get = [0,0,0]

        self.ob_shape = {"joint": [29]}
        self.ob_type = self.ob_shape.keys()
        xml_path = os.path.join(os.getcwd(), "assets/ant-goal.xml")
        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)
        self.record_init_coins_position()


    def step(self, a):
        #do the simulation and check how much we moved on the y axis

        yposbefore = self.get_body_com("agent_torso")[1]
        if self.subtaskid==1 or self.subtaskid==4 or self.subtaskid==7:
            self.wind_force(0)
        else:
            self.wind_force(0)
        distanceToGoalBefore = self.distance_to_goal(self.goals_position_y[self.subtaskid])

        self.do_simulation(a, self.frame_skip)
        self.current_step += 1
        distanceToGoalAfter = self.distance_to_goal(self.goals_position_y[self.subtaskid])
        yposafter = self.get_body_com("agent_torso")[1]
        agent_xpos = self.get_body_com("agent_torso")[0]
        tipped_over = self.get_body_com("agent_torso")[2]<=0.3
        subtask_succeed = False
        get_coin_reward = 0
        
        
        if self.subtaskid==0 or self.subtaskid==3 or self.subtaskid==6 or self.subtaskid==9:
            if distanceToGoalAfter <= 3:
                subtask_succeed = True
                
        elif self.subtaskid==1 or self.subtaskid==4 or self.subtaskid==7:
            if distanceToGoalAfter <= 1.2:
                subtask_succeed = True      
        else:
            first_coin_distance, second_coin_distance, coin_task_no = self.distance_to_coins()
            if self.first_coins_get[coin_task_no] == 0 and first_coin_distance <=1.2:
                get_coin_reward += self.coin_reward_weight
                self.first_coins_get[coin_task_no] = 1
                self.put_firstCoin_away(coin_task_no)
            if self.second_coins_get[coin_task_no] == 0 and second_coin_distance <=1.2:
                get_coin_reward += self.coin_reward_weight
                self.second_coins_get[coin_task_no] = 1
                self.put_secondCoin_away(coin_task_no)
            if distanceToGoalAfter <= 3:
                subtask_succeed = True               

        dense_reward = (distanceToGoalBefore - distanceToGoalAfter)*10    #(yposafter-yposbefore) * 10        #
        state = self.state_vector()
        #check when we need to finish the current episode
        done = False
        self.current_step += 1
        if tipped_over or self.current_step >= self.max_step:
            done = True
        ob = self._get_obs()
        goal_reward = 0
        success = False
        substask_reward = 0
        if subtask_succeed:
            self.subtaskid +=1
            substask_reward += self.substask_succeed_weight
        success_reward = 0
        if self.subtaskid ==10:
            success = True
            done = True
            success_reward += self.success_reward_weight 
        reward =  dense_reward   # success_reward + substask_reward + get_coin_reward + 
        return (
            ob,
            reward,
            done,
            dict(
                success = success,
            ),
        )
    #the new observation is [agent position, curb1 y axis position, agent velocity]

    def _get_obs_sub(self):
        relative_y = 0
        if self.subtaskid ==2 or self.subtaskid ==5 or self.subtaskid ==8:
            relative_y = np.array([self.offset_y[self.subtaskid] - self.sim.data.get_body_xpos('agent_torso')[1]])/8
            _, _, coin_task_no = self.distance_to_coins()
            return np.concatenate(
                [
                    [self.sim.data.qpos.flat[0]/6],
                    relative_y,
                    [self.first_coins_get[coin_task_no]],
                    [self.second_coins_get[coin_task_no]],
                    self.sim.data.qpos.flat[2:],
                    self.sim.data.qvel.flat[:]/5,

                ]
            )            
        else:
            relative_y = np.array([self.offset_y[self.subtaskid] - self.sim.data.get_body_xpos('agent_torso')[1]])/10
            return np.concatenate(
                [
                    [self.sim.data.qpos.flat[0]/10],
                    relative_y,
                    self.sim.data.qpos.flat[2:],
                    self.sim.data.qvel.flat[:]/5,

                ]
            )
    def _get_obs(self):
        task_id_onehot = np.zeros(10)
        task_id_onehot[self.subtaskid] = 1
        #print("current_y:", self.sim.data.get_body_xpos('agent_torso')[1])
        return np.concatenate(
            [
                [self.sim.data.qpos.flat[0] / 10],
                #np.array([self.sim.data.get_body_xpos('agent_torso')[1] - 115]) / 11.5,
                #task_id_onehot,
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:]/5,
            ]
        )

    def reset_model(self):
        self.current_step = 0
        self.first_coins_get = [0,0,0]
        self.second_coins_get = [0,0,0]
        self.subtaskid = 0
        self.set_coins()
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1

        self.set_state(qpos, qvel)

        # for _ in range(int(self.random_steps)):
        #     self.do_simulation(self.unwrapped.action_space.sample(), self.frame_skip)

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


    def wind_force(self, wind_force_coeffecient = 0.5):
        torso_index = self.sim.model._body_name2id["agent_torso"]
        self.sim.data.xfrc_applied[torso_index][0] = wind_force_coeffecient


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

    def distance_to_coins(self):
        coin_task_no = 0
        if self.subtaskid==2:
            coin_task_no = 0
        elif self.subtaskid==5:
            coin_task_no = 1
        elif self.subtaskid==8:
            coin_task_no = 2
        else:
            coin_task_no = -1
        firstcoin_name = "coin_geom1_" + str(coin_task_no)
        secondcoin_name = "coin_geom2_" + str(coin_task_no)

        agent_x = self.get_body_com('agent_torso')[0]
        agent_y = self.get_body_com('agent_torso')[1]

        idx1 = self.model.geom_name2id(firstcoin_name)
        idx2 = self.model.geom_name2id(secondcoin_name)
        first_coin_x, first_coin_y = self.sim.model.geom_pos[idx1][0] , self.sim.model.geom_pos[idx1][1]

        second_coin_x, second_coin_y = self.sim.model.geom_pos[idx2][0] , self.sim.model.geom_pos[idx2][1]
        first_coin_distance = math.sqrt((agent_x - first_coin_x) ** 2 + (agent_y - first_coin_y) ** 2)
        second_coin_distance = math.sqrt((agent_x - second_coin_x) ** 2 + (agent_y - second_coin_y) ** 2)
        return first_coin_distance, second_coin_distance, coin_task_no

    def put_firstCoin_away(self, coin_task_no):
        firstcoin_name = "coin_geom1_" + str(coin_task_no)
        idx = self.model.geom_name2id(firstcoin_name)
        self.sim.model.geom_pos[idx][0] = 0
        self.sim.model.geom_pos[idx][1] = -20

    def put_secondCoin_away(self, coin_task_no):
        secondcoin_name = "coin_geom2_" + str(coin_task_no)
        idx = self.model.geom_name2id(secondcoin_name)
        self.sim.model.geom_pos[idx][0] = 0
        self.sim.model.geom_pos[idx][1] = -22        


    def distance_to_goal(self, goal_y):
        goal_x = 0 
        goal_y = goal_y
        agent_x = self.get_body_com("agent_torso")[0]
        agent_y = self.get_body_com("agent_torso")[1]
        distance =  math.sqrt((goal_x - agent_x) **2 + (goal_y - agent_y) **2)
        return distance


    def record_init_coins_position(self):
        self._idx1 = self.sim.model.geom_name2id('coin_geom1_0')
        self._idx2 = self.sim.model.geom_name2id('coin_geom2_0')
        self._idx3 = self.sim.model.geom_name2id('coin_geom1_1')
        self._idx4 = self.sim.model.geom_name2id('coin_geom2_1')
        self._idx5 = self.sim.model.geom_name2id('coin_geom1_2')
        self._idx6 = self.sim.model.geom_name2id('coin_geom2_2')

        self._coin_geom1_1_position = np.copy(self.sim.model.geom_pos[self._idx1])
        self._coin_geom2_1_position = np.copy(self.sim.model.geom_pos[self._idx2])
        self._coin_geom1_2_position = np.copy(self.sim.model.geom_pos[self._idx3])
        self._coin_geom2_2_position = np.copy(self.sim.model.geom_pos[self._idx4])
        self._coin_geom1_3_position = np.copy(self.sim.model.geom_pos[self._idx5])
        self._coin_geom2_3_position = np.copy(self.sim.model.geom_pos[self._idx6])

    def set_coins(self):
        self.sim.model.geom_pos[self._idx1] = self._coin_geom1_1_position
        self.sim.model.geom_pos[self._idx2] = self._coin_geom2_1_position
        self.sim.model.geom_pos[self._idx3] = self._coin_geom1_2_position
        self.sim.model.geom_pos[self._idx4] = self._coin_geom2_2_position
        self.sim.model.geom_pos[self._idx5] = self._coin_geom1_3_position
        self.sim.model.geom_pos[self._idx6] = self._coin_geom2_3_position
