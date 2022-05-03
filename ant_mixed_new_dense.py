import torch
import numpy as np
import os
import gym
from gym import utils
from gym.envs.mujoco import mujoco_env
import math

ANT_GATHER_LENGTH = 16
ANT_GOAL_LENGTH = 25
ANT_BRIDGE_LENGTH = 26



class AntMixEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, task={}, n_tasks=1, env_type='train', randomize_tasks=True):
        self._task_sets = ["antgoal0", "antbrid0", "antgath0", "antgoal1", "antbrid1", "antgath1", "antgoal2",
                           "antbrid2", "antgath2", "antgoal3"]
        self.task_order = np.arange(10)
        self._task = task
        self.subtaskid = 0
        self.subtasktypes = ["antgoal", "antbridge", "antgather", "antgoal", "antbridge", "antgather", "antgoal",
                             "antbridge", "antgather", "antgoal"]
        self.env_type = env_type
        # self.tasks = self.sample_tasks(n_tasks)
        # these will get overiden when we call reset_task from the outside.
        self._x_pos_sampler = 0.5
        self._curb_y_pos = 10
        self.random_steps = 1
        self.max_step = 3500
        self.passing_reward = 10
        self.goal_reward = 20
        self.outside_reward = -5
        self.floor_width = 10
        if env_type == 'train':
            self.corridor_width = 3
        elif env_type == 'test':
            self.corridor_width = 3
        self.corridor_pos = [0, 0, 0, 0]
        self.survive_reward = 0
        self.current_step = 0
        self.windforce = [0, 0, 0]
        self.coin_pos = [0, 0, 0]
        self.passing_door = [0,0,0,0]
        self.coin_reward_weight = 5
        self.substask_succeed_weight = 5
        self.success_reward_weight = 100
        self.goals_position_y = [25, 51, 67, 92, 118, 134, 159, 185, 201, 226]
        self.offset_y = [0 + 10, 25 + 10, 51 + 8, 67 + 10, 92 + 10, 118 + 8, 134 + 10, 159 + 10, 185 + 8, 201 + 10]
        self.x_pos = (self._x_pos_sampler * 0.8 + 0.1) * 20 - 10
        self.count_down = 0
        self.first_coins_get = [0, 0, 0]
        self.second_coins_get = [0, 0, 0]
        #self.task_order = np.random.choice(10, 10, replace=False)
        print("type:", env_type)
        print("order:", self.task_order)
        self.task_order = np.array([0,1,2,3,4,5,6,7,8,9])
        self.ob_shape = {"joint": [29]}
        self.ob_type = self.ob_shape.keys()
        xml_path = os.path.join(os.getcwd(), "./assets/ant-mix2.xml")
        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a, skillpolicy=False, id=None):
        # do the simulation and check how much we moved on the y axis
        yposbefore = self.get_body_com("agent_torso")[1]
        id_subtask = int(self._task_sets[self.task_order[self.subtaskid]][-1])
        if self.subtasktypes[self.subtaskid] == "antbridge":

            force = self.windforce[id_subtask]
            self.wind_force(force)
        else:
            self.wind_force(0)
        distanceToGoalBefore = self.distance_to_goal(self.goals_position_y[self.subtaskid])
        #distanceToDoorBefore will only be correct when the current task is a door task
        distanceToDoorBefore = self.distance_to_door(id_subtask, self.offset_y[self.subtaskid])
        first_coin_distance_Before, second_coin_distance_Before, coin_task_no = self.distance_to_coins()
        self.do_simulation(a, self.frame_skip)
        distanceToGoalAfter = self.distance_to_goal(self.goals_position_y[self.subtaskid])
        yposafter = self.get_body_com("agent_torso")[1]
        agent_xpos = self.get_body_com("agent_torso")[0]
        tipped_over = self.get_body_com("agent_torso")[2] <= 0.3
        subtask_succeed = False
        get_coin_reward = 0
        dense_reward = 0
        if self.subtasktypes[self.subtaskid] == "antgoal":
            # if we haven't passed the door
            if self.passing_door[id_subtask] ==0:
                distanceToDoorAfter = self.distance_to_door(id_subtask, self.offset_y[self.subtaskid])
                distanceDifference = distanceToDoorBefore - distanceToDoorAfter
                dense_reward = distanceDifference * 10
                # if the agent hasn't pass the door and agent y postion is bigger than the door, passed
                if self.sim.data.get_body_xpos('agent_torso')[1] >= self.offset_y[self.subtaskid]:
                    self.passing_door[id_subtask] = 1
            else:
                distanceDifference = distanceToGoalBefore - distanceToGoalAfter
                dense_reward = distanceDifference * 10
            if distanceToGoalAfter <= 2:
                subtask_succeed = True

        elif self.subtasktypes[self.subtaskid] == "antbridge":
            dense_reward = (distanceToGoalBefore - distanceToGoalAfter) * 10
            if distanceToGoalAfter <= 1.2:
                subtask_succeed = True
        else:
            first_coin_distance, second_coin_distance, coin_task_no = self.distance_to_coins()
            if self.first_coins_get[coin_task_no] == 0:
                dense_reward = (first_coin_distance_Before - first_coin_distance) * 10
                if first_coin_distance <= 1.2:
                    get_coin_reward += self.coin_reward_weight
                    self.first_coins_get[coin_task_no] = 1
                    self.put_firstCoin_away(coin_task_no)

            if self.second_coins_get[coin_task_no] == 0 and self.first_coins_get[coin_task_no] == 1:
                dense_reward = (second_coin_distance_Before - second_coin_distance) * 10
                if second_coin_distance <= 1.2:
                    get_coin_reward += self.coin_reward_weight
                    self.second_coins_get[coin_task_no] = 1
                    self.put_secondCoin_away(coin_task_no)

            if self.first_coins_get[coin_task_no] and self.second_coins_get[coin_task_no]:
                dense_reward = (distanceToGoalBefore - distanceToGoalAfter) * 10
                if distanceToGoalAfter <= 1.3:
                    subtask_succeed = True


        state = self.state_vector()
        # check when we need to finish the current episode
        done = False
        self.current_step += 1
        self.count_down += 1
        if self.count_down >500:
            done = True
        if tipped_over or self.current_step >= self.max_step:
            done = True
        goal_reward = 0
        success = False
        substask_reward = 0
        if subtask_succeed:
            self.count_down = 0
            self.subtaskid += 1
            substask_reward += self.substask_succeed_weight
        success_reward = 0
        if self.subtaskid == 10:
            success = True
            done = True
            success_reward += self.success_reward_weight
            self.subtaskid = 9
        if not skillpolicy:
            ob = self._get_obs()
        else:
            ob = self._get_obs_sub(id)
        reward = success_reward + substask_reward*2 + get_coin_reward + dense_reward
        return (
            ob,
            reward,
            done,
            dict(
                success=success,
                subtask_success=subtask_succeed,
                sparse_reward = success_reward + substask_reward
            ),
        )

    # the new observation is [agent position, curb1 y axis position, agent velocity]

    def _get_obs_sub(self, id=None):
        relative_y = 0
        if id == 2:
            relative_y = np.array([self.offset_y[self.subtaskid] - self.sim.data.get_body_xpos('agent_torso')[1]]) / 8
            _, _, coin_task_no = self.distance_to_coins()
            return np.concatenate(
                [
                    [self.sim.data.qpos.flat[0] / 6],
                    relative_y,
                    [self.first_coins_get[coin_task_no]],
                    [self.second_coins_get[coin_task_no]],
                    self.sim.data.qpos.flat[2:],
                    self.sim.data.qvel.flat[:] / 5,

                ]
            )
        elif id == 0 or id == 1:
            relative_y = np.array([self.offset_y[self.subtaskid] - self.sim.data.get_body_xpos('agent_torso')[1]]) / 10
            return np.concatenate(
                [
                    [self.sim.data.qpos.flat[0] / 10],
                    relative_y,
                    self.sim.data.qpos.flat[2:],
                    self.sim.data.qvel.flat[:] / 5,

                ]
            )

    def _get_obs(self):
        task_type_onehot = np.zeros(3)
        x_pos = 0
        ith_subtask = int(self._task_sets[self.task_order[self.subtaskid]][-1])
        if self.subtasktypes[self.subtaskid] == "antgoal":
            task_type_onehot[0] = 1
            x_pos = self.corridor_pos[ith_subtask] / 3  # normalization

        elif self.subtasktypes[self.subtaskid] == "antbridge":
            task_type_onehot[1] = 1
            x_pos = self.windforce[ith_subtask] / 2  # normalization
        elif self.subtasktypes[self.subtaskid] == "antgather":
            x_pos = self.coin_pos[ith_subtask] / 3  # normalization
            task_type_onehot[2] = 1
        task_id_onehot = np.zeros(10)
        task_id_onehot[self.subtaskid] = 1
        return np.concatenate(
            [
                [self.sim.data.qpos.flat[0] / 10],
                np.array([self.sim.data.get_body_xpos('agent_torso')[1] - 115]) / 11.5,
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:] / 5,
                task_id_onehot,
                self.first_coins_get,
                self.second_coins_get
            ]
        )

    def reset_model(self):
        self.combine_subtask(fixed=True)
        self.current_step = 0
        self.first_coins_get = [0, 0, 0]
        self.second_coins_get = [0, 0, 0]
        self.windforce = [-2,0,2]
        self.passing_door = [0, 0, 0, 0]
        #self.windforce = np.random.uniform(-2, 2, 3)
        self.corridor_pos = [0, 0, 0, 0]
        self.coin_pos = [0, 0, 0]
        self.subtaskid = 0
        self.count_down = 0
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1

        self.set_state(qpos, qvel)

        # for _ in range(int(self.random_steps)):
        #     self.do_simulation(self.unwrapped.action_space.sample(), self.frame_skip)

        return self._get_obs()

    def combine_subtask(self, fixed=True):
        if not fixed:
            self.task_order = np.random.choice(10, 10, replace=False)

        self.goals_position_y = []
        self.offset_y = []
        self.subtasktypes = []
        last_goal_position = 0

        for task_no in self.task_order:
            if self._task_sets[task_no][:7] == "antgoal":
                idx_plane = self.model.geom_name2id("antgoal" + str(self._task_sets[task_no][-1]) + "_plane")
                idx_curbleft = self.model.geom_name2id("curbleft" + str(self._task_sets[task_no][-1]))
                idx_curbright = self.model.geom_name2id("curbright" + str(self._task_sets[task_no][-1]))
                ith_antgoal = int(self._task_sets[task_no][-1])
                self._set_curbs_xposition(idx_curbleft, idx_curbright, ith_antgoal)
                self.sim.model.geom_pos[idx_plane][1] = last_goal_position + ANT_GOAL_LENGTH / 2
                self.sim.model.geom_pos[idx_curbleft][1] = last_goal_position + 10
                self.sim.model.geom_pos[idx_curbright][1] = last_goal_position + 10
                self.goals_position_y.append(ANT_GOAL_LENGTH + last_goal_position)
                self.offset_y.append(last_goal_position + 10)
                self.subtasktypes.append("antgoal")

            elif self._task_sets[task_no][:7] == "antbrid":
                idx_frontplane = self.model.geom_name2id(
                    "antbridge" + str(self._task_sets[task_no][-1]) + "_frontplane")
                idx_rearplane = self.model.geom_name2id("antbridge" + str(self._task_sets[task_no][-1]) + "_rearplane")
                idx_bridge = self.model.geom_name2id("bridge" + str(self._task_sets[task_no][-1]))

                self.sim.model.geom_pos[idx_frontplane][1] = last_goal_position + 2.5
                self.sim.model.geom_pos[idx_bridge][1] = last_goal_position + 5 + 8
                self.sim.model.geom_pos[idx_rearplane][1] = last_goal_position + 5 + 16 + 2.5
                self.goals_position_y.append(ANT_BRIDGE_LENGTH + last_goal_position)
                self.offset_y.append(last_goal_position + 10)
                self.subtasktypes.append("antbridge")

            elif self._task_sets[task_no][:7] == "antgath":

                idx_plane = self.model.geom_name2id("antgather" + str(self._task_sets[task_no][-1]) + "_plane")
                self.sim.model.geom_pos[idx_plane][1] = last_goal_position + ANT_GATHER_LENGTH / 2
                coin1 = "coin_geom1_" + str(self._task_sets[task_no][-1])
                coin2 = "coin_geom2_" + str(self._task_sets[task_no][-1])
                ith_antgather = int(self._task_sets[task_no][-1])
                self.set_coins(last_goal_position, coin1, coin2, ith_antgather)
                self.goals_position_y.append(ANT_GATHER_LENGTH + last_goal_position)
                self.offset_y.append(last_goal_position + 8)
                self.subtasktypes.append("antgather")

            else:
                raise NameError('Wrong subtask type')

            last_goal_position = self.goals_position_y[-1]

    def set_coins(self, start_position, coin1, coin2, ith_antgather):
        idx1 = self.model.geom_name2id(coin1)
        idx2 = self.model.geom_name2id(coin2)
        # first_coin_x_position = 3
        if ith_antgather == 0:
            first_coin_x_position = 3
        elif ith_antgather == 1:
            first_coin_x_position = 2
        elif ith_antgather == 2:
            first_coin_x_position = 1
        else:
            first_coin_x_position = -1
        #first_coin_x_position = np.random.uniform(-3, 3)
        first_coin_position = (first_coin_x_position, 5)
        second_coin_position = (-first_coin_x_position, 11)
        self.coin_pos[ith_antgather] = first_coin_x_position
        self.sim.model.geom_pos[idx1][0] = first_coin_position[0]
        self.sim.model.geom_pos[idx1][1] = first_coin_position[1] + start_position
        self.sim.model.geom_pos[idx2][0] = second_coin_position[0]
        self.sim.model.geom_pos[idx2][1] = second_coin_position[1] + start_position

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def wind_force(self, force):
        torso_index = self.sim.model._body_name2id["agent_torso"]
        self.sim.data.xfrc_applied[torso_index][0] = force

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

        coin_task_no = int(self._task_sets[self.task_order[self.subtaskid]][-1])
        if coin_task_no >= 3:
            coin_task_no = 2
        firstcoin_name = "coin_geom1_" + str(coin_task_no)
        secondcoin_name = "coin_geom2_" + str(coin_task_no)

        agent_x = self.get_body_com('agent_torso')[0]
        agent_y = self.get_body_com('agent_torso')[1]

        idx1 = self.model.geom_name2id(firstcoin_name)
        idx2 = self.model.geom_name2id(secondcoin_name)
        first_coin_x, first_coin_y = self.sim.model.geom_pos[idx1][0], self.sim.model.geom_pos[idx1][1]

        second_coin_x, second_coin_y = self.sim.model.geom_pos[idx2][0], self.sim.model.geom_pos[idx2][1]
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

    def _set_curbs_xposition(self, idxleft, idxright, ith_antgoal):
        x_pos_sampler = 0.3
        if ith_antgoal == 0:
            x_pos_sampler = 0.3
        elif ith_antgoal == 1:
            x_pos_sampler = 0.4
        elif ith_antgoal == 2:
            x_pos_sampler = 0.6
        elif ith_antgoal == 3:
            x_pos_sampler = 0.7
        else:
            x_pos_sampler = 0.5
        #x_pos_sampler = np.random.uniform(0.3, 0.7)
        x_pos = (x_pos_sampler * 0.8 + 0.1) * 20 - 10
        self.corridor_pos[ith_antgoal] = x_pos

        right_curb_leftend_pos = x_pos + self.corridor_width / 2
        right_curb_length = 10 - right_curb_leftend_pos
        right_curb_pos = right_curb_leftend_pos + right_curb_length / 2

        self.sim.model.geom_pos[idxright][0] = right_curb_pos
        self.sim.model.geom_size[idxright][0] = right_curb_length / 2

        left_curb_rightend_pos = x_pos - self.corridor_width / 2
        left_curb_length = left_curb_rightend_pos + 10
        left_curb_pos = left_curb_rightend_pos - left_curb_length / 2

        self.sim.model.geom_pos[idxleft][0] = left_curb_pos
        self.sim.model.geom_size[idxleft][0] = left_curb_length / 2
        # print("x_pos is at:", self.x_pos, "right curb pos", right_curb_pos, "left curb pos", left_curb_pos)

    def distance_to_goal(self, goal_y):
        goal_x = 0
        goal_y = goal_y
        agent_x = self.get_body_com("agent_torso")[0]
        agent_y = self.get_body_com("agent_torso")[1]
        distance = math.sqrt((goal_x - agent_x) ** 2 + (goal_y - agent_y) ** 2)
        return distance

    def distance_to_door(self, ith, offset):
        door_x = self.corridor_pos[ith]
        door_y = offset
        agent_x = self.get_body_com("agent_torso")[0]
        agent_y = self.get_body_com("agent_torso")[1]
        distance =  math.sqrt((door_x - agent_x) **2 + (door_y - agent_y) **2)
        return distance


