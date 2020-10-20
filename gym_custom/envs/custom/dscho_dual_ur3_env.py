import copy
import numpy as np
import os
import warnings
import gym_custom
from gym_custom.spaces import Box
from gym_custom import utils
from gym_custom.envs.mujoco import MujocoEnv
from gym_custom.envs.custom.dual_ur3_env import DualUR3Env
from gym_custom.envs.custom.ur_utils import URScriptWrapper, URScriptWrapper_DualUR3
from gym_custom import Wrapper
from gym_custom.envs.custom.ur_utils import SO3Constraint, UprightConstraint, NoConstraint

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

class DualWrapper(URScriptWrapper):
    def __init__(self, q_control_type, multi_step, *args):
        super().__init__(*args)
        
        self.q_control_type = q_control_type
        self.multi_step = multi_step
        self.dt = self.env.dt*self.multi_step

        if self.q_control_type == 'servoj':
            low = np.array([-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-50, -2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-50])
            high= np.array([ 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 50,  2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 50])
        elif self.q_control_type == 'speedj':
            low = np.array([-1,-1,-1,-1,-1,-1,-50,-1,-1,-1,-1,-1,-1,-50])
            high= np.array([ 1, 1, 1, 1, 1, 1, 50, 1, 1, 1, 1, 1, 1, 50])
        self.ur3_act_dim = self.ndof
        self.gripper_act_dim = int(self.ngripperdof/2)
        self.action_space = Box(low=low, high=high, dtype=np.float32)
        print(colorize('WARNING : CHECK action space boundary : {}'.format(self.action_space), 'green', bold=True))

        command = {
            'ur3': {'type': 'speedj', 'command': np.zeros(self.env.ur3_nqpos*2)},
            'gripper': {'type': 'forceg', 'command': np.zeros(2)}
        }
        observation, _reward, done, _info = self.step(command)
        assert not done
        self.observation_space = self.env._set_observation_space(observation)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)
    
    def step(self, action):
        for _ in range(self.multi_step-1):
            self._step(action)
        return self._step(action)

    def _step(self, action):
        observation, reward, done, info = self.env.step(self.action(action))
        return self.observation(observation), self.reward(reward), self.done(done), self.info(info)

    def observation(self, observation):
        original_obs = self.env._get_obs()
        my_extra_obs = self.env._get_my_extra_obs()
        self.wrapped_obs = np.concatenate([original_obs, my_extra_obs])
        return self.wrapped_obs

    def reward(self, reward):
        obs_dict = self.env._get_my_obs_dict()
        self.wrapped_rew = self.env.compute_reward(obs_dict)
        return self.wrapped_rew
    
    def done(self, done):
        if self.env.curr_path_length == self.env.max_path_length: 
            done = True
        elif not self.env.dense_reward and self.wrapped_rew > 0 :# which means it reaches the goal in sparse reward setting
            done =True
        else:
            done = False
        self.wrapped_done = done
        return self.wrapped_done

    def info(self, info):
        self.wrapped_info = info
        return self.wrapped_info


class SingleWrapper(URScriptWrapper):
    '''
    obs : only ur3, gripper qpos,qvel
    not including object's qpos,qvel
    '''

    def __init__(self, q_control_type, multi_step, *args):
        super().__init__(*args)
        self.left_get_away_qpos = np.concatenate([np.array([-90.0, -90.0, 90.0, -90.0, 135.0, 0.0])*np.pi/180.0, np.zeros(self.env.gripper_nqpos)])
        self.right_get_away_qpos = np.concatenate([np.array([-90.0, -90.0, 90.0, -90.0, 135.0, 0.0])*np.pi/180.0, np.zeros(self.env.gripper_nqpos)])
        # self.right_get_away_qpos = np.concatenate([np.array([-1.15843726,-2.05576686,-0.62114284,-2.2506277,-2.01666951,1.94285417]), np.zeros(self.env.gripper_nqpos)])
        self.q_control_type = q_control_type
        self.multi_step = multi_step
        self.dt = self.env.dt*self.multi_step
        # self.action_space = self.env._set_action_space() # mujoco xml 기준 actuator의 ctrlrange 받아서 구하는거 -> 16, But 우리는 14원함
        if self.q_control_type == 'servoj':
            low = np.array([-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-50, -2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-50])
            high= np.array([ 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 50,  2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 50])
        elif self.q_control_type == 'speedj':
            low = np.array([-1,-1,-1,-1,-1,-1,-50,-1,-1,-1,-1,-1,-1,-50])
            high= np.array([ 1, 1, 1, 1, 1, 1, 50, 1, 1, 1, 1, 1, 1, 50])
        self.ur3_act_dim = self.ndof
        self.gripper_act_dim = int(self.ngripperdof/2)
        self.action_space = Box(low=low, high=high, dtype=np.float32)
        print(colorize('WARNING : CHECK action space boundary : {}'.format(self.action_space), 'green', bold=True))

        command = {
            'ur3': {'type': self.q_control_type, 'command': np.zeros(2*self.env.ur3_nqpos)},
            'gripper': {'type': 'forceg', 'command': np.zeros(2)}
        }
        observation, _reward, done, _info = self.step(command)
        assert not done
        self.observation_space = self.env._set_observation_space(observation)

    def reset(self, **kwargs):
        #dscho mod
        # observation = self.env.reset(**kwargs)
        observation = super().reset(**kwargs)
        return self.observation(observation)

    def step(self, action):
        for _ in range(self.multi_step-1):
            self._step(action)
        return self._step(action)

    def _step(self, action):
        # actions of remaning arm will be garbage
        # gripper [-1,1] should be mod to
        observation, reward, done, info = self.env.step(self.action(action))
        
        # process to make remaing arm not to move
        qpos = self.env.data.qpos.flat.copy()
        qvel = self.env.data.qvel.flat.copy()        
        if self.env.which_hand =='right':
            #left arm's qpos,qvel index
            start_p, end_p = self.env.ur3_nqpos+self.env.gripper_nqpos, 2*self.env.ur3_nqpos+2*self.env.gripper_nqpos
            start_v, end_v = self.env.ur3_nqvel+self.env.gripper_nqvel, 2*self.env.ur3_nqvel+2*self.env.gripper_nqvel
            qpos[start_p:end_p] = self.left_get_away_qpos
            qvel[start_v:end_v] = np.zeros(end_v-start_v)
        elif self.env.which_hand=='left':
            #right arm's qpos,qvel index
            start_p, end_p = 0, self.env.ur3_nqpos+self.env.gripper_nqpos
            start_v, end_v = 0, self.env.ur3_nqvel+self.env.gripper_nqvel
            qpos[start_p:end_p] = self.right_get_away_qpos
            qvel[start_v:end_v] = np.zeros(end_v-start_v)
        
        
        self.env.set_state(qpos, qvel)
        # set state하면 site pos도 다 초기화됨! #TODO: 이 부분은 wrapper에 있을 함수가 아님!
        self.env._set_goal_marker(self.env._state_goal)
        # print('env state goal : {}'.format(self.env._state_goal))

        wrapped_obs = self.observation(observation)
        
        obs_dict = self.env._get_my_obs_dict()
        wrapped_rew, wrapped_info = self.env.compute_reward(obs_dict, action)
        wrapped_done = self.env.check_done(wrapped_info)
        
        return wrapped_obs, wrapped_rew, wrapped_done, wrapped_info

    def observation(self, observation): 
        agent_obs = self.env._get_agent_obs(arm = self.env.which_hand)
        # env_obs = self.env._get_env_obs(arm=self.env.which_hand)
        # wrapped_obs = np.concatenate([agent_obs, env_obs])
        wrapped_obs = agent_obs
        return wrapped_obs

class DSCHODualUR3Env(DualUR3Env):

    def __init__(self,
                random_init=False, #goal [0.25, -0.3, 0.8]
                tasks = [{'goal': np.array([0.0, -0.4, 0.8]), 'goal_quat' : np.array([1., 0., 0., 0.]), 'obj_init_pos':np.array([0, -0.4, 0.65]), 'obj_init_angle': 0.0}], 
                obj_low=None,
                obj_high=None,
                goal_low=None,
                goal_high=None,
                dense_reward = False,
                initMode='vertical',
                xml_filename = 'dscho_dual_ur3.xml',
                **kwargs
                ):
        print(colorize('WARNING : Sholud be used with ObsActRewWrapper !', 'green', bold=True))
        self.tasks = tasks
        self.num_tasks = len(tasks)
        task = self.sample_task()
        self._state_goal = task['goal']
        self.random_init=random_init
        self.curr_path_length = 0
        self.max_path_length = 20000 # 0.002 per step -> 40 sec
        self.dense_reward = dense_reward
        self.initMode = initMode
        
        super().__init__(xml_filename, initMode)

        goal_low = np.array([-1,-1,-1])
        goal_high = np.array([1,1,1])
        obj_low = goal_low
        obj_high = goal_high
        self.goal_and_obj_space = Box(
            np.hstack((goal_low, obj_low)),
            np.hstack((goal_high, obj_high)),
        )
        agent_low = np.concatenate([self.kinematics_params['lb'], -np.ones(self.ur3_nqpos)], axis =-1)
        agent_high = np.concatenate([self.kinematics_params['ub'], np.ones(self.ur3_nqpos)], axis =-1)
        
        self.goal_space = Box(goal_low, goal_high)
        self.agent_space = Box(agent_low, agent_high)

        self.obj_init_pos = self.get_obj_pos()
        self.obj_names = ['obj']
        # self._q_des = self.init_qpos
        
        self._q_des = self._get_ur3_qpos()
    
    def step(self, action):
        ob, reward, done, info = super().step(action)
        self._set_goal_marker(self._state_goal)
        self.curr_path_length +=1
        # print('curr_path_length ', self.curr_path_length)
        return ob, reward, done, {}
    

    def reset_model(self):
        original_ob = super().reset_model()
        task = self.sample_task()
        self._state_goal = np.array(task['goal'])
        self._state_goal_quat = np.array(task['goal_quat'])
        self.obj_init_pos = task['obj_init_pos']
        
        if self.random_init :
            goal_pos = np.random.uniform(
                self.goal_and_obj_space.low,
                self.goal_and_obj_space.high,
                size=(self.goal_and_obj_space.low.size),
            )
            while np.linalg.norm(goal_pos[:3] - goal_pos[-3:]) < 0.1:
                goal_pos = np.random.uniform(
                    self.goal_and_obj_space.low,
                    self.goal_and_obj_space.high,
                    size=(self.goal_and_obj_space.low.size),
                )
            self._state_goal = goal_pos[:3]
            #TODO: set state_goal_quat
            self.obj_init_pos = np.concatenate((goal_pos[-3:-1], np.array([self.obj_init_pos[-1]])))
        self._set_goal_marker(self._state_goal)
        self._set_obj_xyz(self.obj_init_pos)
        self.curr_path_length = 0
        
        return original_ob

    def _get_my_extra_obs(self):
        hand = self.get_endeff_pos(arm = 'right')
        second_hand = self.get_endeff_pos(arm='left')        
        grasp_point = self.get_site_pos('objSite')
        second_grasp_point = self.get_site_pos('second_objSite')
        from_ee_to_gp1 = grasp_point - hand
        from_ee_to_gp2 = second_grasp_point - second_hand
        state_goal = self._state_goal
        my_extra_obs = np.concatenate([hand, second_hand, from_ee_to_gp1, from_ee_to_gp2, state_goal], axis = -1)
        
        return my_extra_obs

    def _get_agent_obs(self, arm):
        # ur3, gripper, obj qpos & qvel
        if arm=='right':
            ur3_qpos = self._get_ur3_qpos()[:self.ur3_nqpos]
            gripper_qpos = self._get_gripper_qpos()[:self.gripper_nqpos]
                
            ur3_qvel = self._get_ur3_qvel()[:self.ur3_nqvel]
            gripper_qvel = self._get_gripper_qvel()[:self.gripper_nqvel]
            
        elif arm =='left':
            ur3_qpos = self._get_ur3_qpos()[self.ur3_nqpos:]
            gripper_qpos = self._get_gripper_qpos()[self.gripper_nqpos:]
                
            ur3_qvel = self._get_ur3_qvel()[self.ur3_nqvel:]
            gripper_qvel = self._get_gripper_qvel()[self.gripper_nqvel:]
        else :
            raise NotImplementedError
        
        # return np.concatenate([ur3_qpos, gripper_qpos, ur3_qvel, gripper_qvel])
        return np.concatenate([ur3_qpos, ur3_qvel])


    def _get_my_obs_dict(self):
        obs = self._get_obs() #from parent class
        
        hand = self.get_endeff_pos(arm = 'right')
        second_hand = self.get_endeff_pos(arm='left')
        # dscho mod
        objPos = self.get_obj_pos()
        objQuat = self.get_obj_quat()

        grasp_point = self.get_site_pos('objSite')
        second_grasp_point = self.get_site_pos('second_objSite')
        
        from_ee_to_gp1 = grasp_point - hand
        from_ee_to_gp2 = second_grasp_point - second_hand

        state_goal = self._state_goal
        q_des = self._q_des

        agent_obs = self._get_agent_obs(arm='right')
        second_agent_obs = self._get_agent_obs(arm='left')

        return dict(
            state_observation=obs,
            state_hand=hand,
            state_second_hand = second_hand,
            state_obj_pos = objPos,
            state_obj_quat = objQuat,
            state_grasp_point = grasp_point, 
            state_second_grasp_point = second_grasp_point,
            state_desired_goal=state_goal,
            state_desired_qpos = q_des,
            state_agent_obs = agent_obs,
            state_second_agent_obs = second_agent_obs
        )

    def get_mid_reward_done(self, obs_dict, mid_act, final_goal, final_goal_q_des, arm, dense_reward =False):
        assert isinstance(obs_dict, dict)
        obs = obs_dict['state_observation']
        hand = obs_dict['state_hand']
        second_hand = obs_dict['state_second_hand']
        objPos = obs_dict['state_obj_pos']
        objQuat = obs_dict['state_obj_quat']
        placingGoal = obs_dict['state_desired_goal']
        obj_grasp_point = obs_dict['state_grasp_point']
        second_obj_grasp_point = obs_dict['state_second_grasp_point']
        q_des = obs_dict['state_desired_qpos']
        
        if arm=='right':
            ee_pos = hand
            grasp_point = obj_grasp_point
            ur3_qpos = self._get_ur3_qpos()[:self.ur3_nqpos]
            ur3_qdes = final_goal_q_des
        elif arm=='left':
            ee_pos = second_hand
            grasp_point = second_obj_grasp_point
            ur3_qpos = self._get_ur3_qpos()[self.ur3_nqpos:]
            ur3_qdes = final_goal_q_des

        # Currently implemented with sparse reward
        q_err = np.linalg.norm(ur3_qdes - ur3_qpos)
        from_ee_to_subgoal = copy.deepcopy(mid_act)
        goalDist = np.linalg.norm(final_goal - ee_pos) 
        c1, c2, c3 = [1, 1, 0.1]
        # final goal에 도달하면 done True, & sparse reward 혹은 final goal에 가까울수록 받게끔해야(obs avoid 등 원하면 아무래도 sparse 해야할듯?)
        if dense_reward :
            reward = -c1*goalDist-c2*q_err -c3*np.linalg.norm(mid_act)
            done = True if goalDist < 0.03 else False
        else :
            if goalDist < 0.02 and q_err < 0.05:
                reward = 1
                done = True
            else :
                reward = 0
                done = False

        return reward, done, {}


    def compute_reward(self, obs_dict):
        
        assert isinstance(obs_dict, dict)
        obs = obs_dict['state_observation']
        hand = obs_dict['state_hand']
        second_hand = obs_dict['state_second_hand']
        objPos = obs_dict['state_obj_pos']
        objQuat = obs_dict['state_obj_quat']
        placingGoal = obs_dict['state_desired_goal']
        obj_grasp_point = obs_dict['state_grasp_point']
        second_obj_grasp_point = obs_dict['state_second_grasp_point']
        q_des = obs_dict['state_desired_qpos']
        placingDist = np.linalg.norm(objPos - placingGoal)
       
        
        if placingDist < 0.03:
            reward = 1
        else :
            reward = 0 

        return reward

    def compute_dense_reward(self,action, obs_dict, mode):
        raise NotImplementedError


    def get_obj_pos(self, name=None):
        if name is None:
            return self.data.get_body_xpos('obj')
        else :
            return self.data.get_body_xpos(name) 

    def get_obj_quat(self, name=None):
        if name is None:
            return self.data.get_body_xquat('obj')
        else :
            return self.data.get_body_xquat(name)
    
    def get_obj_qpos(self, name=None):
        if name is None:
            body_idx = self.model.body_names.index('obj')
        else :
            body_idx = self.model.body_names.index(name)
        jnt_idx = self.model.body_jntadr[body_idx]
        qpos_start_idx = self.model.jnt_qposadr[jnt_idx]
        
        qpos = self.data.qpos.flat.copy()
        return qpos[qpos_start_idx:qpos_start_idx+7]

    def get_obj_qvel(self, name=None):
        if name is None:
            body_idx = self.model.body_names.index('obj')
        else :
            body_idx = self.model.body_names.index(name)
        jnt_idx = self.model.body_jntadr[body_idx]
        qvel_start_idx = self.model.jnt_dofadr[jnt_idx]
    
        qvel = self.data.qvel.flat.copy()
        return qvel[qvel_start_idx:qvel_start_idx+6]

    def get_endeff_pos(self, arm):
        if arm == 'right':
            q= self._get_ur3_qpos()[:self.ur3_nqpos]
        elif arm =='left' :
            q= self._get_ur3_qpos()[self.ur3_nqpos:]
        R, p, T = self.forward_kinematics_ee(q, arm)
        return p
    
    def get_current_goal(self):
        return self._state_goal
        

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        # print('set_goal_marker : {}'.format(goal))
        self.data.site_xpos[self.model.site_name2id('goal')] = (
            goal[:3]
        )

    def _set_objCOM_marker(self):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        objPos =  self.data.get_geom_xpos('objGeom')
        self.data.site_xpos[self.model.site_name2id('objSite')] = (
            objPos
        )
    
    def _set_obj_xyz(self, pos):
        '''
        bodyid = mj_name2id(model, mjOBJ_BODY, name) is the index of the body with a given name

        jntid = model->body_jntadr[bodyid] is the index of the first joint for that body

        model->jnt_qposadr[jntid] is the first position of that joint in the system qpos vector

        for indexing into qvel and qfrc, you need the dof_XXX arrays in mjModel. the first velocity index in this example is given by body_dofadr[bodyid], which equals jnt_dofadr[jntid]
        '''
        body_idx = self.model.body_names.index('obj')
        jnt_idx = self.model.body_jntadr[body_idx]
        qpos_start_idx = self.model.jnt_qposadr[jnt_idx]
        qvel_start_idx = self.model.jnt_dofadr[jnt_idx]
        
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[qpos_start_idx:qpos_start_idx+3] = pos.copy() #자세는 previous에서 불변
        qvel[qvel_start_idx:qvel_start_idx+6] = 0 #object vel
        self.set_state(qpos, qvel)


    def sample_task(self):
        task_idx = np.random.randint(0, self.num_tasks)
        return self.tasks[task_idx]
    

    
class DSCHOSingleReachUR3Env(DSCHODualUR3Env):
    '''
    Sholud be used with Obs,Act,Rew Wrapper
    '''
    def __init__(self, which_hand='right', *args,**kwargs):
        super().__init__(*args, **kwargs)
        self.which_hand = which_hand
        self.constraint_fun = SO3Constraint(SO3='vertical_side')
        # end eff xyz (roll, pitch, yaw) , if you want arbitrary orientation of ee, you have to convert roll pitch yaw into SO3
        print(colorize('WARNING : Currently working on vertical side constraint, only with ee xyz goal', 'green', bold=True))
        # vertical(horizontal) fix이면 더 좁혀야, 임의의 rotation이면 좀 넓게 잡아야?
        if self.which_hand =='right':
            goal_low = np.array([-0.1, -0.6, 0.65])
            goal_high = np.array([0.6, -0.2, 0.9])
        elif self.which_hand =='left':
            goal_low = np.array([-0.6, -0.6, 0.65])
            goal_high = np.array([0.1, -0.2, 0.9])

        self.goal_space = Box(low=goal_low, high=goal_high, dtype=np.float32)
        if self.which_hand=='right':
            self._q_des = self._get_ur3_qpos()[:self.ur3_nqpos]
        elif self.which_hand=='left':
            self._q_des = self._get_ur3_qpos()[self.ur3_nqpos:]

    def reset_model(self):
        original_ob = super().reset_model()
        assert not self.random_init, 'it is only for reaching env'
        
        while True:
            goal_pos = np.random.uniform(
                self.goal_space.low,
                self.goal_space.high,
                size=(self.goal_space.low.size),
            )
            SO3= 'vertical_side'
            null_obj_func = SO3Constraint(SO3)
            q_des, iter_taken, err, null_obj = self.inverse_kinematics_ee(goal_pos, null_obj_func, arm=self.which_hand, max_iter = 100)
            if iter_taken != 100 :
                break
            
        self._state_goal = goal_pos[:3]
        self._q_des = q_des
        self._set_goal_marker(self._state_goal)
        
        return original_ob

    def _get_agent_obs(self, arm):
        # ur3, gripper, obj qpos & qvel
        if arm=='right':
            ur3_qpos = self._get_ur3_qpos()[:self.ur3_nqpos]
            gripper_qpos = self._get_gripper_qpos()[:self.gripper_nqpos]
                
            ur3_qvel = self._get_ur3_qvel()[:self.ur3_nqvel]
            gripper_qvel = self._get_gripper_qvel()[:self.gripper_nqvel]
            
        elif arm =='left':
            ur3_qpos = self._get_ur3_qpos()[self.ur3_nqpos:]
            gripper_qpos = self._get_gripper_qpos()[self.gripper_nqpos:]
                
            ur3_qvel = self._get_ur3_qvel()[self.ur3_nqvel:]
            gripper_qvel = self._get_gripper_qvel()[self.gripper_nqvel:]
        else :
            raise NotImplementedError
        
        # return np.concatenate([ur3_qpos, gripper_qpos, ur3_qvel, gripper_qvel])
        return np.concatenate([ur3_qpos, ur3_qvel])

    
    def _get_env_obs(self, arm):
        if arm == 'right':
            hand = self.get_endeff_pos(arm='right')
        elif arm == 'left':
            hand = self.get_endeff_pos(arm='left')
        # objPos = self.get_obj_pos()
        # objQuat = self.get_obj_quat()
        goal = self.get_current_goal()
        
        from_ee_to_goal = goal - hand
        
        return from_ee_to_goal
    
    def compute_reward(self, obs_dict, action):
        
        assert isinstance(obs_dict, dict)
        obs = obs_dict['state_observation']
        hand = obs_dict['state_hand']
        second_hand = obs_dict['state_second_hand']
        objPos = obs_dict['state_obj_pos']
        objQuat = obs_dict['state_obj_quat']
        placingGoal = obs_dict['state_desired_goal']
        obj_grasp_point = obs_dict['state_grasp_point']
        second_obj_grasp_point = obs_dict['state_second_grasp_point']
        
        
        if self.which_hand =='right':
            handPos = hand
            ur3_qpos = self._get_ur3_qpos()[:self.ur3_nqpos]
        elif self.which_hand =='left':
            handPos = second_hand
            ur3_qpos = self._get_ur3_qpos()[self.ur3_nqpos:]
        self.placingDist = placingDist = np.linalg.norm(handPos - placingGoal)
        
        
        SO3, _, _ = self.forward_kinematics_ee(ur3_qpos, self.which_hand)
        self.so3_err = so3_err = self.constraint_fun.evaluate(SO3)
        c1,c2,c3,c4 = [1,0.1,1,1]
        q_des = self._q_des
        q_pos_err = np.linalg.norm(ur3_qpos - q_des)
        
        if action['ur3']['type'] == 'speedj':
            reward = -c1*placingDist -c2*so3_err -c3*np.linalg.norm(q_pos_err) -c4*np.linalg.norm(action['ur3']['command'])
        if action['ur3']['type'] == 'servoj': # could not penalize to minimize because it is position control. If you want, you have to deal with low level torque
            reward = -c1*placingDist -c2*so3_err -c3*np.linalg.norm(q_pos_err)
                
        return reward, {'placingDist' : self.placingDist, 'so3_err' : self.so3_err, 'q_des' : self._q_des}
    
    def check_done(self, wrapped_info = None):
        placingDist = wrapped_info['placingDist']
        so3_err = wrapped_info['so3_err']
        if placingDist < 0.02 and so3_err < 0.05: #so3 error 는 보통 0.01이면 거의 완벽히 맞는 수준
            wrapped_done = True
        else :
            wrapped_done = False
        return wrapped_done


class DSCHODualUR3EnvObstacle(DSCHODualUR3Env):
    def __init__(self, *args, **kwargs):
        xml_filename = 'dscho_dual_ur3_obstacle.xml'
        tasks = [{'goal': np.array([0.0, -0.45, 0.8]), 'goal_quat' : np.array([1., 0., 0., 0.]), 'obj_init_pos':np.array([0, -0.45, 0.65]), 'obj_init_angle': 0.0}]
        super().__init__(tasks = tasks, xml_filename = xml_filename, *args, **kwargs)
    
class DSCHOSingleReachUR3EnvObstacle(DSCHOSingleReachUR3Env):
    def __init__(self, which_hand, *args, **kwargs):
        xml_filename = 'dscho_dual_ur3_obstacle.xml'
        tasks = [{'goal': np.array([0.0, -0.45, 0.8]), 'goal_quat' : np.array([1., 0., 0., 0.]), 'obj_init_pos':np.array([1.0, -0.45, 0.65]), 'obj_init_angle': 0.0}]
        super().__init__(which_hand= which_hand, tasks = tasks, xml_filename = xml_filename, *args, **kwargs)
        