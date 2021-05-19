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
# from gym_custom.envs.custom.constraint.pose_constraint import SO3Constraint, UprightConstraint, NoConstraint
import tensorflow as tf
import joblib
import time
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


class DualWrapper(URScriptWrapper_DualUR3):
    def __init__(self, env, q_control_type, g_control_type, multi_step, gripper_action, serializable_initialized = False, gripper_force_scale = 50,*args, **kwargs):
        # self._wrapped_env needs to be called first because
        # Serializable.quick_init calls getattr, on this class. And the
        # implementation of getattr (see below) calls self._wrapped_env.
        # Without setting this first, the call to self._wrapped_env would call
        # getattr again (since it's not set yet) and therefore loop forever.
        self.env = env
        # Or else serialization gets delegated to the wrapped_env. Serialize
        # this env separately from the wrapped_env.
        self._serializable_initialized = serializable_initialized
        self.save_init_params(locals())
        
        super().__init__(env, *args, **kwargs)
        self.q_control_type = q_control_type
        self.g_control_type = g_control_type
        self.gripper_force_scale = gripper_force_scale
        self.multi_step = multi_step
        self.gripper_action = gripper_action
        self.dt = self.env.dt*multi_step

        self.speedj_args = {'a': 5, 't': None, 'wait': None}
        self.servoj_args = {'t': None, 'wait': None}


        if q_control_type == 'servoj':
            if gripper_action:
                self.act_low = act_low = np.array([-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi, -50, -2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi, -50])
                self.act_high = act_high= np.array([2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi,  50,  2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi,  50])
            else :
                self.act_low = act_low = np.array([-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi])
                self.act_high = act_high= np.array([2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
        elif q_control_type == 'speedj':
            if gripper_action:
                self.act_low = act_low = np.array([-1,-1,-1,-1,-1,-1,-50, -1,-1,-1,-1,-1,-1,-50])
                self.act_high = act_high= np.array([1, 1, 1, 1, 1, 1, 50,  1, 1, 1, 1, 1, 1, 50]) 
            else :
                self.act_low = act_low = np.array([-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1])
                self.act_high = act_high= np.array([1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1])
        
        self.ur3_act_dim = self.wrapper_right.ndof
        self.gripper_act_dim = self.wrapper_right.ngripperdof
        assert self.ur3_act_dim==6
        assert self.gripper_act_dim==1
        self.action_space = Box(low=act_low, high=act_high, dtype=np.float32)
        print(colorize('WARNING : CHECK action space boundary : {}'.format(self.action_space), 'green', bold=True))

    def reset(self, **kwargs):
        # return self.env.reset(**kwargs)
        return super().reset(**kwargs)
        
    def step(self, action):
        action = action.copy()
        for _ in range(self.multi_step-1):
            self._step(action)
        return self._step(action)

    def _step(self, action): 
        
        # Assume 
        # if gripper_action is True:
        # action is np.array(right_ur3(6), right_gripper(1), left ur3(6), left_gripper(1)) 
        # elif gripper_action is False:
        # action is np.array(right_ur3(6), left ur3(6)) 
        
        if self.gripper_action:
            right_ur3_action = action[:self.ur3_act_dim]
            right_gripper_action = self.gripper_force_scale*action[self.ur3_act_dim:self.ur3_act_dim+self.gripper_act_dim]
            left_ur3_action = action[self.ur3_act_dim+self.gripper_act_dim:2*self.ur3_act_dim+self.gripper_act_dim]
            left_gripper_action = self.gripper_force_scale*action[2*self.ur3_act_dim+self.gripper_act_dim:2*self.ur3_act_dim+2*self.gripper_act_dim]
        else :
            right_ur3_action = action[:self.ur3_act_dim]
            right_gripper_action = np.zeros(self.gripper_act_dim)
            left_ur3_action = action[self.ur3_act_dim:]
            left_gripper_action = np.zeros(self.gripper_act_dim)
        
        
        right_q_control_args, right_g_control_args = self._get_control_kwargs(self.q_control_type, self.g_control_type, right_ur3_action, right_gripper_action)
        left_q_control_args, left_g_control_args = self._get_control_kwargs(self.q_control_type, self.g_control_type, left_ur3_action, left_gripper_action) 
        
        command = {
                    'right': {
                        self.q_control_type : right_q_control_args,
                        self.g_control_type : right_g_control_args
                    },
                    'left': {
                        self.q_control_type : left_q_control_args,
                        self.g_control_type : left_g_control_args
                    }
                  }

        # actions of remaning arm will be garbage
        # gripper [-1,1] should be mod to
        observation, reward, done, info = self.env.step(self.action(command)) # obs is dict

        wrapped_obs = observation
        wrapped_rew = reward
        wrapped_done = done
        wrapped_info = info
        
        return wrapped_obs, wrapped_rew, wrapped_done, wrapped_info

    #TODO : if it is slow, you'd better use predefined dictionary format by using for loop to avoid if, elif execution per step
    def _get_control_kwargs(self, q_control_type, g_control_type, ur3_action, gripper_action):

        if q_control_type=='speedj':
            q_control_args = copy.deepcopy(self.speedj_args)
            q_control_args.update({'qd' : ur3_action})
        elif q_control_type=='servoj':
            q_control_args = copy.deepcopy(self.servoj_args)
            q_control_args.update({'q' : ur3_action})
        if g_control_type=='move_gripper_force':
            g_control_args = {'gf' : gripper_action}
        elif g_control_type=='move_gripper_position':
            g_control_args = {'g' : gripper_action}
        elif g_control_type=='move_gripper_velocity':
            g_control_args = {'gd' : gripper_action}
        
        return q_control_args, g_control_args
    
    def __getattr__(self, name):
        return getattr(self.env, name)
        
class DummyWrapper():
    
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

class SingleWrapper(URScriptWrapper_DualUR3):
    '''
    obs : only ur3 qpos, qvel, ee_pos
    not including object's qpos,qvel, gripper qpos, qvel
    '''

    def __init__(self, env, q_control_type, g_control_type, multi_step, gripper_action, serializable_initialized = False, gripper_force_scale=50, *args, **kwargs):
        # self._wrapped_env needs to be called first because
        # Serializable.quick_init calls getattr, on this class. And the
        # implementation of getattr (see below) calls self._wrapped_env.
        # Without setting this first, the call to self._wrapped_env would call
        # getattr again (since it's not set yet) and therefore loop forever.
        self.env = env
        # Or else serialization gets delegated to the wrapped_env. Serialize
        # this env separately from the wrapped_env.
        self._serializable_initialized = serializable_initialized
        self.save_init_params(locals())

        super().__init__(env, *args, **kwargs)
        self.q_control_type = q_control_type
        self.g_control_type = g_control_type
        self.gripper_force_scale = gripper_force_scale
        self.multi_step = multi_step
        self.gripper_action = gripper_action
        self.dt = self.env.dt*multi_step

        self.speedj_args = {'a': 5, 't': None, 'wait': None}
        self.servoj_args = {'t': None, 'wait': None}


        if q_control_type == 'servoj':
            if gripper_action:
                self.act_low = act_low = np.array([-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi, -50])
                self.act_high = act_high= np.array([ 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 50])
            else :
                self.act_low = act_low = np.array([-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi])
                self.act_high = act_high= np.array([ 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
        elif q_control_type == 'speedj':
            if gripper_action:
                self.act_low = act_low = np.array([-1,-1,-1,-1,-1,-1,-50])
                self.act_high = act_high= np.array([ 1, 1, 1, 1, 1, 1, 50]) 
            else :
                self.act_low = act_low = np.array([-1,-1,-1,-1,-1,-1])
                self.act_high = act_high= np.array([ 1, 1, 1, 1, 1, 1])
        
        self.ur3_act_dim = self.wrapper_right.ndof
        self.gripper_act_dim = self.wrapper_right.ngripperdof
        assert self.ur3_act_dim==6
        assert self.gripper_act_dim==1
        self.action_space = Box(low=act_low, high=act_high, dtype=np.float32)
        print(colorize('WARNING : CHECK action space boundary : {}'.format(self.action_space), 'green', bold=True))
        
    def reset(self, **kwargs):
        # return self.env.reset(**kwargs)
        return super().reset(**kwargs)

    # Wrapper 에서 multistep을 해야 제일 lowlevel에선 매 timestep마다 command 새로계산(정확도 증가)
    def step(self, action):
        action = action.copy()
        for _ in range(self.multi_step-1):
            self._step(action)
        return self._step(action)        

    def _step(self, action): 
        # assume action is np.array(ur3(6)) (if gripper_action is True, then , gripper(1) dimension added )
        ur3_act = action[:self.ur3_act_dim]
        if self.gripper_action:
            gripper_act = self.gripper_force_scale*action[-self.gripper_act_dim:]
        else :
            gripper_act = np.zeros(self.gripper_act_dim)
        
        if self.env.which_hand=='right':
            right_ur3_action = ur3_act
            right_gripper_action = gripper_act
            left_ur3_action = np.zeros(self.ur3_act_dim)
            left_gripper_action = np.zeros(self.gripper_act_dim)
            
        elif self.env.which_hand=='left':
            right_ur3_action = np.zeros(self.ur3_act_dim)
            right_gripper_action = np.zeros(self.gripper_act_dim)
            left_ur3_action = ur3_act
            left_gripper_action = gripper_act
        
        right_q_control_args, right_g_control_args = self._get_control_kwargs(self.q_control_type, self.g_control_type, right_ur3_action, right_gripper_action)
        left_q_control_args, left_g_control_args = self._get_control_kwargs(self.q_control_type, self.g_control_type, left_ur3_action, left_gripper_action) 
        
        command = {
                    'right': {
                        self.q_control_type : right_q_control_args,
                        self.g_control_type : right_g_control_args
                    },
                    'left': {
                        self.q_control_type : left_q_control_args,
                        self.g_control_type : left_g_control_args
                    }
                  }

        # actions of remaning arm will be garbage
        # gripper [-1,1] should be mod to
        observation, reward, done, info = self.env.step(self.action(command)) # obs is dict

        wrapped_obs = observation
        wrapped_rew = reward
        wrapped_done = done
        wrapped_info = info
        
        return wrapped_obs, wrapped_rew, wrapped_done, wrapped_info

    #TODO : if it is slow, you'd better use predefined dictionary format by using for loop to avoid if, elif execution per step
    def _get_control_kwargs(self, q_control_type, g_control_type, ur3_action, gripper_action):

        if q_control_type=='speedj':
            q_control_args = copy.deepcopy(self.speedj_args)
            q_control_args.update({'qd' : ur3_action})
        elif q_control_type=='servoj':
            q_control_args = copy.deepcopy(self.servoj_args)
            q_control_args.update({'q' : ur3_action})
        if g_control_type=='move_gripper_force':
            g_control_args = {'gf' : gripper_action}
        elif g_control_type=='move_gripper_position':
            g_control_args = {'g' : gripper_action}
        elif g_control_type=='move_gripper_velocity':
            g_control_args = {'gd' : gripper_action}
        
        return q_control_args, g_control_args
    
    def __getattr__(self, name):
        return getattr(self.env, name)

class EndEffectorPositionControlSingleWrapper(URScriptWrapper_DualUR3):
    
    def __init__(self, 
                env, 
                q_control_type, 
                g_control_type, 
                multi_step, 
                gripper_action, 
                serializable_initialized = False, 
                so3_constraint='vertical_side', 
                action_downscale=0.01,
                gripper_force_scale = 50, 
                *args, 
                **kwargs
                ):
        # self._wrapped_env needs to be called first because
        # Serializable.quick_init calls getattr, on this class. And the
        # implementation of getattr (see below) calls self._wrapped_env.
        # Without setting this first, the call to self._wrapped_env would call
        # getattr again (since it's not set yet) and therefore loop forever.
        self.env = env
        # Or else serialization gets delegated to the wrapped_env. Serialize
        # this env separately from the wrapped_env.
        self._serializable_initialized = serializable_initialized
        self.save_init_params(locals())

        super().__init__(env, *args, **kwargs)
        self.q_control_type = q_control_type
        self.g_control_type = g_control_type
        
        self.ee_xyz_pos_dim = 3
        self.action_downscale = action_downscale
        self.gripper_force_scale = gripper_force_scale
        self.null_obj_func = SO3Constraint(SO3=so3_constraint)

        self.multi_step = multi_step
        self.gripper_action = gripper_action
        self.dt = self.env.dt*multi_step

        self.speedj_args = {'a': 5, 't': None, 'wait': None}
        self.servoj_args = {'t': None, 'wait': None}


       
        if gripper_action:
            self.act_low = act_low = np.array([-1, -1, -1, -50])
            self.act_high = act_high= np.array([1, 1, 1, 50])
        else :
            self.act_low = act_low = np.array([-1, -1, -1])
            self.act_high = act_high= np.array([1, 1, 1])
    
        
        self.ur3_act_dim = self.wrapper_right.ndof
        self.gripper_act_dim = self.wrapper_right.ngripperdof
        assert self.ur3_act_dim==6
        assert self.gripper_act_dim==1
        self.action_space = Box(low=act_low, high=act_high, dtype=np.float32)
        print(colorize('WARNING : CHECK action space boundary : {}'.format(self.action_space), 'green', bold=True))
    

    def _gripper_action_clip(self, gripper_act):
        if self.env.which_hand=='right':
            right_gripper_right_state = self.env.data.get_site_xpos('right_gripper:rightEndEffector')
            right_gripper_left_state = self.env.data.get_site_xpos('right_gripper:leftEndEffector')
            distance = np.linalg.norm(right_gripper_right_state - right_gripper_left_state, axis =-1)
            if gripper_act > 0 : # try to close
                if distance >0.08 and distance <=0.1:
                    gripper_act = np.clip(gripper_act, -150, 150)
                elif distance <= 0.08: # 0.08정도가 거의 grasping한 수준?
                    gripper_act = np.clip(gripper_act, -20, 20)
            else: # try to open
                if distance >= 0.1:
                    gripper_act = np.clip(gripper_act, -30, 30)
            
        elif self.env.which_hand=='left':
            left_gripper_right_state = self.env.data.get_site_xpos('left_gripper:rightEndEffector')
            left_gripper_left_state = self.env.data.get_site_xpos('left_gripper:leftEndEffector')
            distance = np.linalg.norm(left_gripper_right_state - left_gripper_left_state, axis =-1)
            if gripper_act > 0: # try to close
                if distance >0.08 and distance <=0.1:
                    gripper_act = np.clip(gripper_act, -150, 150)
                elif distance <= 0.08: # 0.08정도가 거의 grasping한 수준?
                    gripper_act = np.clip(gripper_act, -20, 20)
            else:
                if distance >= 0.1:
                    gripper_act = np.clip(gripper_act, -30, 30)
            
        return gripper_act, distance

    def reset(self, **kwargs):
        # return self.env.reset(**kwargs)
        obs = super().reset(**kwargs)
        if self.env.task in ['pickandplace'] and self.env.warm_start:
            # close gripper near the object
            # ur3_act = np.zeros(self.ee_xyz_pos_dim)            
            for i in range(20): # enough time to succeed for grasping
                # gripper_act = np.array([self.gripper_force_scale]) # closing action
                # gripper_act, distance = self._gripper_action_clip(gripper_act)
                # print('distance in wrapper reset warm start : {} gripper act : {}'.format(distance, gripper_act))
                # action = np.concatenate([ur3_act, gripper_act], axis =-1)
                action = np.array([0,0,0,1]) # closing action
                obs, _, _, _ = self.step(action)
                if self.env.which_hand=='right':
                    right_gripper_right_state = self.env.data.get_site_xpos('right_gripper:rightEndEffector')
                    right_gripper_left_state = self.env.data.get_site_xpos('right_gripper:leftEndEffector')
                    distance = np.linalg.norm(right_gripper_right_state - right_gripper_left_state, axis =-1)
                elif self.env.which_hand=='left':
                    left_gripper_right_state = self.env.data.get_site_xpos('left_gripper:rightEndEffector')
                    left_gripper_left_state = self.env.data.get_site_xpos('left_gripper:leftEndEffector')
                    distance = np.linalg.norm(left_gripper_right_state - left_gripper_left_state, axis =-1)
                print('distance in wrapper reset warm start : {} '.format(distance))
                if distance <= 0.07: #충분히 grasp했다 판단
                    break
        return obs
            
    # Wrapper 에서 multistep을 해야 제일 lowlevel에선 매 timestep마다 command 새로계산함으로써 정확도 증가되는데, 내부적으로 IK sol 쓰다보니 이런구조는 아니라 정확도 살짝 떨어질수도
    def step(self, action):
        action = action.copy()
        # down scale [-1,1] to [-0.005, 0.005]
        
        # action = copy.deepcopy(action) # 통째로 *downscale이면 문제 없는데 index로 접근할땐 array가 mutable이라 copy해줘야함, but 매 스텝마다 action을 새로 뽑는상황이라면 굳이 이렇게 안해도 상관없음. 똑같은 action으로 계속 step밟을 때나 문제가 되는거지
        action[:self.ee_xyz_pos_dim] = np.clip(action[:self.ee_xyz_pos_dim], -1, 1)
        action[:self.ee_xyz_pos_dim] = self.action_downscale*action[:self.ee_xyz_pos_dim]
        ur3_act = action[:self.ee_xyz_pos_dim] # delta xyz pos of ee
        
        if self.gripper_action:
            gripper_act = self.gripper_force_scale*action[-self.gripper_act_dim:]
            
        
            if self.env.block_gripper: # close gripper
                gripper_act = 50*np.ones_like(gripper_act)
            else:
                # To control closing speed w.r.t. how much the gripper is closed
                gripper_act, distance = self._gripper_action_clip(gripper_act)                    
                print('distance in wrapper step : {} grip : {}'.format(distance, gripper_act))

            # print('gripper act : ', gripper_act)
        else :
            gripper_act = np.zeros(self.gripper_act_dim)
        
        
        if self.env.which_hand=='right':
            ee_pos = self.get_endeff_pos('right')
            current_qpos = self._get_ur3_qpos()[:self.ur3_nqpos]
            desired_ee_pos = ee_pos + ur3_act
            start = time.time()
            # ee_pos, null_obj_func, arm, q_init='current', threshold=0.01, threshold_null=0.001, max_iter=100, epsilon=1e-6
            q_des, iter_taken, err, null_obj = self.inverse_kinematics_ee(desired_ee_pos, self.null_obj_func, arm='right', threshold=0.001, max_iter = 10)
            # print('iter taken : {}, time : {}'.format(iter_taken, time.time()-start))
            if self.q_control_type =='speedj':
                right_ur3_action = (q_des-current_qpos)/(self.dt)
            elif self.q_control_type =='servoj':
                right_ur3_action = q_des
            right_gripper_action = gripper_act
            
            left_ur3_action = np.zeros(self.ur3_act_dim)
            left_gripper_action = np.zeros(self.gripper_act_dim)
            
        elif self.env.which_hand=='left':
            ee_pos = self.get_endeff_pos('left')
            current_qpos = self._get_ur3_qpos()[self.ur3_nqpos:]
            desired_ee_pos = ee_pos + ur3_act
            start = time.time()
            q_des, iter_taken, err, null_obj = self.inverse_kinematics_ee(desired_ee_pos, self.null_obj_func, arm='left', threshold=0.001, max_iter = 10)
            # print('iter taken : {}, time : {}'.format(iter_taken, time.time()-start))
            if self.q_control_type =='speedj':
                left_ur3_action = (q_des-current_qpos)/(self.dt)
            elif self.q_control_type =='servoj':
                left_ur3_action = q_des
            left_gripper_action = gripper_act

            right_ur3_action = np.zeros(self.ur3_act_dim)
            right_gripper_action = np.zeros(self.gripper_act_dim)
            
            
        
        right_q_control_args, right_g_control_args = self._get_control_kwargs(self.q_control_type, self.g_control_type, right_ur3_action, right_gripper_action)
        left_q_control_args, left_g_control_args = self._get_control_kwargs(self.q_control_type, self.g_control_type, left_ur3_action, left_gripper_action) 
        
        command = {
                    'right': {
                        self.q_control_type : right_q_control_args,
                        self.g_control_type : right_g_control_args
                    },
                    'left': {
                        self.q_control_type : left_q_control_args,
                        self.g_control_type : left_g_control_args
                    }
                  }

        

        for _ in range(self.multi_step-1):
            # print('command : ', command['right'][self.g_control_type])
            self.env.step(self.action(command)) # obs is dict)
        return self.env.step(self.action(command))


    #TODO : if it is slow, you'd better use predefined dictionary format by using for loop to avoid if, elif execution per step
    def _get_control_kwargs(self, q_control_type, g_control_type, ur3_action, gripper_action):

        if q_control_type=='speedj':
            q_control_args = copy.deepcopy(self.speedj_args)
            q_control_args.update({'qd' : ur3_action})
        elif q_control_type=='servoj':
            q_control_args = copy.deepcopy(self.servoj_args)
            q_control_args.update({'q' : ur3_action})
        if g_control_type=='move_gripper_force':
            g_control_args = {'gf' : gripper_action}
        elif g_control_type=='move_gripper_position':
            g_control_args = {'g' : gripper_action}
        elif g_control_type=='move_gripper_velocity':
            g_control_args = {'gd' : gripper_action}
        
        return q_control_args, g_control_args
    
    def __getattr__(self, name):
        return getattr(self.env, name)


class EndEffectorPositionControlDualWrapper(URScriptWrapper_DualUR3):
    
    def __init__(self,
                env, 
                q_control_type, 
                g_control_type, 
                multi_step, 
                gripper_action, 
                serializable_initialized = False, 
                so3_constraint='vertical_side', 
                action_downscale=0.01,
                gripper_force_scale = 50, 
                *args, 
                **kwargs
                ):
        # self._wrapped_env needs to be called first because
        # Serializable.quick_init calls getattr, on this class. And the
        # implementation of getattr (see below) calls self._wrapped_env.
        # Without setting this first, the call to self._wrapped_env would call
        # getattr again (since it's not set yet) and therefore loop forever.
        self.env = env
        # Or else serialization gets delegated to the wrapped_env. Serialize
        # this env separately from the wrapped_env.
        self._serializable_initialized = serializable_initialized
        self.save_init_params(locals())

        super().__init__(env, *args, **kwargs)
        self.q_control_type = q_control_type
        self.g_control_type = g_control_type
        
        self.ee_xyz_pos_dim = 3
        self.action_downscale = action_downscale
        self.gripper_force_scale = gripper_force_scale
        self.null_obj_func = SO3Constraint(SO3=so3_constraint)

        self.multi_step = multi_step
        self.gripper_action = gripper_action
        self.dt = self.env.dt*multi_step

        self.speedj_args = {'a': 5, 't': None, 'wait': None}
        self.servoj_args = {'t': None, 'wait': None}
       
        if gripper_action:
            self.act_low = act_low = np.array([-1, -1, -1, -50, -1, -1, -1, -50])
            self.act_high = act_high= np.array([1, 1, 1, 50, 1, 1, 1, 50])
        else :
            self.act_low = act_low = np.array([-1, -1, -1, -1, -1, -1])
            self.act_high = act_high= np.array([1, 1, 1, 1, 1, 1])
    
        
        self.ur3_act_dim = self.wrapper_right.ndof
        self.gripper_act_dim = self.wrapper_right.ngripperdof
        assert self.ur3_act_dim==6
        assert self.gripper_act_dim==1
        self.action_space = Box(low=act_low, high=act_high, dtype=np.float32)
        print(colorize('WARNING : CHECK action space boundary : {}'.format(self.action_space), 'green', bold=True))
        
    def reset(self, **kwargs):
        # return self.env.reset(**kwargs)
        return super().reset(**kwargs)

    # Wrapper 에서 multistep을 해야 제일 lowlevel에선 매 timestep마다 command 새로계산함으로써 정확도 증가되는데, 내부적으로 IK sol 쓰다보니 이런구조는 아니라 정확도 살짝 떨어질수도
    def step(self, action):
        action = action.copy()
        # Assume 
        # if gripper_action is True:
        # action is np.array(right delta ee pos(3), right_gripper(1), left delta ee pos(3), left_gripper(1)) 
        # elif gripper_action is False:
        # action is np.array(right delta ee pos(3), left delta ee pos(3)) 
        
        if self.gripper_action:
            # down scale [-1,1] to [-0.005, 0.005]
            action[:self.ee_xyz_pos_dim] = np.clip(action[:self.ee_xyz_pos_dim], -1, 1)
            action[:self.ee_xyz_pos_dim] = self.action_downscale*action[:self.ee_xyz_pos_dim]
            action[self.ee_xyz_pos_dim+self.gripper_act_dim:2*self.ee_xyz_pos_dim+self.gripper_act_dim] = self.action_downscale*action[self.ee_xyz_pos_dim+self.gripper_act_dim:2*self.ee_xyz_pos_dim+self.gripper_act_dim]
            right_ur3_act = action[:self.ee_xyz_pos_dim] # delta xyz pos of ee
            right_gripper_act = self.gripper_force_scale*action[self.ee_xyz_pos_dim:self.ee_xyz_pos_dim+self.gripper_act_dim]
            left_ur3_act = action[self.ee_xyz_pos_dim+self.gripper_act_dim:2*self.ee_xyz_pos_dim+self.gripper_act_dim]
            left_gripper_act = self.gripper_force_scale*action[2*self.ee_xyz_pos_dim+self.gripper_act_dim:2*self.ee_xyz_pos_dim+2*self.gripper_act_dim]
        else :
            # down scale [-1,1] to [-0.005, 0.005]
            action= np.clip(action, -1, 1)
            action = self.action_downscale*action
            right_ur3_act = action[:self.ee_xyz_pos_dim] # delta xyz pos of ee
            right_gripper_act = np.zeros(self.gripper_act_dim)
            left_ur3_act = action[self.ee_xyz_pos_dim:]
            left_gripper_act = np.zeros(self.gripper_act_dim)
        
        right_ee_pos = self.get_endeff_pos('right')
        left_ee_pos = self.get_endeff_pos('left')
        right_current_qpos = self._get_ur3_qpos()[:self.ur3_nqpos]
        left_current_qpos = self._get_ur3_qpos()[self.ur3_nqpos:]
        right_desired_ee_pos = right_ee_pos + right_ur3_act
        left_desired_ee_pos = left_ee_pos + left_ur3_act
        start = time.time()
        # ee_pos, null_obj_func, arm, q_init='current', threshold=0.01, threshold_null=0.001, max_iter=100, epsilon=1e-6
        right_q_des, right_iter_taken, right_err, right_null_obj = self.inverse_kinematics_ee(right_desired_ee_pos, self.null_obj_func, arm='right', threshold=0.001, max_iter = 10)
        left_q_des, left_iter_taken, left_err, left_null_obj = self.inverse_kinematics_ee(left_desired_ee_pos, self.null_obj_func, arm='left', threshold=0.001, max_iter = 10)
        # print('right_iter taken : {}, left_iter taken : {}, time : {}'.format(right_iter_taken, left_iter_taken, time.time()-start))
        if self.q_control_type =='speedj':
            right_ur3_action = (right_q_des-right_current_qpos)/(self.dt)
            left_ur3_action = (left_q_des-left_current_qpos)/(self.dt)
        elif self.q_control_type =='servoj':
            right_ur3_action = right_q_des
            left_ur3_action = left_q_des
        right_gripper_action = right_gripper_act
        left_gripper_action = left_gripper_act
        
        right_q_control_args, right_g_control_args = self._get_control_kwargs(self.q_control_type, self.g_control_type, right_ur3_action, right_gripper_action)
        left_q_control_args, left_g_control_args = self._get_control_kwargs(self.q_control_type, self.g_control_type, left_ur3_action, left_gripper_action) 
        
        command = {
                    'right': {
                        self.q_control_type : right_q_control_args,
                        self.g_control_type : right_g_control_args
                    },
                    'left': {
                        self.q_control_type : left_q_control_args,
                        self.g_control_type : left_g_control_args
                    }
                  }

        for _ in range(self.multi_step-1):
            self.env.step(self.action(command)) # obs is dict)
        return self.env.step(self.action(command))


    #TODO : if it is slow, you'd better use predefined dictionary format by using for loop to avoid if, elif execution per step
    def _get_control_kwargs(self, q_control_type, g_control_type, ur3_action, gripper_action):

        if q_control_type=='speedj':
            q_control_args = copy.deepcopy(self.speedj_args)
            q_control_args.update({'qd' : ur3_action})
        elif q_control_type=='servoj':
            q_control_args = copy.deepcopy(self.servoj_args)
            q_control_args.update({'q' : ur3_action})
        if g_control_type=='move_gripper_force':
            g_control_args = {'gf' : gripper_action}
        elif g_control_type=='move_gripper_position':
            g_control_args = {'g' : gripper_action}
        elif g_control_type=='move_gripper_velocity':
            g_control_args = {'gd' : gripper_action}
        
        return q_control_args, g_control_args
    
    def __getattr__(self, name):
        return getattr(self.env, name)


class DSCHODualUR3Env(DualUR3Env):
    def __init__(self, ur3_random_init_so3_constraint = 'vertical_side', *args, **kwargs):
        self.save_init_params(locals())
        self.init_qpos_candidates = {}
        self.ur3_random_init_so3_constraint = ur3_random_init_so3_constraint
        self.null_obj_func = SO3Constraint(SO3=self.ur3_random_init_so3_constraint)
        # 양팔 널찍이 벌려있는 상태
        # default_right_qpos = np.array([[-90.0, -90.0, -90.0, -90.0, -135.0, 180.0]])*np.pi/180.0 #[num_candidate+1, qpos_dim]
        default_left_qpos = np.array([[90.0, -90.0, 90.0, -90.0, 135.0, -180.0]])*np.pi/180.0 #[num_candidate+1, qpos_dim]
        
        # right : [0.15, -0.35, 0.9] left : [-0.2, -0.3, 0.8]
        # NOTE : 주의 ! qpos array rank is 2 !
        # [0.2, -0.3, 0.8]
        # default_right_qpos = np.array([[-0.73475149, -1.91237669, -1.78802014, -1.6064106, -2.07919236,  2.16932592]])
        # [0.15, -0.35, 0.9]
        # default_right_qpos = np.array([[-0.90259643, -2.24937667, -1.82423119, -1.23998854, -2.15827838,  2.2680261 ]])
        # [0.15, -0.35, 0.8]
        default_right_qpos = np.array([[-0.76263046, -2.21085609, -1.50821658, -1.57404046, -2.08100962, 2.19369591]])
        # default_left_qpos = np.array([[0.73490191, -1.22867589, 1.78775333, -1.53617814, 2.07956014, -2.16994491]])
        
        # add default qpos configuration        
        self.init_qpos_candidates['q_right_des'] =default_right_qpos
        self.init_qpos_candidates['q_left_des'] = default_left_qpos
        self.init_qpos_candidates['gripper_q_right_close'] = np.array([[0.70005217, 0.01419325, 0.0405478, 0.0134475, 0.74225534, 0.70005207, 0.01402114, 0.04054553, 0.01344841, 0.74224361]])
        self.init_qpos_candidates['gripper_q_left_close'] = np.array([[0.70005217, 0.01419325, 0.0405478, 0.0134475, 0.74225534, 0.70005207, 0.01402114, 0.04054553, 0.01344841, 0.74224361]])

        # for push or reach (closed gripper)
        if self.task in ['push', 'reach']:
            default_gripper_right_qpos = np.array([[0.70005217, 0.01419325, 0.0405478, 0.0134475, 0.74225534, 0.70005207, 0.01402114, 0.04054553, 0.01344841, 0.74224361]])
            default_gripper_left_qpos = np.array([[0.70005217, 0.01419325, 0.0405478, 0.0134475, 0.74225534, 0.70005207, 0.01402114, 0.04054553, 0.01344841, 0.74224361]])
            # add default qpos configuration        
            self.init_qpos_candidates['gripper_q_right_des'] =default_gripper_right_qpos
            self.init_qpos_candidates['gripper_q_left_des'] =default_gripper_left_qpos

        super().__init__(*args, **kwargs)
        
        

    def _get_init_qpos(self):

        init_qpos = self.init_qpos.copy()
        if self.ur3_random_init:            
            # randomly initilize ee pos. IK's q is default qpos (when no random init)
            q_right_des_candidates = self.init_qpos_candidates['q_right_des'] # [num_candidate, qpos dim]
            q_left_des_candidates = self.init_qpos_candidates['q_left_des']
            assert q_right_des_candidates.shape[0] == q_left_des_candidates.shape[0]

            right_ee_random_init_low = np.array([0.0, -0.45, 0.79])
            right_ee_random_init_high = np.array([0.3, -0.25, 0.8])
            left_ee_random_init_low = np.array([-0.3, -0.45, 0.79])
            left_ee_random_init_high = np.array([0.0, -0.25, 0.8])

            right_ee_random_init_space = Box(low=right_ee_random_init_low, high=right_ee_random_init_high, dtype=np.float32)
            left_ee_random_init_space = Box(low=left_ee_random_init_low, high=left_ee_random_init_high, dtype=np.float32)

            right_ee_pos = right_ee_random_init_space.sample()
            left_ee_pos = left_ee_random_init_space.sample()
            
            right_q_des, right_q_iter, right_err, right_null_obj_val = self.inverse_kinematics_ee(ee_pos = right_ee_pos,null_obj_func = self.null_obj_func, arm= 'right',q_init=q_right_des_candidates[0], max_iter=5)
            left_q_des, left_q_iter, left_err, left_null_obj_val = self.inverse_kinematics_ee(ee_pos = left_ee_pos,null_obj_func = self.null_obj_func, arm= 'left',q_init=q_left_des_candidates[0], max_iter=5)
            
            init_qpos[0:self.ur3_nqpos] = right_q_des
            init_qpos[self.ur3_nqpos+self.gripper_nqpos:2*self.ur3_nqpos+self.gripper_nqpos] = left_q_des

            # num_candidates = q_right_des_candidates.shape[0]
            # right_idx = np.random.choice(num_candidates,1) 
            # left_idx = np.random.choice(num_candidates,1) 
            # init_qpos[0:self.ur3_nqpos] = q_right_des_candidates[right_idx]
            # init_qpos[self.ur3_nqpos+self.gripper_nqpos:2*self.ur3_nqpos+self.gripper_nqpos] = q_left_des_candidates[left_idx]

        else :
            # Currently for dual env test with 0th index init qpos
            q_right_des_candidates = self.init_qpos_candidates['q_right_des'] # [num_candidate, qpos dim]
            q_left_des_candidates = self.init_qpos_candidates['q_left_des']
            
            right_idx = 0
            left_idx = 0
            init_qpos[0:self.ur3_nqpos] = q_right_des_candidates[right_idx]
            init_qpos[self.ur3_nqpos+self.gripper_nqpos:2*self.ur3_nqpos+self.gripper_nqpos] = q_left_des_candidates[left_idx]
        
        if self.task in ['push', 'reach']: # initially, close gripper
            gripper_q_right_des_candidates = self.init_qpos_candidates['gripper_q_right_des']
            gripper_q_left_des_candidates = self.init_qpos_candidates['gripper_q_left_des']
            right_idx = 0
            left_idx = 0
            init_qpos[self.ur3_nqpos:self.ur3_nqpos + self.gripper_nqpos] = gripper_q_right_des_candidates[right_idx]
            init_qpos[2*self.ur3_nqpos+self.gripper_nqpos:2*self.ur3_nqpos+2*self.gripper_nqpos] = gripper_q_left_des_candidates[left_idx]


        return init_qpos


    def get_endeff_pos(self, arm):
        if arm == 'right':
            q= self._get_ur3_qpos()[:self.ur3_nqpos]
        elif arm =='left' :
            q= self._get_ur3_qpos()[self.ur3_nqpos:]
        R, p, T = self.forward_kinematics_ee(q, arm)
        return p

    # state_goal should be defined in child class
    def get_current_goal(self):
        return self._state_goal.copy()
        
    def get_site_pos(self, siteName):
        try :
            _id = self.model.site_names.index(siteName)
        except Exception as e:
            return None
            
        return self.data.site_xpos[_id].copy()


    # state_goal, subgoals should be defined in child class
    def set_goal(self, goal):
        self._state_goal = goal
        self._set_goal_marker(goal)

    def set_subgoals(self, subgoals):
        assert isinstance(subgoals, list)
        self._state_subgoals = subgoals
        self._set_subgoal_marker(subgoals)

    def set_finalgoal(self, finalgoal):
        self._state_finalgoal = finalgoal
        self._set_finalgoal_marker(finalgoal)

    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('goal')] = (
            goal[-3:]
        )

    def _set_subgoal_marker(self, subgoals):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        assert isinstance(subgoals, list)
        for idx, subgoal in enumerate(subgoals):
            self.data.site_xpos[self.model.site_name2id('subgoal_'+str(idx+1))] = (
                subgoal[-3:]
        )

    def _set_finalgoal_marker(self, finalgoal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('finalgoal')] = (
            finalgoal[-3:]
        )

#single ur3 만들고 그 하위로 reachenv 만들수도
class DSCHOSingleUR3GoalEnv(DSCHODualUR3Env):
    
    # Sholud be used with URScriptWrapper
    
    def __init__(self,
                sparse_reward = False,
                # reduced_observation = False,
                trigonometry_observation = True, 
                ur3_random_init=False,
                full_state_goal = False,
                reward_by_ee = False, 
                automatically_set_spaces=False,
                fixed_goal_qvel = True, 
                reward_success_criterion='ee_pos',
                distance_threshold = 0.05,
                initMode='vertical',
                xml_filename = 'dscho_dual_ur3_object.xml',
                which_hand='right', 
                so3_constraint ='vertical_side',
                has_object = False,
                block_gripper = False,
                has_obstacle = False,
                task = 'pickandplace',
                observation_type='joint_q', #'ee_object_object', #'ee_object_all'
                warm_start = False,
                *args,
                **kwargs
                ):
        self.save_init_params(locals())

        self.full_state_goal = full_state_goal
        # self.reduced_observation = reduced_observation
        self.trigonometry_observation = trigonometry_observation
        self.fixed_goal_qvel = fixed_goal_qvel
        self.task = task
        self.observation_type = observation_type
        self.warm_start = warm_start
        # for LEAP(S=G) -> modified for S!=G (20200930)
        # assert (reduced_observation and not full_state_goal) or (not reduced_observation and full_state_goal)
        
        # assert (reduced_observation and not trigonometry_observation) or (not reduced_observation and trigonometry_observation)
        self.reward_by_ee = reward_by_ee
        #self.ur3_random_init = ur3_random_init
        
        self.curr_path_length = 0

        if so3_constraint == 'no' or so3_constraint is None:
            self.so3_constraint = NoConstraint()
        elif so3_constraint == 'upright':
            self.so3_constraint = UprightConstraint()
        else:
            self.so3_constraint = SO3Constraint(SO3=so3_constraint)
        
        self.sparse_reward = sparse_reward
        self.distance_threshold = distance_threshold
        
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.has_obstacle = has_obstacle

        if has_obstacle:
            raise NotImplementedError


        self.reward_success_criterion = reward_success_criterion
        #self.automatically_set_spaces = automatically_set_spaces
        self.which_hand = which_hand        
        super().__init__(xml_filename=xml_filename,
                        initMode=initMode, 
                        automatically_set_spaces=automatically_set_spaces, 
                        ur3_random_init=ur3_random_init, 
                        *args, 
                        **kwargs
                        )
        
        if not self.automatically_set_spaces:
            self._set_action_space()

        # self.obj_init_pos = self.get_obj_pos()
        # self.obj_names = ['obj']
        self.left_get_away_qpos = np.concatenate([np.array([-90.0, -90.0, 90.0, -90.0, 135.0, 0.0])*np.pi/180.0, np.zeros(self.gripper_nqpos)]) # it was self.gripper_nqpos
        self.right_get_away_qpos = np.concatenate([np.array([-90.0, -90.0, 90.0, -90.0, 135.0, 0.0])*np.pi/180.0, np.zeros(self.gripper_nqpos)])
        
        # if not self.reduced_observation:
        if self.observation_type=='joint_q':
            if self.trigonometry_observation:
                self.obs_nqpos = self.ur3_nqpos*2
            else :
                self.obs_nqpos = self.ur3_nqpos
        elif self.observation_type == 'ee_object_pos' :
            assert not self.trigonometry_observation
            self.obs_nqpos = 3 # ee_pos
        elif self.observation_type =='ee_object_all':
            assert not self.trigonometry_observation
            raise NotImplementedError
            self.obs_nqpos = None

        qpos_low = -np.ones(int(self.ur3_nqpos))*2*np.pi
        qpos_high = np.ones(int(self.ur3_nqpos))*2*np.pi
        qvel_low = -np.ones(int(self.ur3_nqpos))*0.01
        qvel_high = np.ones(int(self.ur3_nqpos))*0.01
        
        self.goal_qpos_space = Box(low=qpos_low, high=qpos_high, dtype=np.float32)
        self.goal_qvel_space = Box(low=qvel_low, high=qvel_high, dtype=np.float32)
        
        if self.which_hand =='right': 
            ee_low = np.array([-0.1, -0.5, 0.77])
            ee_high = np.array([0.35, -0.2, 0.95])
        
        elif self.which_hand =='left':
            ee_low = np.array([-0.35, -0.5, 0.77])
            ee_high = np.array([0.1, -0.2, 0.95])

        self.goal_ee_pos_space = Box(low = ee_low, high = ee_high, dtype=np.float32)
        
        # Currently, Set the goal obj space same sa ee pos sapce
        if self.which_hand =='right': 
            goal_obj_low = np.array([0.0, -0.45, 0.77])
            goal_obj_high = np.array([0.3, -0.3, 0.95])
        
        elif self.which_hand =='left':
            goal_obj_low = np.array([-0.3, -0.45, 0.77])
            goal_obj_high = np.array([0.0, -0.3, 0.95])

        self.goal_obj_pos_space = Box(low = goal_obj_low, high = goal_obj_high, dtype=np.float32)
        
        # TODO : think whether xy space is ouf of range of the table
        floor_z_height = np.array([0.73]) # xml보니 책상높이가 0.73?
        goal_obj_floor_low = np.concatenate([goal_obj_low[:-1], floor_z_height], axis =-1)
        goal_obj_floor_high = np.concatenate([goal_obj_high[:-1], floor_z_height], axis =-1)
        self.goal_obj_floor_space = Box(low = goal_obj_floor_low, high = goal_obj_floor_high, dtype=np.float32)


        self._state_goal = self.sample_goal(self.full_state_goal)
        

        observation, reward, done, _info = self.step(self.action_space.sample()) # goalenv는 return obs_dict
        assert not done
        
        self.observation_space = self._set_observation_space(observation) 
    
        
    # goal space == state space
    # ee_pos 뽑고, 그에 따른 qpos 계산(IK) or qpos뽑고 그에따른 ee_pos 계산(FK)
    # 우선은 후자로 생각(어처피 학습할땐 여기저기 goal 다 뽑고, 쓸때는 제한해서 goal 샘플할꺼니까)
    
    def sample_goal(self, full_state_goal):
        # need to mod for resample if goal is inside the wall
        if full_state_goal:
            
            t=0
            p = np.array([np.inf, np.inf, np.inf])
            # while not ((p <= self.goal_ee_pos_space.high).all() and (p >= self.goal_ee_pos_space.low).all()):
            while True:
                goal_qpos = np.random.uniform(
                    self.goal_qpos_space.low,
                    self.goal_qpos_space.high,
                    size=(self.goal_qpos_space.low.size),
                )
                # p = self.get_endeff_pos(arm=self.which_hand) 이걸로 구하면 샘플한 qpos가 아닌 현재 qpos기준 ee나오니까 안돼!
                R, p, _ = self.forward_kinematics_ee(goal_qpos, arm=self.which_hand)
                

                t+=1
            print('{} hand resample num : {}'.format(self.which_hand, t))
            if self.fixed_goal_qvel:
                goal_qvel = np.zeros(int(self.ur3_nqvel))
            else :
                goal_qvel = self.goal_qvel_space.sample()
            if self.trigonometry_observation:
                goal_qpos = np.concatenate([np.cos(goal_qpos), np.sin(goal_qpos)], axis = -1)

            goal = np.concatenate([goal_qpos, goal_qvel, p], axis =-1) #[qpos, qvel, ee_pos]
        else :
            if not self.has_object: # reach
                goal_ee_pos = np.random.uniform(
                    self.goal_ee_pos_space.low,
                    self.goal_ee_pos_space.high,
                    size=(self.goal_ee_pos_space.low.size),
                )
                goal = goal_ee_pos
            else: # pick and place, push, ...
                
                goal_obj_pos = np.random.uniform(
                    self.goal_obj_pos_space.low,
                    self.goal_obj_pos_space.high,
                    size=(self.goal_obj_pos_space.low.size),
                )
                if self.task in ['pickandplace']:
                    goal = goal_obj_pos
                elif self.task in ['push']:
                    goal = np.concatenate([goal_obj_pos[:2], np.array([self.goal_obj_pos_space.low[-1]])], axis=-1)
                elif self.task in ['assemble']:
                    goal = None
                    raise NotImplementedError
                elif self.task in ['drawer_open']:
                    goal = None
                    raise NotImplementedError
                else:
                    raise NotImplementedError

        return goal


    def reset_model(self):
        
        qpos = self._get_init_qpos() + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        
        if self.which_hand == 'right':
            start_p, end_p = self.ur3_nqpos+self.gripper_nqpos, 2*self.ur3_nqpos+2*self.gripper_nqpos
            start_v, end_v = self.ur3_nqvel+self.gripper_nqvel, 2*self.ur3_nqvel+2*self.gripper_nqvel
            qpos[start_p:end_p] = self.left_get_away_qpos
        elif self.which_hand == 'left':
            start_p, end_p = 0, self.ur3_nqpos+self.gripper_nqpos
            start_v, end_v = 0, self.ur3_nqvel+self.gripper_nqvel
            qpos[start_p:end_p] = self.right_get_away_qpos
            
        self.set_state(qpos, qvel)

        # randomly reset the initial position of an object
        if self.has_object:
            if self.task in ['pickandplace', 'push']:
                ee_pos = self.get_endeff_pos(arm=self.which_hand)
                if self.warm_start : # set object pos same as ee pos
                    object_pos = ee_pos.copy() - np.array([0.0, 0.0, 0.01]) # minus offset
                else:
                    object_xpos = np.random.uniform(
                                    self.goal_obj_pos_space.low,
                                    self.goal_obj_pos_space.high,
                                    size=(self.goal_obj_pos_space.low.size),
                                )[:2]
                    object_pos = np.concatenate([object_xpos, np.array([self.goal_obj_pos_space.low[-1]])], axis=-1)
                    
                    
                    while np.linalg.norm(object_pos - ee_pos) < 0.05:
                        object_xpos = np.random.uniform(
                                    self.goal_obj_pos_space.low,
                                    self.goal_obj_pos_space.high,
                                    size=(self.goal_obj_pos_space.low.size),
                                )[:2]
                        object_pos = np.concatenate([object_xpos, np.array([self.goal_obj_pos_space.low[-1]])], axis=-1)
                    # print('In reset model, ee pos : {} obj pos : {}'.format(ee_pos, object_pos))

                object_qpos = self.sim.data.get_joint_qpos('objjoint')
                assert object_qpos.shape == (7,)
                object_qpos[:3] = object_pos
                self.sim.data.set_joint_qpos('objjoint', object_qpos)
            else:
                pass
        
        self._state_goal = self.sample_goal(full_state_goal = self.full_state_goal)
        
        if self.has_object: # reach인 경우엔 필요x
            while np.linalg.norm(object_pos - self._state_goal) < 0.05:
                self._state_goal = self.sample_goal(full_state_goal = self.full_state_goal)
        
        self.sim.forward()
        
        observation = self._get_obs()

        # observation = super().reset_model() # init qpos,qvel set_state and get_obs
        
        info = {
            'is_success': self._is_success(observation['achieved_goal'], self._state_goal),
            'right_ee_pos' : self.get_endeff_pos(arm='right'),
            'left_ee_pos' : self.get_endeff_pos(arm='left'),
            # 'object_pos': self.get_obj_pos(name='obj'),
            # 'object_qpos': self.get_obj_qpos(name='obj'),
            # 'object_vel': self.get_obj_qvel(name='obj'),
            # 'object_quat': self.get_obj_quat(name='obj'),
            'null_obj_val' : self._calculate_so3_error().copy(),
            'l2_distance_to_goal' : np.linalg.norm(observation['desired_goal']-observation['achieved_goal'], ord=2, axis = -1), 
            'l1_distance_to_goal' : np.linalg.norm(observation['desired_goal']-observation['achieved_goal'], ord=1, axis = -1),
            'l2_distance_to_goal_of_interest' : np.linalg.norm(observation['desired_goal'][-3:]-observation['achieved_goal'][-3:], ord=2, axis = -1), # diffrent from reward_dim 
            'l1_distance_to_goal_of_interest' : np.linalg.norm(observation['desired_goal'][-3:]-observation['achieved_goal'][-3:], ord=1, axis = -1),
        }
        if not self.full_state_goal:
            info.update({'l2_distance_to_goal_for_reward' : info['l2_distance_to_goal'],
                         'l1_distance_to_goal_for_reward' : info['l1_distance_to_goal']})
        elif self.reward_by_ee:
            info.update({'l2_distance_to_goal_for_reward' : np.linalg.norm(observation['desired_goal'][-3:]-observation['achieved_goal'][-3:], ord=2, axis=-1),  
                         'l1_distance_to_goal_for_reward' : np.linalg.norm(observation['desired_goal'][-3:]-observation['achieved_goal'][-3:], ord=1, axis=-1),
                         })
        else:
            info.update({'l2_distance_to_goal_for_reward' : np.linalg.norm(
                np.concatenate([observation['desired_goal'][:self.obs_nqpos], observation['desired_goal'][-3:]], axis =-1)-
                np.concatenate([observation['achieved_goal'][:self.obs_nqpos], observation['achieved_goal'][-3:]], axis =-1),
                ord=2, axis = -1
            ),
            'l1_distance_to_goal_for_reward' : np.linalg.norm(
                np.concatenate([observation['desired_goal'][:self.obs_nqpos], observation['desired_goal'][-3:]], axis =-1)-
                np.concatenate([observation['achieved_goal'][:self.obs_nqpos], observation['achieved_goal'][-3:]], axis =-1),
                ord=1, axis = -1
            )})
        self.info = copy.deepcopy(info)

        self._set_goal_marker(self._state_goal)
        # self._set_subgoal_marker(self._state_subgoals)
        # self._set_finalgoal_marker(self._state_finalgoal)
        self.curr_path_length = 0
        # self._set_obj_xyz(np.array([0.0, -0.8, 0.65]))
        # original_ob['desired_goal'] = self._state_goal
        return observation
        
    # Only ur3 qpos,vel(not include gripper), object pos(achieved_goal), desired_goal
    def _get_obs(self):
        
        if self.which_hand=='right':
            qpos = self._get_ur3_qpos()[:self.ur3_nqpos]
            qvel = self._get_ur3_qvel()[:self.ur3_nqvel]
            ee_pos = self.get_endeff_pos(arm='right') # qpos idx찾아서 써야

        elif self.which_hand=='left':
            qpos = self._get_ur3_qpos()[self.ur3_nqpos:]
            qvel = self._get_ur3_qvel()[self.ur3_nqvel:]
            ee_pos = self.get_endeff_pos(arm='left')
        
        if self.trigonometry_observation:
            qpos = np.concatenate([np.cos(qpos), np.sin(qpos)], axis = -1)

        if self.has_object:
            obj_pos = self.get_obj_pos(name='obj')
        else :
            obj_pos = np.array([])

        if self.observation_type=='joint_q':
            obs = np.concatenate([qpos, qvel, ee_pos, obj_pos])
        elif self.observation_type == 'ee_object_pos':
            obs = np.concatenate([ee_pos, obj_pos])
        elif self.observation_type == 'ee_object_all':
            raise NotImplementedError
            ee_vel = None
            obj_vel = None
            obj_rot = None
            obs = np.concatenate([ee_pos, obj_pos])

        '''
        For reference, Fetch Env's observation consist of
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])        
        '''

        # if self.reduced_observation:
        #     obs = np.concatenate([ee_pos, obj_pos])
        # else:
        #     pass
        
        if self.full_state_goal:
            achieved_goal = obs
        else :
            if not self.has_object: # reach
                achieved_goal = ee_pos
            else: # pick and place, push, ...
                achieved_goal = obj_pos
            
        return {
            'observation' : obs.copy(),
            'achieved_goal' : achieved_goal.copy(),
            'desired_goal' : self._state_goal.copy(), 
        }    
    
    def step(self, action):
        # actions of remaning arm will be garbage
        action = action.copy()

        self.do_simulation(action, self.frame_skip)
        self.curr_path_length +=1

        # print('curr_path_length ', self.curr_path_length)

        observation = self._get_obs()
        done = False
        
        info = {
            'is_success': self._is_success(observation['achieved_goal'], self._state_goal),
            'right_ee_pos' : self.get_endeff_pos(arm='right'),
            'left_ee_pos' : self.get_endeff_pos(arm='left'),
            # 'object_pos': self.get_obj_pos(name='obj'),
            # 'object_qpos': self.get_obj_qpos(name='obj'),
            # 'object_vel': self.get_obj_qvel(name='obj'),
            # 'object_quat': self.get_obj_quat(name='obj'),
            'null_obj_val' : self._calculate_so3_error().copy(),
            'l2_distance_to_goal' : np.linalg.norm(observation['desired_goal']-observation['achieved_goal'], ord=2, axis = -1), 
            'l1_distance_to_goal' : np.linalg.norm(observation['desired_goal']-observation['achieved_goal'], ord=1, axis = -1),
            'l2_distance_to_goal_of_interest' : np.linalg.norm(observation['desired_goal'][-3:]-observation['achieved_goal'][-3:], ord=2, axis = -1), # diffrent from reward_dim 
            'l1_distance_to_goal_of_interest' : np.linalg.norm(observation['desired_goal'][-3:]-observation['achieved_goal'][-3:], ord=1, axis = -1),
        }
        if not self.full_state_goal:
            info.update({'l2_distance_to_goal_for_reward' : info['l2_distance_to_goal'],
                         'l1_distance_to_goal_for_reward' : info['l1_distance_to_goal']})
        elif self.reward_by_ee:
            info.update({'l2_distance_to_goal_for_reward' : np.linalg.norm(observation['desired_goal'][-3:]-observation['achieved_goal'][-3:], ord=2, axis=-1),  
                         'l1_distance_to_goal_for_reward' : np.linalg.norm(observation['desired_goal'][-3:]-observation['achieved_goal'][-3:], ord=1, axis=-1),
                         })
        else:
            info.update({'l2_distance_to_goal_for_reward' : np.linalg.norm(
                np.concatenate([observation['desired_goal'][:self.obs_nqpos], observation['desired_goal'][-3:]], axis =-1)-
                np.concatenate([observation['achieved_goal'][:self.obs_nqpos], observation['achieved_goal'][-3:]], axis =-1),
                ord=2, axis = -1
            ),
            'l1_distance_to_goal_for_reward' : np.linalg.norm(
                np.concatenate([observation['desired_goal'][:self.obs_nqpos], observation['desired_goal'][-3:]], axis =-1)-
                np.concatenate([observation['achieved_goal'][:self.obs_nqpos], observation['achieved_goal'][-3:]], axis =-1),
                ord=1, axis = -1
            )})
        self.info = copy.deepcopy(info)
        reward = self.compute_reward(observation['achieved_goal'], self._state_goal, info)
       
        # process to make remaing arm not to move
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()        
        if self.which_hand =='right':
            #left arm's qpos,qvel index
            start_p, end_p = self.ur3_nqpos+self.gripper_nqpos, 2*self.ur3_nqpos+2*self.gripper_nqpos
            start_v, end_v = self.ur3_nqvel+self.gripper_nqvel, 2*self.ur3_nqvel+2*self.gripper_nqvel
            qpos[start_p:end_p] = self.left_get_away_qpos
            qvel[start_v:end_v] = np.zeros(end_v-start_v)
        elif self.which_hand=='left':
            #right arm's qpos,qvel index
            start_p, end_p = 0, self.ur3_nqpos+self.gripper_nqpos
            start_v, end_v = 0, self.ur3_nqvel+self.gripper_nqvel
            qpos[start_p:end_p] = self.right_get_away_qpos
            qvel[start_v:end_v] = np.zeros(end_v-start_v)
        
        self.set_state(qpos, qvel)
        # set state하면 site pos도 다 초기화됨! #TODO: 이 부분은 wrapper에 있을 함수가 아님!
        self._set_goal_marker(self._state_goal)
        # self._set_subgoal_marker(self._state_subgoals)
        # self._set_finalgoal_marker(self._state_finalgoal)
        # print('env state goal : {}'.format(self.env._state_goal))
        
        #TODO: Should consider how to address done
        # done = True if info['is_success'] else False
        
        return observation, reward, done, info


    def _calculate_so3_error(self):
        if self.which_hand =='right':
            ur3_qpos = self._get_ur3_qpos()[:self.ur3_nqpos]    
        elif self.which_hand =='left':
            ur3_qpos = self._get_ur3_qpos()[self.ur3_nqpos:]

        SO3, x, _ = self.forward_kinematics_ee(ur3_qpos, arm=self.which_hand)
        null_obj_val = self.so3_constraint.evaluate(SO3)

        return null_obj_val
    
    def get_info(self):
        return copy.deepcopy(self.info)

    def convert_goal_for_reward(self, goal):
        if goal.ndim==1:
            if not self.full_state_goal:
                return goal
            elif self.reward_by_ee:
                return goal[-3:]
            else: #exclude qvel in reward computation in outer wrapper
                return np.concatenate([goal[:self.obs_nqpos], goal[-3:]], axis =-1)
        elif goal.ndim==2:
            if not self.full_state_goal:
                return goal
            elif self.reward_by_ee:
                return goal[:, -3:]
            else: #exclude qvel in reward computation in outer wrapper
                return np.concatenate([goal[:, :self.obs_nqpos], goal[:, -3:]], axis =-1)
        else :
            raise NotImplementedError

    def convert_goal_for_reward_tf(self, goals):
        #Caution : Assume batch data is given.
        if not self.full_state_goal:
            return goals
        elif self.reward_by_ee:
            return goals[:, -3:]
        else: #exclude qvel in reward computation in outer wrapper
            return tf.concat([goals[:, :self.obs_nqpos], goals[:, -3:]], axis =-1)

    def _is_success(self, achieved_goal, desired_goal):
        if self.reward_success_criterion=='full_state':
            d = np.linalg.norm(achieved_goal-desired_goal)
        elif self.reward_success_criterion=='ee_pos':
            d = np.linalg.norm(achieved_goal[-3:]-desired_goal[-3:])
        else :
            raise NotImplementedError
        return (d < self.distance_threshold).astype(np.float32)

    # Has no meaning in TDM, LEAP
    def compute_reward(self, achieved_goal, desired_goal, info):
        if self.reward_success_criterion=='full_state':
            placingDist = np.linalg.norm(achieved_goal - desired_goal)
        elif self.reward_success_criterion=='ee_pos':
            placingDist = np.linalg.norm(achieved_goal[-3:] - desired_goal[-3:])
       
        if self.sparse_reward : 
            if placingDist < self.distance_threshold:
                reward = 0.0
            else :
                reward = -1.0
        else :
            reward = -placingDist

        return reward


    # obj related
    def get_obj_pos(self, name=None):
        if not self.has_object:
            raise NotImplementedError
        if name is None:
            return self.data.get_body_xpos('obj')
        else :
            return self.data.get_body_xpos(name) 

    def get_obj_quat(self, name=None):
        if not self.has_object:
            raise NotImplementedError
        if name is None:
            return self.data.get_body_xquat('obj')
        else :
            return self.data.get_body_xquat(name)
    
    def get_obj_qpos(self, name=None):
        if not self.has_object:
            raise NotImplementedError
        if name is None:
            body_idx = self.model.body_names.index('obj')
        else :
            body_idx = self.model.body_names.index(name)
        jnt_idx = self.model.body_jntadr[body_idx]
        qpos_start_idx = self.model.jnt_qposadr[jnt_idx]
        
        qpos = self.data.qpos.flat.copy()
        return qpos[qpos_start_idx:qpos_start_idx+7]

    def get_obj_qvel(self, name=None):
        if not self.has_object:
            raise NotImplementedError
        if name is None:
            body_idx = self.model.body_names.index('obj')
        else :
            body_idx = self.model.body_names.index(name)
        jnt_idx = self.model.body_jntadr[body_idx]
        qvel_start_idx = self.model.jnt_dofadr[jnt_idx]
    
        qvel = self.data.qvel.flat.copy()
        return qvel[qvel_start_idx:qvel_start_idx+6]


    def _set_objCOM_marker(self):
        
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        if not self.has_object:
            raise NotImplementedError
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
        if not self.has_object:
            raise NotImplementedError
        body_idx = self.model.body_names.index('obj')
        jnt_idx = self.model.body_jntadr[body_idx]
        qpos_start_idx = self.model.jnt_qposadr[jnt_idx]
        qvel_start_idx = self.model.jnt_dofadr[jnt_idx]
        
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[qpos_start_idx:qpos_start_idx+3] = pos.copy() #자세는 previous에서 불변
        qvel[qvel_start_idx:qvel_start_idx+6] = 0 #object vel
        self.set_state(qpos, qvel)








class DSCHOSingleUR3PickAndPlaceEnv(DSCHOSingleUR3GoalEnv):
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(has_object=True, block_gripper=False, task='pickandplace', *args, **kwargs)

class DSCHOSingleUR3PushEnv(DSCHOSingleUR3GoalEnv):
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(has_object=True, block_gripper=True, task='push',*args, **kwargs)

class DSCHOSingleUR3ReachEnv(DSCHOSingleUR3GoalEnv):
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(has_object=False, block_gripper=True, task='reach', *args, **kwargs)

class DSCHOSingleUR3AssembleEnv(DSCHOSingleUR3GoalEnv):
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(has_object=True, block_gripper=False, task='assemble', *args, **kwargs)

class DSCHOSingleUR3DrawerOpenEnv(DSCHOSingleUR3GoalEnv):
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(has_object=True, block_gripper=False, task='drawer_open', *args, **kwargs)


# class DSCHODualUR3PickAndPlaceEnvObstacle(DSCHODualUR3PickAndPlaceEnv):
#     def __init__(self, *args, **kwargs):
#         raise NotImplementedError
#         self.save_init_params(locals())
#         # xml_filename = 'dscho_dual_ur3_obstacle.xml'
#         super().__init__( *args, **kwargs)
    
# class DSCHOSingleUR3ReachEnvObstacle(DSCHOSingleUR3ReachEnv):
#     def __init__(self, *args, **kwargs):
#         raise NotImplementedError
#         self.save_init_params(locals())
#         # xml_filename = 'dscho_dual_ur3_obstacle.xml'
#         super().__init__( *args, **kwargs)
        

