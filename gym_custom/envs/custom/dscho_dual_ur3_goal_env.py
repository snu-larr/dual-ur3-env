import copy
import numpy as np
import os
import warnings
import gym_custom
from gym_custom.spaces import Box
from gym_custom import utils
# from gym_custom.envs.mujoco import MujocoEnv
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
    def __init__(self, env, q_control_type, g_control_type, multi_step, gripper_action, serializable_initialized = False, *args, **kwargs):
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
            right_gripper_action = action[self.ur3_act_dim:self.ur3_act_dim+self.gripper_act_dim]
            left_ur3_action = action[self.ur3_act_dim+self.gripper_act_dim:2*self.ur3_act_dim+self.gripper_act_dim]
            left_gripper_action = action[2*self.ur3_act_dim+self.gripper_act_dim:2*self.ur3_act_dim+2*self.gripper_act_dim]
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

    def __init__(self, env, q_control_type, g_control_type, multi_step, gripper_action, serializable_initialized = False, *args, **kwargs):
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
        for _ in range(self.multi_step-1):
            self._step(action)
        return self._step(action)        

    def _step(self, action): 
        # assume action is np.array(ur3(6)) (if gripper_action is True, then , gripper(1) dimension added )
        ur3_act = action[:self.ur3_act_dim]
        if self.gripper_action:
            gripper_act = action[-self.gripper_act_dim:]
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
        
    def reset(self, **kwargs):
        # return self.env.reset(**kwargs)
        return super().reset(**kwargs)

    # Wrapper 에서 multistep을 해야 제일 lowlevel에선 매 timestep마다 command 새로계산함으로써 정확도 증가되는데, 내부적으로 IK sol 쓰다보니 이런구조는 아니라 정확도 살짝 떨어질수도
    def step(self, action):
        # down scale [-1,1] to [-0.005, 0.005]
        
        # action = copy.deepcopy(action) # 통째로 *downscale이면 문제 없는데 index로 접근할땐 array가 mutable이라 copy해줘야함, but 매 스텝마다 action을 새로 뽑는상황이라면 굳이 이렇게 안해도 상관없음. 똑같은 action으로 계속 step밟을 때나 문제가 되는거지
        action[:self.ee_xyz_pos_dim] = np.clip(action[:self.ee_xyz_pos_dim], -1, 1)
        action[:self.ee_xyz_pos_dim] = self.action_downscale*action[:self.ee_xyz_pos_dim]
        ur3_act = action[:self.ee_xyz_pos_dim] # delta xyz pos of ee
        if self.gripper_action:
            gripper_act = action[-self.gripper_act_dim:]
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
            right_gripper_act = action[self.ee_xyz_pos_dim:self.ee_xyz_pos_dim+self.gripper_act_dim]
            left_ur3_act = action[self.ee_xyz_pos_dim+self.gripper_act_dim:2*self.ee_xyz_pos_dim+self.gripper_act_dim]
            left_gripper_act = action[2*self.ee_xyz_pos_dim+self.gripper_act_dim:2*self.ee_xyz_pos_dim+2*self.gripper_act_dim]
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
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        # super().__init__(xml_filename, initMode, automatically_set_spaces=automatically_set_spaces)
        default_right_qpos = np.array([[-90.0, -90.0, -90.0, -90.0, -135.0, 180.0]])*np.pi/180.0 #[num_candidate+1, qpos_dim]
        default_left_qpos = np.array([[90.0, -90.0, 90.0, -90.0, 135.0, -180.0]])*np.pi/180.0 #[num_candidate+1, qpos_dim]
        if 'obstacle_v0' in self.xml_filename :
            self.init_qpos_candidates = joblib.load('init_qpos_obstacle-v0.pkl')
        elif 'obstacle_v1' in self.xml_filename :
            self.init_qpos_candidates = joblib.load('init_qpos_obstacle-v1.pkl')
        elif 'obstacle_v2' in self.xml_filename :
            self.init_qpos_candidates = joblib.load('init_qpos_obstacle-v2.pkl')
        elif 'obstacle_v3' in self.xml_filename :
            self.init_qpos_candidates = joblib.load('init_qpos_obstacle-v3.pkl')
        elif 'obstacle_v4' in self.xml_filename :
            self.init_qpos_candidates = joblib.load('init_qpos_obstacle-v4.pkl')
        elif 'obstacle_v5' in self.xml_filename :
            self.init_qpos_candidates = joblib.load('init_qpos_obstacle-v5.pkl')
        else :
            raise NotImplementedError

        # add default qpos configuration
        self.init_qpos_candidates['q_right_des'] = np.concatenate([self.init_qpos_candidates['q_right_des'], default_right_qpos], axis = 0)
        self.init_qpos_candidates['q_left_des'] = np.concatenate([self.init_qpos_candidates['q_left_des'], default_left_qpos], axis = 0)

    def _get_init_qpos(self):

        init_qpos = self.init_qpos.copy()
        if self.ur3_random_init:
            q_right_des_candidates = self.init_qpos_candidates['q_right_des'] # [num_candidate, qpos dim]
            q_left_des_candidates = self.init_qpos_candidates['q_left_des']
            
            assert q_right_des_candidates.shape[0] == q_left_des_candidates.shape[0]

            num_candidates = q_right_des_candidates.shape[0]
            right_idx = np.random.choice(num_candidates,1) 
            left_idx = np.random.choice(num_candidates,1) 
            init_qpos[0:self.ur3_nqpos] = q_right_des_candidates[right_idx]
            init_qpos[self.ur3_nqpos+self.gripper_nqpos:2*self.ur3_nqpos+self.gripper_nqpos] = q_left_des_candidates[left_idx]

        else :
            # Currently for dual env test with 0th index init qpos
            q_right_des_candidates = self.init_qpos_candidates['q_right_des'] # [num_candidate, qpos dim]
            q_left_des_candidates = self.init_qpos_candidates['q_left_des']
            if 'obstacle_v0' in self.xml_filename : # Not implemented
                right_idx = -1
                left_idx = -1  
            elif 'obstacle_v1' in self.xml_filename :
                right_idx = -1
                left_idx = -1
            elif 'obstacle_v2' in self.xml_filename :
                right_idx = 0
                left_idx = 0
            elif 'obstacle_v3' in self.xml_filename : # Not implemented
                right_idx = -1
                left_idx = -1
            elif 'obstacle_v4' in self.xml_filename :
                right_idx = 0
                left_idx = 0
            elif 'obstacle_v5' in self.xml_filename :
                right_idx = 0
                left_idx = 0
            init_qpos[0:self.ur3_nqpos] = q_right_des_candidates[right_idx]
            init_qpos[self.ur3_nqpos+self.gripper_nqpos:2*self.ur3_nqpos+self.gripper_nqpos] = q_left_des_candidates[left_idx]
        
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



class DSCHODualUR3ObjectEnv(DSCHODualUR3Env):
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        # super().__init__(xml_filename, initMode, automatically_set_spaces=automatically_set_spaces)

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



class DSCHODualUR3PickAndPlaceEnv(DSCHODualUR3ObjectEnv):

    def __init__(self,
                sparse_reward = False,
                reduced_observation = False,
                trigonometry_observation = True, 
                ur3_random_init = False, 
                goal_obj_random_init=False, 
                automatically_set_spaces=False,
                init_goal = (0.0, -0.3, 0.95),
                distance_threshold = 0.05,
                initMode='vertical',
                xml_filename = 'dscho_dual_ur3.xml',
                *args,
                **kwargs
                ):
        self.save_init_params(locals())
        self.reduced_observation = reduced_observation
        self.trigonometry_observation = trigonometry_observation
        
        assert (reduced_observation and not trigonometry_observation) or (not reduced_observation and trigonometry_observation)
        
        self.goal_obj_random_init = goal_obj_random_init
        self.curr_path_length = 0
        
        self.sparse_reward = sparse_reward
        self.distance_threshold = distance_threshold
        
        self._state_goal = np.array(init_goal)
        
        # TODO: goal setting for multiple tasks!
        
        goal_low = np.array([-0.2, -0.6, 0.77])
        goal_high = np.array([0.2, -0.2, 1.02])
        obj_low = np.array([-0.2, -0.6, 0.769])
        obj_high = np.array([0.2, -0.2, 0.771])
        self.goal_and_obj_space = Box(
            np.hstack((goal_low, obj_low)),
            np.hstack((goal_high, obj_high)),
        )
        
        super().__init__(xml_filename=xml_filename, initMode=initMode, automatically_set_spaces=automatically_set_spaces, ur3_random_init = ur3_random_init)

        if 'pickandplace' in xml_filename:
            self.place_box_center = place_box_center = self._get_placebox_center()
            self._state_goal = place_box_center
        else:
            self._state_goal = self.get_site_pos('goal')
            

        if 'obstacle' in xml_filename or 'Obstacle' in xml_filename:
            self.obstacle_exist = True
        else :
            self.obstacle_exist = False

        
        self._state_right_subgoals = []
        self._state_left_subgoals = []
        for idx, site_name in enumerate(self.model.site_names):
            if 'right_subgoal' in site_name:
                self._state_right_subgoals.append(self.data.site_xpos[self.model.site_name2id(site_name)].copy())
            elif 'left_subgoal' in site_name:
                self._state_left_subgoals.append(self.data.site_xpos[self.model.site_name2id(site_name)].copy())
            else :
                pass
        # for i in range(3):
        #     self._state_right_subgoals.append(self.data.site_xpos[self.model.site_name2id('right_subgoal_'+str(i+1))].copy())
        #     self._state_left_subgoals.append(self.data.site_xpos[self.model.site_name2id('left_subgoal_'+str(i+1))].copy())
        
        self._state_finalgoal = self._state_goal.copy()            

        if not automatically_set_spaces:
            self._set_action_space()
  
        observation, reward, done, _info = self.step(self.action_space.sample()) # goalenv는 return obs_dict
        assert not done
        
        self.observation_space = self._set_observation_space(observation)

        self.obj_init_pos = self.get_obj_pos(name='obj')
        self.obj_names = ['obj']

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        
        info = {
                'is_success': self._is_success(ob['achieved_goal'], self._state_goal),
                'l2_distance_to_goal' : np.linalg.norm(ob['desired_goal']-ob['achieved_goal'], ord=2, axis = -1),
                'l1_distance_to_goal' : np.linalg.norm(ob['desired_goal']-ob['achieved_goal'], ord=1, axis = -1),
                'desired_goal' : ob['desired_goal'],
                'right_ee_pos' : self.get_endeff_pos(arm='right'),
                'left_ee_pos' : self.get_endeff_pos(arm='left'),
                'right_agent_obs' : self.right_agent_obs,
                'left_agent_obs' : self.left_agent_obs,
                'object_pos': self.get_obj_pos(name='obj'),
                'object_qpos': self.get_obj_qpos(name='obj'),
                'object_vel': self.get_obj_qvel(name='obj'),
                'object_quat': self.get_obj_quat(name='obj'),
                'object_right_grasp_point' : self.get_site_pos('objSite'),
                'object_left_grasp_point' : self.get_site_pos('second_objSite'),
                'state_finalgoal' : self._state_finalgoal.copy(),
                'state_goal' : self._state_goal,
                'state_right_subgoals' : self._state_right_subgoals,
                'state_left_subgoals' : self._state_left_subgoals,
                'place_point' : self.get_site_pos('insertion'),
                'place_box_center' : self._get_placebox_center(),
                
            }

        #TODO: Should consider how to address done
        done = True if info['is_success'] else False

        self.info = copy.deepcopy(info)
        reward = self.compute_reward(ob['achieved_goal'], self._state_goal, None)
        self._set_goal_marker(self._state_goal)
        self._set_subgoal_marker(self._state_right_subgoals, 'right')
        self._set_subgoal_marker(self._state_left_subgoals, 'left')
        self.curr_path_length +=1
        return ob, reward, done, info

    def _get_obs(self):
        
        right_qpos = self._get_ur3_qpos()[:self.ur3_nqpos]
        right_qvel = self._get_ur3_qvel()[:self.ur3_nqvel]
        right_ee_pos = self.get_endeff_pos(arm='right') # qpos idx찾아서 써야


        left_qpos = self._get_ur3_qpos()[self.ur3_nqpos:]
        left_qvel = self._get_ur3_qvel()[self.ur3_nqvel:]
        left_ee_pos = self.get_endeff_pos(arm='left')
        
        if self.trigonometry_observation:
            right_qpos = np.concatenate([np.cos(right_qpos), np.sin(right_qpos)], axis = -1)
            left_qpos = np.concatenate([np.cos(left_qpos), np.sin(left_qpos)], axis = -1)

        obj_pos = self.get_obj_pos(name='obj')

        obs = np.concatenate([right_qpos, right_qvel, right_ee_pos, left_qpos, left_qvel, left_ee_pos, obj_pos])
        
        # goal까지 augment된 obs는 x. dualenv는 goal까지 관여하는건 아니니까
        self.right_agent_obs = np.concatenate([right_qpos, right_qvel, right_ee_pos]).copy()
        self.left_agent_obs = np.concatenate([left_qpos, left_qvel, left_ee_pos]).copy()
        
        if self.reduced_observation:
            obs = np.concatenate([right_ee_pos, left_ee_pos, obj_pos])
        else:
            pass
        
        achieved_goal = obj_pos

        return {
            'observation' : obs.copy(),
            'achieved_goal' : achieved_goal.copy(),
            'desired_goal' : self._state_goal.copy(), 
        }    
    
    def get_info(self):
        return copy.deepcopy(self.info)

    def _is_success(self, achieved_goal, desired_goal):
        if 'pickandplace' in self.xml_filename:
            place_box_sites = self._get_placebox_sites()
            '''
            3     1
              box
            4     2
            '''
            #x,y는 박스 안에 있고, z는 placebox 평면 기준 10cm이내
            success_criterion = (place_box_sites[2][0] < achieved_goal[0] < place_box_sites[0][0]) \
                                    *(place_box_sites[1][1] < achieved_goal[1] < place_box_sites[0][1]) \
                                        *(place_box_sites[0][2]-0.05 < achieved_goal[2] < place_box_sites[0][2]+0.05)
            
            
            
            return success_criterion
        else :
            d = np.linalg.norm(achieved_goal-desired_goal, axis = -1)
            return (d < self.distance_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, desired_goal, info):
        if 'pickandplace' in self.xml_filename:
            place_box_sites = self._get_placebox_sites()
            '''
            3     1
              box
            4     2
            '''
            #x,y는 박스 안에 있고, z는 placebox 평면 기준 10cm이내
            assert place_box_sites[2][0] < place_box_sites[0][0]
            assert place_box_sites[1][1] < place_box_sites[0][1]
            
            success_criterion = (place_box_sites[2][0] < achieved_goal[0] < place_box_sites[0][0]) \
                                    *(place_box_sites[1][1] < achieved_goal[1] < place_box_sites[0][1]) \
                                        *(place_box_sites[0][2]-0.05 < achieved_goal[2] < place_box_sites[0][2]+0.05)
                                        
            distance = np.linalg.norm(achieved_goal - desired_goal, axis =-1)
            
            if not self.sparse_reward:
                return -distance
            else:
                if success_criterion:
                    return 1.0
                else :
                    return 0.0
            
        else :
            distance = np.linalg.norm(achieved_goal - desired_goal, axis = -1)
            if not self.sparse_reward:
                return -distance
            else:
                if distance < self.distance_threshold:
                    return 1.0
                else :
                    return 0.0

    # super's reset_model
    # def reset_model(self):
    #     '''overridable method'''
    #     qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
    #     qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()
    
        

    def reset_model(self):
        qpos = self._get_init_qpos() + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        ob = self._get_obs()
        ###########################
        # ob = super().reset_model() # obs is dict because of get_obs
        if self.goal_obj_random_init :
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
        self._set_subgoal_marker(self._state_right_subgoals, 'right')
        self._set_subgoal_marker(self._state_left_subgoals, 'left')
        
        self._set_obj_xyz(self.obj_init_pos)
        self.curr_path_length = 0
        

        self.info = {
                    'is_success': self._is_success(ob['achieved_goal'], self._state_goal),
                    'l2_distance_to_goal' : np.linalg.norm(ob['desired_goal']-ob['achieved_goal'], ord=2, axis = -1),
                    'l1_distance_to_goal' : np.linalg.norm(ob['desired_goal']-ob['achieved_goal'], ord=1, axis = -1),
                    'desired_goal' : ob['desired_goal'],
                    'right_ee_pos' : self.get_endeff_pos(arm='right'),
                    'left_ee_pos' : self.get_endeff_pos(arm='left'),
                    'right_agent_obs' : self.right_agent_obs,
                    'left_agent_obs' : self.left_agent_obs,
                    'object_pos': self.get_obj_pos(name='obj'),
                    'object_qpos': self.get_obj_qpos(name='obj'),
                    'object_vel': self.get_obj_qvel(name='obj'),
                    'object_quat': self.get_obj_quat(name='obj'),
                    'object_right_grasp_point' : self.get_site_pos('objSite'),
                    'object_left_grasp_point' : self.get_site_pos('second_objSite'),
                    'state_finalgoal' : self._state_finalgoal.copy(),
                    'state_goal' : self._state_goal,
                    'state_right_subgoals' : self._state_right_subgoals,
                    'state_left_subgoals' : self._state_left_subgoals,
                    'place_point' : self.get_site_pos('insertion'),
                    'place_box_center' : self._get_placebox_center(),
                    
                }

        return ob

    def set_right_subgoals(self, subgoals):
        assert isinstance(subgoals, list)
        self._state_right_subgoals = subgoals
        self._set_subgoal_marker(subgoals, 'right')

    def set_left_subgoals(self, subgoals):
        assert isinstance(subgoals, list)
        self._state_left_subgoals = subgoals
        self._set_subgoal_marker(subgoals, 'left')

    # def set_finalgoal(self, finalgoal):
    #    self._state_finalgoal = finalgoal
    #    self._set_finalgoal_marker(finalgoal)

    def _set_subgoal_marker(self, subgoals, which_hand=None):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        assert isinstance(subgoals, list)
        if which_hand =='right':
            for idx, subgoal in enumerate(subgoals):
                self.data.site_xpos[self.model.site_name2id('right_subgoal_'+str(idx+1))] = (
                    subgoal[-3:]
            )
        elif which_hand =='left':
            for idx, subgoal in enumerate(subgoals):
                self.data.site_xpos[self.model.site_name2id('left_subgoal_'+str(idx+1))] = (
                    subgoal[-3:]
            )
        else :
            raise NotImplementedError
        
    def get_obstacle_positions(self):
        if self.obstacle_exist:
            obstacle_idx = np.where(['obstacle' in name for name in self.model.body_names])[0]
            
            obstacle_positions = []
            for idx in obstacle_idx:
                obstacle_name = self.model.body_names[idx]
                obstacle_positions.append(self.data.get_body_xpos(obstacle_name))

            return obstacle_positions
            
        else :
            return None

    def _get_placebox_sites(self):
        placebox_sites = []
        for idx, site_name in enumerate(self.model.site_names):
            if 'box_edge' in site_name:
                placebox_sites.append(self.data.site_xpos[self.model.site_name2id(site_name)].copy())
            
        return placebox_sites

    def _get_placebox_center(self):
        place_box_sites = self._get_placebox_sites()
        if len(place_box_sites) ==0:
            return None
        else :
            return np.mean(np.stack([site for site in place_box_sites], axis =0), axis=0) #[num site, dim] -> [dim]

    # <site name="box_edge_1" pos="0.2 0.08 0.077" size="0.01" rgba="1 0.8 0 1"/>
    #     <site name="box_edge_2" pos="0.2 -0.08 0.077" size="0.01" rgba="1 0.8 0 1"/>
    #     <site name="box_edge_3" pos="-0.2 0.08 0.077" size="0.01" rgba="1 0.8 0 1"/>
    #     <site name="box_edge_4" pos="-0.2 -0.08 0.077" size="0.01" rgba="1 0.8 0 1"/>
    # def _set_finalgoal_marker(self, finalgoal):
    #    """
    #    This should be use ONLY for visualization. Use self._state_goal for
    #    logging, learning, etc.
    #    """
    #    self.data.site_xpos[self.model.site_name2id('finalgoal')] = (
    #        finalgoal[-3:]
    #    )

#single ur3 만들고 그 하위로 reachenv만들수도
class DSCHOSingleUR3ReachEnv(DSCHODualUR3Env):
    
    # Sholud be used with URScriptWrapper
    
    def __init__(self,
                sparse_reward = False,
                reduced_observation = False,
                trigonometry_observation = True, 
                ur3_random_init=False,
                goal_random_init = True, 
                full_state_goal = True,
                reward_by_ee = False, 
                automatically_set_spaces=False,
                fixed_goal_qvel = True, 
                reward_success_criterion='ee_pos',
                distance_threshold = 0.05,
                initMode='vertical',
                xml_filename = 'dscho_dual_ur3.xml',
                which_hand='right', 
                so3_constraint ='vertical_side',
                *args,
                **kwargs
                ):
        self.save_init_params(locals())

        self.full_state_goal = full_state_goal
        self.reduced_observation = reduced_observation
        self.trigonometry_observation = trigonometry_observation
        self.fixed_goal_qvel = fixed_goal_qvel
        # for LEAP(S=G) -> modified for S!=G (20200930)
        # assert (reduced_observation and not full_state_goal) or (not reduced_observation and full_state_goal)
        assert (reduced_observation and not trigonometry_observation) or (not reduced_observation and trigonometry_observation)
        self.reward_by_ee = reward_by_ee
        #self.ur3_random_init = ur3_random_init
        self.goal_random_init = goal_random_init
        self.curr_path_length = 0

        if so3_constraint == 'no' or so3_constraint is None:
            self.so3_constraint = NoConstraint()
        elif so3_constraint == 'upright':
            self.so3_constraint = UprightConstraint()
        else:
            self.so3_constraint = SO3Constraint(SO3=so3_constraint)
        
        self.sparse_reward = sparse_reward
        self.distance_threshold = distance_threshold
        #self.initMode = initMode
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
        
        if not self.reduced_observation:
            if self.trigonometry_observation:
                self.obs_nqpos = self.ur3_nqpos*2
            else :
                self.obs_nqpos = self.ur3_nqpos
        else :
            self.obs_nqpos = 3 # ee_pos

        
        
        
        qpos_low = -np.ones(int(self.ur3_nqpos))*2*np.pi
        qpos_high = np.ones(int(self.ur3_nqpos))*2*np.pi
        qvel_low = -np.ones(int(self.ur3_nqpos))*0.01
        qvel_high = np.ones(int(self.ur3_nqpos))*0.01
        
        self.goal_qpos_space = Box(low=qpos_low, high=qpos_high, dtype=np.float32)
        self.goal_qvel_space = Box(low=qvel_low, high=qvel_high, dtype=np.float32)
        if 'obstacle_v1' in self.xml_filename :
            from gym_custom.envs.custom.dscho_wall import Wall3D
            collision_buffer = 0.1
            self.walls = [
                Wall3D(0.0, -0.4, 0.82, 0.4, 0.015, 0.1, collision_buffer),
            ]
            if self.which_hand =='right':
                ee_low = np.array([-0.2, -0.65, 0.75])
                ee_high = np.array([0.55, -0.2, 1.1])
            
            elif self.which_hand =='left':
                ee_low = np.array([-0.55, -0.65, 0.75])
                ee_high = np.array([0.2, -0.2, 1.1])
            self.goal_ee_pos_space=Box(low = ee_low, high = ee_high, dtype=np.float32)
        elif 'obstacle_v2' in self.xml_filename :
            from gym_custom.envs.custom.dscho_wall import Wall3D
            collision_buffer = 0.1
            self.walls = [
                Wall3D(0.35, -0.4, 0.87, 0.05, 0.05, 0.15, collision_buffer),
                Wall3D(-0.35, -0.4, 0.87, 0.05, 0.05, 0.15, collision_buffer),
            ]
            if self.which_hand =='right':
                ee_low = np.array([-0.2, -0.65, 0.75])
                ee_high = np.array([0.55, -0.2, 1.1])
            
            elif self.which_hand =='left':
                ee_low = np.array([-0.55, -0.65, 0.75])
                ee_high = np.array([0.2, -0.2, 1.1])
            self.goal_ee_pos_space=Box(low = ee_low, high = ee_high, dtype=np.float32)
        elif 'obstacle_v4' in self.xml_filename :
            from gym_custom.envs.custom.dscho_wall import Wall3D
            collision_buffer = 0.1
            self.walls = [
                Wall3D(0.35, -0.35, 0.82, 0.03, 0.3, 0.1, collision_buffer),
                Wall3D(-0.35, -0.35, 0.82, 0.03, 0.3, 0.1, collision_buffer),
            ]
            if self.which_hand =='right':
                ee_low = np.array([-0.2, -0.6, 0.75])
                ee_high = np.array([0.6, -0.15, 1.2])
            
            elif self.which_hand =='left':
                ee_low = np.array([-0.6, -0.6, 0.75])
                ee_high = np.array([0.2, -0.15, 1.2])
            self.goal_ee_pos_space=Box(low = ee_low, high = ee_high, dtype=np.float32)
        
        elif 'obstacle_v5' in self.xml_filename :
            from gym_custom.envs.custom.dscho_wall import Wall3D
            collision_buffer = 0.1
            self.walls = [
                Wall3D(0.35, -0.55, 0.97, 0.03, 0.2, 0.25, collision_buffer),
                Wall3D(-0.35, -0.55, 0.97, 0.03, 0.2, 0.25, collision_buffer),
            ]
            if self.which_hand =='right':
                ee_low = np.array([-0.2, -0.6, 0.75])
                ee_high = np.array([0.55, -0.1, 1.1])
            
            elif self.which_hand =='left':
                ee_low = np.array([-0.55, -0.6, 0.75])
                ee_high = np.array([0.2, -0.1, 1.1])
            self.goal_ee_pos_space=Box(low = ee_low, high = ee_high, dtype=np.float32)
        
        else : # for obstacle v1~v3 which was not proper for current setting
            raise NotImplementedError
            print('Warning! : It is for v0, v3')
            self.walls = None
            if self.which_hand =='right': 
                ee_low = np.array([-0.2, -0.6, 0.65])
                ee_high = np.array([0.5, -0.18, 0.9])
            
            elif self.which_hand =='left':
                ee_low = np.array([-0.5, -0.6, 0.65])
                ee_high = np.array([0.2, -0.18, 0.9])
            self.goal_ee_pos_space=Box(low = ee_low, high = ee_high, dtype=np.float32)


        self._state_goal = self.sample_goal(self.full_state_goal)
        
        # self._state_subgoals = []
        # for i in range(2):
        #     sitename = 'subgoal'
        #     self._state_subgoals.append(self.data.site_xpos[self.model.site_name2id(sitename+'_'+str(i+1))].copy())
            
            
        # self._state_finalgoal = self.data.site_xpos[self.model.site_name2id('finalgoal')].copy()
        # self._state_finalgoal = self._state_goal.copy()

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
                if self.walls is None:
                    if (p <= self.goal_ee_pos_space.high).all() and (p >= self.goal_ee_pos_space.low).all():
                        break
                else :
                    if (p <= self.goal_ee_pos_space.high).all() and (p >= self.goal_ee_pos_space.low).all() and self._check_no_wall_collision(p):
                        break

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
            if self.walls is None:
                goal_ee_pos = np.random.uniform(
                    self.goal_ee_pos_space.low,
                    self.goal_ee_pos_space.high,
                    size=(self.goal_ee_pos_space.low.size),
                )
                goal = goal_ee_pos
            else :
                while True:
                    goal_ee_pos = np.random.uniform(
                        self.goal_ee_pos_space.low,
                        self.goal_ee_pos_space.high,
                        size=(self.goal_ee_pos_space.low.size),
                    )
                    goal = goal_ee_pos
                    if self._check_no_wall_collision(goal_ee_pos):
                        break
                    
            
        return goal

    # def _check_no_wall_collision(self, position):
    #     for wall in self.walls:
    #         if wall.contains_point(position):
    #             return False
    #     return True
    def _check_no_wall_collision(self, position):
        if position.ndim ==1:
            for wall in self.walls:
                if wall.contains_point(position):
                    return False
            return True
        elif position.ndim==2:
            collision_for_each_wall = np.array([False]*position.shape[0]) #[False, False, ..] #[bs]
            for wall in self.walls:
                collision_for_each_wall = np.logical_or(collision_for_each_wall, wall.contains_points(position))
            return np.logical_not(collision_for_each_wall) #[bs]
            

    def reset_model(self):
        # original_ob = super().reset_model() # init qpos,qvel set_state and get_obs
        # assert not self.random_init, 'it is only for reaching env'
        # 이렇게 하면 obs에 샘플한 state_goal이 반영이 안됨. -> 밑에서 별도 설정!
        
        self._state_goal = self.sample_goal(full_state_goal = self.full_state_goal)
        qpos = self._get_init_qpos() + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
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

        obs = np.concatenate([qpos, qvel, ee_pos])
        
        if self.reduced_observation:
            obs = ee_pos
        else:
            pass
        
        if self.full_state_goal:
            achieved_goal = obs
        else :
            achieved_goal = ee_pos

        return {
            'observation' : obs.copy(),
            'achieved_goal' : achieved_goal.copy(),
            'desired_goal' : self._state_goal.copy(), 
        }    

        
    
    def step(self, action):
        # actions of remaning arm will be garbage
        # gripper [-1,1] should be mod to
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
                reward = 1
            else :
                reward = 0 
        else :
            reward = -placingDist

        return reward



class DSCHODualUR3PickAndPlaceEnvObstacle(DSCHODualUR3PickAndPlaceEnv):
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        # xml_filename = 'dscho_dual_ur3_obstacle.xml'
        super().__init__( *args, **kwargs)
    
class DSCHOSingleUR3ReachEnvObstacle(DSCHOSingleUR3ReachEnv):
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        # xml_filename = 'dscho_dual_ur3_obstacle.xml'
        super().__init__( *args, **kwargs)
        

