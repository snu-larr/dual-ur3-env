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
import pickle
import joblib
import time
import mujoco_py
from gym_custom.envs.robotics import rotations #, robot_env, utils
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


class DummyWrapper():
    
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)


class MocapSingleWrapper(URScriptWrapper_DualUR3):
    
    def __init__(self, 
                env, 
                # q_control_type, 
                # g_control_type, 
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
        # self.q_control_type = q_control_type
        # self.g_control_type = g_control_type
        
        self.ee_xyz_pos_dim = 3
        self.action_downscale = action_downscale
        self.gripper_force_scale = gripper_force_scale
        # self.null_obj_func = SO3Constraint(SO3=so3_constraint)

        self.multi_step = multi_step
        print('Currently, multi step in SingleWrapper has no meaning!')
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
    
        
        self.ur3_act_dim = 3 #self.wrapper_right.ndof
        self.gripper_act_dim = self.wrapper_right.ngripperdof
        assert self.ur3_act_dim==3
        assert self.gripper_act_dim==1
        self.action_space = Box(low=act_low, high=act_high, dtype=np.float32)
        print(colorize('WARNING : CHECK action space boundary : {}'.format(self.action_space), 'green', bold=True))
        
    def reset(self, **kwargs):
        # return self.env.reset(**kwargs)
        return super().reset(**kwargs)

    # Wrapper 에서 multistep을 해야 제일 lowlevel에선 매 timestep마다 command 새로계산함으로써 정확도 증가되는데, 내부적으로 IK sol 쓰다보니 이런구조는 아니라 정확도 살짝 떨어질수도
    def step(self, action):
        action = action.copy()
        # print('action in MocapSingleWrapper : ', action)
        # down scale [-1,1] to [-0.005, 0.005]
        
        # action = copy.deepcopy(action) # 통째로 *downscale이면 문제 없는데 index로 접근할땐 array가 mutable이라 copy해줘야함, but 매 스텝마다 action을 새로 뽑는상황이라면 굳이 이렇게 안해도 상관없음. 똑같은 action으로 계속 step밟을 때나 문제가 되는거지
        action[:self.ee_xyz_pos_dim] = np.clip(action[:self.ee_xyz_pos_dim], -1, 1)
        action[:self.ee_xyz_pos_dim] = self.action_downscale*action[:self.ee_xyz_pos_dim]
        ur3_act = action[:self.ee_xyz_pos_dim] # delta xyz pos of ee
        
        if self.gripper_action:
            gripper_act = self.gripper_force_scale*action[-self.gripper_act_dim:]
            # print('gripper act : ', gripper_act)
        else :
            gripper_act = np.zeros(self.gripper_act_dim)
        
        
        if self.env.which_hand=='right':            
            right_ur3_action = ur3_act            
            right_gripper_action = gripper_act
            
            left_ur3_action = np.zeros(self.ur3_act_dim)
            left_gripper_action = np.zeros(self.gripper_act_dim)
            
        elif self.env.which_hand=='left':
            left_ur3_action = ur3_act
            left_gripper_action = gripper_act

            right_ur3_action = np.zeros(self.ur3_act_dim)
            right_gripper_action = np.zeros(self.gripper_act_dim)
            
        
        command = np.concatenate([right_ur3_action, right_gripper_action, left_ur3_action, left_gripper_action], axis =-1)
        # for _ in range(self.multi_step-1):
        #     # print('action in MocapSingleWrapper command : ', command)
        #     self.env.step(command) # obs is dict)
        return self.env.step(command)

    def __getattr__(self, name):
        return getattr(self.env, name)

class MocapDualWrapper(URScriptWrapper_DualUR3):
    
    def __init__(self,
                env, 
                # q_control_type, 
                # g_control_type, 
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
        # self.q_control_type = q_control_type
        # self.g_control_type = g_control_type
        
        self.ee_xyz_pos_dim = 3
        self.action_downscale = action_downscale
        self.gripper_force_scale = gripper_force_scale
        # self.null_obj_func = SO3Constraint(SO3=so3_constraint)

        self.multi_step = multi_step
        print('Currently, multi step in DualWrapper has no meaning!')
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
    
        
        self.ur3_act_dim = 3 #self.wrapper_right.ndof
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
        action = action.copy() # NOTE : VERY IMPORTANT!!! If not copy, action will be changed outside!
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
        
        
        right_ur3_action = right_ur3_act
        left_ur3_action = left_ur3_act
        
        right_gripper_action = right_gripper_act
        left_gripper_action = left_gripper_act
        
        command = np.concatenate([right_ur3_action, right_gripper_action, left_ur3_action, left_gripper_action], axis =-1)
        # for _ in range(self.multi_step-1):
        #     self.env.step(command)
        return self.env.step(command)
    
    def __getattr__(self, name):
        return getattr(self.env, name)


class DualUR3Gripper(DualUR3Env):
    def __init__(self, flat_gripper = False, *args, **kwargs):
        
        self.flat_gripper = flat_gripper
        self.flat_gripper_nqpos = 0 # per gripper joint pos dim
        self.flat_gripper_nqvel = 0 # per gripper joint vel dim
        self.flat_gripper_nact = 0 # per gripper action dim
        if flat_gripper :
            self.flat_gripper_nqpos = 2 # per gripper joint pos dim
            self.flat_gripper_nqvel = 2 # per gripper joint vel dim
            self.flat_gripper_nact = 2 # per gripper action dim
        
        super().__init__(*args, **kwargs)
        
        if flat_gripper:
            assert 'flat_gripper' in self.xml_filename


    def _check_model_parameter_dimensions(self):
        '''overridable method'''
        assert 2*self.ur3_nqpos + 2*self.gripper_nqpos + sum(self.objects_nqpos) + 2*self.flat_gripper_nqpos == self.model.nq, 'Number of qpos elements mismatch'
        assert 2*self.ur3_nqvel + 2*self.gripper_nqvel + sum(self.objects_nqvel) + 2*self.flat_gripper_nqvel == self.model.nv, 'Number of qvel elements mismatch'
        assert 2*self.ur3_nact + 2*self.gripper_nact == self.model.nu, 'Number of action elements mismatch'

    def _set_init_qpos(self):
        '''overridable method'''
        # Initial position for UR3
        # dscho mod
        if self.ur3_random_init:
            pass
        if self.initMode is None :
            self.init_qpos[0:self.ur3_nqpos] = \
                np.array([-90.0, -90.0, -90.0, -90.0, -135.0, 180.0])*np.pi/180.0 # right arm
            self.init_qpos[self.ur3_nqpos+self.gripper_nqpos+self.flat_gripper_nqpos:2*self.ur3_nqpos+self.gripper_nqpos+self.flat_gripper_nqpos] = \
                np.array([90.0, -90.0, 90.0, -90.0, 135.0, -180.0])*np.pi/180.0 # left arm
        elif self.initMode =='vertical':
            self.init_qpos[0:self.ur3_nqpos] = \
                np.array([-0.32213427, -1.81002217, -1.87559869, -1.72603011, -1.79932887,  1.82011286]) # right arm
            self.init_qpos[self.ur3_nqpos+self.gripper_nqpos+self.flat_gripper_nqpos:2*self.ur3_nqpos+self.gripper_nqpos+self.flat_gripper_nqpos] = \
                np.array([ 0.3209594,  -1.33282653,  1.87653391, -1.41410399, 1.79674747, -1.81847637])# left arm
        elif self.initMode =='horizontal':
            # horizontal init
            self.init_qpos[0:self.ur3_nqpos] = \
                np.array([ 1.82496873, -1.78037016,  1.86075417,  4.40278818,  5.47660708, -2.8826006]) # right arm
            self.init_qpos[self.ur3_nqpos+self.gripper_nqpos+self.flat_gripper_nqpos:2*self.ur3_nqpos+self.gripper_nqpos+self.flat_gripper_nqpos] = \
                np.array([-1.85786483, -1.3540493,  -1.89351501, -1.18579177,  0.82976128, -0.50789828])# left arm
        else :
            raise NotImplementedError

    def _get_ur3_qpos(self):
        return np.concatenate([self.sim.data.qpos[0:self.ur3_nqpos], 
            self.sim.data.qpos[self.ur3_nqpos+self.gripper_nqpos+self.flat_gripper_nqpos:2*self.ur3_nqpos+self.gripper_nqpos+self.flat_gripper_nqpos]]).ravel()

    def _get_gripper_qpos(self):
        return np.concatenate([self.sim.data.qpos[self.ur3_nqpos:self.ur3_nqpos+self.gripper_nqpos], 
            self.sim.data.qpos[2*self.ur3_nqpos+self.gripper_nqpos:2*self.ur3_nqpos+2*self.gripper_nqpos]]).ravel()
    
    def _get_flat_gripper_qpos(self):
        return np.concatenate([self.sim.data.qpos[self.ur3_nqpos+self.gripper_nqpos:self.ur3_nqpos+self.gripper_nqpos+self.flat_gripper_nqpos], 
            self.sim.data.qpos[2*self.ur3_nqpos+2*self.gripper_nqpos+self.flat_gripper_nqpos:2*self.ur3_nqpos+2*self.gripper_nqpos+2*+self.flat_gripper_nqpos]]).ravel()

    def _get_ur3_qvel(self):
        return np.concatenate([self.sim.data.qvel[0:self.ur3_nqvel], 
            self.sim.data.qvel[self.ur3_nqvel+self.gripper_nqvel+self.flat_gripper_nqvel:2*self.ur3_nqvel+self.gripper_nqvel+self.flat_gripper_nqvel]]).ravel()

    def _get_gripper_qvel(self):
        return np.concatenate([self.sim.data.qvel[self.ur3_nqvel:self.ur3_nqvel+self.gripper_nqvel], 
            self.sim.data.qvel[2*self.ur3_nqvel+self.gripper_nqvel:2*self.ur3_nqvel+2*self.gripper_nqvel]]).ravel()

    def _get_flat_gripper_qvel(self):
        return np.concatenate([self.sim.data.qvel[self.ur3_nqvel+self.gripper_nqvel:self.ur3_nqvel+self.gripper_nqvel+self.flat_gripper_nqvel], 
            self.sim.data.qvel[2*self.ur3_nqvel+2*self.gripper_nqvel+self.flat_gripper_nqvel:2*self.ur3_nqvel+2*self.gripper_nqvel+2*+self.flat_gripper_nqvel]]).ravel()

    def _get_ur3_bias(self):
        return np.concatenate([self.sim.data.qfrc_bias[0:self.ur3_nqvel], 
            self.sim.data.qfrc_bias[self.ur3_nqvel+self.gripper_nqvel+self.flat_gripper_nqvel:2*self.ur3_nqvel+self.gripper_nqvel+self.flat_gripper_nqvel]]).ravel()

    def _get_gripper_bias(self):
        return np.concatenate([self.sim.data.qfrc_bias[self.ur3_nqvel:self.ur3_nqvel+self.gripper_nqvel], 
            self.sim.data.qfrc_bias[2*self.ur3_nqvel+self.gripper_nqvel:2*self.ur3_nqvel+2*self.gripper_nqvel]]).ravel()

    def _get_flat_gripper_bias(self):
        raise NotImplementedError()

    def _get_ur3_constraint(self):
        return np.concatenate([self.sim.data.qfrc_constraint[0:self.ur3_nqvel], 
            self.sim.data.qfrc_constraint[self.ur3_nqvel+self.gripper_nqvel+self.flat_gripper_nqvel:2*self.ur3_nqvel+self.gripper_nqvel+self.flat_gripper_nqvel]]).ravel()

    def _get_ur3_actuator(self):
        return np.concatenate([self.sim.data.qfrc_actuator[0:self.ur3_nqvel], 
            self.sim.data.qfrc_actuator[self.ur3_nqvel+self.gripper_nqvel+self.flat_gripper_nqvel:2*self.ur3_nqvel+self.gripper_nqvel+self.flat_gripper_nqvel]]).ravel()

    def get_obs_dict(self):
        '''overridable method'''
        return {'right': {
                'qpos': self._get_ur3_qpos()[:self.ur3_nqpos],
                'qvel': self._get_ur3_qvel()[:self.ur3_nqvel],
                'gripperpos': self._get_gripper_qpos()[:self.gripper_nqpos],
                'grippervel': self._get_gripper_qvel()[:self.gripper_nqvel],
                'flat_grpperpos' : self._get_flat_gripper_qpos()[:self.flat_gripper_nqpos],
                'flat_grppervel' : self._get_flat_gripper_qvel()[:self.flat_gripper_nqvel],
            },
            'left': {
                'qpos': self._get_ur3_qpos()[self.ur3_nqpos:],
                'qvel': self._get_ur3_qvel()[self.ur3_nqvel:],
                'gripperpos': self._get_gripper_qpos()[self.gripper_nqpos:],
                'grippervel':self._get_gripper_qvel()[self.gripper_nqvel:],
                'flat_grpperpos' : self._get_flat_gripper_qpos()[self.flat_gripper_nqpos:],
                'flat_grppervel' : self._get_flat_gripper_qvel()[self.flat_gripper_nqvel:],
            }
        }
        

    




class DSCHODualUR3MocapEnv(DualUR3Gripper):
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        self.init_qpos_candidates = {}
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
        
        # for push or reach (closed gripper)
        if self.task in ['push', 'reach']:
            default_gripper_right_qpos = np.array([[0.70005217, 0.01419325, 0.0405478, 0.0134475, 0.74225534, 0.70005207, 0.01402114, 0.04054553, 0.01344841, 0.74224361]])
            default_gripper_left_qpos = np.array([[0.70005217, 0.01419325, 0.0405478, 0.0134475, 0.74225534, 0.70005207, 0.01402114, 0.04054553, 0.01344841, 0.74224361]])
            # add default qpos configuration        
            self.init_qpos_candidates['gripper_q_right_des'] =default_gripper_right_qpos
            self.init_qpos_candidates['gripper_q_left_des'] =default_gripper_left_qpos

        

        super().__init__(*args, **kwargs)
        # super().__init__(xml_filename, initMode, automatically_set_spaces=automatically_set_spaces)
        
        

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
            init_qpos[self.ur3_nqpos+self.gripper_nqpos+self.flat_gripper_nqpos:2*self.ur3_nqpos+self.gripper_nqpos+self.flat_gripper_nqpos] = q_left_des_candidates[left_idx]

        else :
            # Currently for dual env test with 0th index init qpos
            q_right_des_candidates = self.init_qpos_candidates['q_right_des'] # [num_candidate, qpos dim]
            q_left_des_candidates = self.init_qpos_candidates['q_left_des']
            
            right_idx = 0
            left_idx = 0
            init_qpos[0:self.ur3_nqpos] = q_right_des_candidates[right_idx]
            init_qpos[self.ur3_nqpos+self.gripper_nqpos+self.flat_gripper_nqpos:2*self.ur3_nqpos+self.gripper_nqpos+self.flat_gripper_nqpos] = q_left_des_candidates[left_idx]

            if self.flat_gripper:
                init_qpos[self.ur3_nqpos+self.gripper_nqpos:self.ur3_nqpos+self.gripper_nqpos+self.flat_gripper_nqpos] = np.array([0.0, 0.0])
                init_qpos[2*self.ur3_nqpos+2*self.gripper_nqpos+self.flat_gripper_nqpos:2*self.ur3_nqpos+2*self.gripper_nqpos+2*self.flat_gripper_nqpos] = np.array([0.0, 0.0])

        if self.task in ['push', 'reach']: # initially, close gripper
            gripper_q_right_des_candidates = self.init_qpos_candidates['gripper_q_right_des']
            gripper_q_left_des_candidates = self.init_qpos_candidates['gripper_q_left_des']
            right_idx = 0
            left_idx = 0
            init_qpos[self.ur3_nqpos:self.ur3_nqpos + self.gripper_nqpos] = gripper_q_right_des_candidates[right_idx]
            init_qpos[2*self.ur3_nqpos+self.gripper_nqpos+self.flat_gripper_nqpos:2*self.ur3_nqpos+2*self.gripper_nqpos+self.flat_gripper_nqpos] = gripper_q_left_des_candidates[left_idx]
            if self.flat_gripper:
                raise NotImplementedError('think closed flat gripper !')

        return init_qpos

    def _mujocoenv_init(self):
        '''overridable method'''
        MujocoEnv.__init__(self, self.mujoco_xml_full_path, self.mujocoenv_frame_skip, automatically_set_spaces = self.automatically_set_spaces)
        
        if not self.automatically_set_spaces:
            self._env_setup(self._get_init_qpos())
            # self._set_action_space()
            # self.do_simulation(self.action_space.sample(), self.frame_skip)

    def _check_model_parameter_dimensions(self):
        '''overridable method'''
        if 'flat_gripper' in self.xml_filename:
            assert 2*self.ur3_nqpos + 2*self.gripper_nqpos + sum(self.objects_nqpos) + 2*self.flat_gripper_nqpos == self.model.nq, 'Number of qpos elements mismatch'
            assert 2*self.ur3_nqvel + 2*self.gripper_nqvel + sum(self.objects_nqvel) + 2*self.flat_gripper_nqvel == self.model.nv, 'Number of qvel elements mismatch'
            # assert 2*self.ur3_nact + 2*self.gripper_nact == self.model.nu, 'Number of action elements mismatch'
            assert 2*self.flat_gripper_nact == self.model.nu, 'Number of action elements mismatch'
        else:
            assert 2*self.ur3_nqpos + 2*self.gripper_nqpos + sum(self.objects_nqpos) == self.model.nq, 'Number of qpos elements mismatch'
            assert 2*self.ur3_nqvel + 2*self.gripper_nqvel + sum(self.objects_nqvel) == self.model.nv, 'Number of qvel elements mismatch'
            # assert 2*self.ur3_nact + 2*self.gripper_nact == self.model.nu, 'Number of action elements mismatch'
            assert 2*self.gripper_nact == self.model.nu, 'Number of action elements mismatch'
    
    # NOTE : Should init the mocap?
    def _env_setup(self, initial_qpos):
        # print('debug before set state : right ee body pos  : {} left ee body pos : {}'.format(self.data.get_body_xpos('right_gripper:hand'), self.data.get_body_xpos('left_gripper:hand')))
        
        self.set_state(initial_qpos, self.init_qvel)
            # self.sim.data.set_joint_qpos(name, value)
        
        # print('debug before reset modcap weld : right ee body pos  : {} left ee body pos : {}'.format(self.data.get_body_xpos('right_gripper:hand'), self.data.get_body_xpos('left_gripper:hand')))
        self.reset_mocap_welds(self.sim)
        
        # print('debug before sim.forward reset modcap weld : right ee body pos  : {} left ee body pos : {}'.format(self.data.get_body_xpos('right_gripper:hand'), self.data.get_body_xpos('left_gripper:hand')))
        self.sim.forward()
        
        # right_gripper_target = self.get_endeff_pos('right')
        # left_gripper_target = self.get_endeff_pos('left')
        right_gripper_target = np.array([0.2, -0.4, 0.85])
        left_gripper_target = np.array([-0.2, -0.4, 0.85])
        
        # print('debug after sim.forward: right ee body pos  : {} left ee body pos : {}'.format(self.data.get_body_xpos('right_gripper:hand'), self.data.get_body_xpos('left_gripper:hand')))

        # Move end effector into position.
        # gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('right_gripper:hand')
        
        # right_gripper_rotation = np.array([1., 0., 1., 0.])
        # left_gripper_rotation = np.array([1., 0., 1., 0.])
        right_gripper_rotation = np.array([0., 1., 0., 0.])
        left_gripper_rotation = np.array([0., 1., 0., 0.])


        self.sim.data.set_mocap_pos('right_mocap', right_gripper_target)
        self.sim.data.set_mocap_quat('right_mocap', right_gripper_rotation)
        self.sim.data.set_mocap_pos('left_mocap', left_gripper_target)
        self.sim.data.set_mocap_quat('left_mocap', left_gripper_rotation)
        for _ in range(100):
            self.sim.step()
            # print('debug after sim.step: right ee body pos  : {} left ee body pos : {}'.format(self.data.get_body_xpos('right_gripper:hand'), self.data.get_body_xpos('left_gripper:hand')))
        # Extract information for sampling goals.
        # self.initial_gripper_xpos = self.sim.data.get_site_xpos('right_gripper:hand').copy()
        # if self.has_object:
        #     self.height_offset = self.sim.data.get_site_xpos('obj')[2]

    def _set_action(self, action):
        # print('action in set_action : ', action)
        # Assume already scaled action
        assert action.shape == (8,) # should use wrapper
        action = action.copy()  # ensure that we don't change the action outside of this scope
        right_pos_ctrl, right_gripper_ctrl, left_pos_ctrl, left_gripper_ctrl = action[:3], action[3], action[4:7], action[7]

        # right_pos_ctrl *= 0.05  # limit maximum change in position
        # right_rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        right_rot_ctrl = [0., 1., 0., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        right_gripper_ctrl = np.array([right_gripper_ctrl, right_gripper_ctrl])

        # left_pos_ctrl *= 0.05  # limit maximum change in position
        # left_rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        left_rot_ctrl = [0., 1., 0., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        left_gripper_ctrl = np.array([left_gripper_ctrl, left_gripper_ctrl])
        

        assert right_gripper_ctrl.shape == (2,) and left_gripper_ctrl.shape == (2,)
        if self.block_gripper:
            right_gripper_ctrl = np.zeros_like(right_gripper_ctrl)
            left_gripper_ctrl = np.zeros_like(left_gripper_ctrl)
        
        action = np.concatenate([right_pos_ctrl, right_rot_ctrl, left_pos_ctrl, left_rot_ctrl, right_gripper_ctrl, left_gripper_ctrl])

        # Apply action to simulation.
        self.ctrl_set_action(self.sim, action)
        self.mocap_set_action(self.sim, action)


    def ctrl_set_action(self, sim, action):
        """For torque actuators it copies the action into mujoco ctrl field.
        For position actuators it sets the target relative to the current qpos.
        """
        if sim.model.nmocap > 0:
            _, action = np.split(action, (sim.model.nmocap * 7, )) # only gripper ctrl
        if sim.data.ctrl is not None:
            for i in range(action.shape[0]):
                if sim.model.actuator_biastype[i] == 0:
                    sim.data.ctrl[i] = action[i]
                else:
                    idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]]
                    sim.data.ctrl[i] = sim.data.qpos[idx] + action[i]


    def mocap_set_action(self,sim, action):
        """The action controls the robot using mocaps. Specifically, bodies
        on the robot (for example the gripper wrist) is controlled with
        mocap bodies. In this case the action is the desired difference
        in position and orientation (quaternion), in world coordinates,
        of the of the target body. The mocap is positioned relative to
        the target body according to the delta, and the MuJoCo equality
        constraint optimizer tries to center the welded body on the mocap.
        """
        if sim.model.nmocap > 0:
            action, _ = np.split(action, (sim.model.nmocap * 7, ))
            action = action.reshape(sim.model.nmocap, 7)

            pos_delta = action[:, :3]
            quat_delta = action[:, 3:]
            
            # print('pos_delta : {} quat_delta : {}'.format(pos_delta, quat_delta))
            self.reset_mocap2body_xpos(sim)
            sim.data.mocap_pos[:] = sim.data.mocap_pos + pos_delta
            sim.data.mocap_quat[:] = sim.data.mocap_quat + quat_delta


    def reset_mocap_welds(self, sim):
        """Resets the mocap welds that we use for actuation.
        """
        if sim.model.nmocap > 0 and sim.model.eq_data is not None:
            for i in range(sim.model.eq_data.shape[0]):
                if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.])
        sim.forward()


    def reset_mocap2body_xpos(self, sim): #called at init
        """Resets the position and orientation of the mocap bodies to the same
        values as the bodies they're welded to.
        """

        if (sim.model.eq_type is None or
            sim.model.eq_obj1id is None or
            sim.model.eq_obj2id is None):
            return
        for eq_type, obj1_id, obj2_id in zip(sim.model.eq_type,
                                            sim.model.eq_obj1id,
                                            sim.model.eq_obj2id):
            if eq_type != mujoco_py.const.EQ_WELD:
                continue
            
            

            mocap_id = sim.model.body_mocapid[obj1_id]
            if mocap_id != -1:
                # obj1 is the mocap, obj2 is the welded body
                body_idx = obj2_id
            else:
                # dscho add for additional weld constraint (Not for mocap)
                if 'mocap' not in sim.model.body_id2name(obj2_id):
                    continue


                # obj2 is the mocap, obj1 is the welded body
                mocap_id = sim.model.body_mocapid[obj2_id]
                body_idx = obj1_id

            assert (mocap_id != -1)
            sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
            sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]

    def step(self, action):
        action = action.copy()
        raise NotImplementedError('Currently, Not implemented for dual arm. We just overrided it in sigle arm env')
        self._set_action(action)
        self.sim.step()
        # self._step_callback() # gripper close related
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info
        


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
class DSCHOSingleUR3GoalMocapEnv(DSCHODualUR3MocapEnv):
    
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
        self.left_get_away_qpos = np.concatenate([np.array([-90.0, -90.0, 90.0, -90.0, 135.0, 0.0])*np.pi/180.0, np.zeros(self.gripper_nqpos), np.zeros(self.flat_gripper_nqpos)]) # it was self.gripper_nqpos
        self.right_get_away_qpos = np.concatenate([np.array([-90.0, -90.0, 90.0, -90.0, 135.0, 0.0])*np.pi/180.0, np.zeros(self.gripper_nqpos), np.zeros(self.flat_gripper_nqpos)])
        
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
            # raise NotImplementedError
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
    
    def _set_action_space(self):
        act_low = np.array([-1,-1,-1,-1, -1,-1,-1,-1])
        act_high = np.array([1,1,1,1, 1,1,1,1])
        self.action_space = Box(low=act_low, high=act_high, dtype=np.float32)

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
        # original_ob = super().reset_model() # init qpos,qvel set_state and get_obs
        # assert not self.random_init, 'it is only for reaching env'
        # 이렇게 하면 obs에 샘플한 state_goal이 반영이 안됨. -> 밑에서 별도 설정!
        
        # self._state_goal = self.sample_goal(full_state_goal = self.full_state_goal)
        qpos = self._get_init_qpos() #+ self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        
        self.set_state(qpos, qvel)
        # NOTE : check mocap pos quat

        # randomly reset the initial position of an object
        if self.has_object:
            if self.task in ['pickandplace', 'push']:
                object_xpos = np.random.uniform(
                                self.goal_obj_pos_space.low,
                                self.goal_obj_pos_space.high,
                                size=(self.goal_obj_pos_space.low.size),
                            )[:2]
                object_pos = np.concatenate([object_xpos, np.array([self.goal_obj_pos_space.low[-1]])], axis=-1)
                ee_pos = self.get_endeff_pos(arm=self.which_hand)
                
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
        self.curr_path_length = 0
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
            # 실제 로봇실험에서 불가능한것 : gripper_state, gripper_vel, object_velp, object_velr, object_velp
            obj_rot = rotations.mat2euler(self.sim.data.get_site_xmat('objSite'))
            # velocities
            dt = self.sim.nsubsteps * self.sim.model.opt.timestep # same as self.dt
            obj_velp = self.sim.data.get_site_xvelp('objSite') * dt
            obj_velr = self.sim.data.get_site_xvelr('objSite') * dt
            obj_rel_pos = obj_pos - ee_pos
            if self.flat_gripper:                
                gripper_state = np.array([self.sim.data.get_joint_qpos(self.which_hand+'_gripper:r_gripper_finger_joint'),
                                        self.sim.data.get_joint_qpos(self.which_hand+'_gripper:l_gripper_finger_joint'),
                                        ])
                gripper_vel = np.array([self.sim.data.get_joint_qvel(self.which_hand+'_gripper:r_gripper_finger_joint'),
                                        self.sim.data.get_joint_qvel(self.which_hand+'_gripper:l_gripper_finger_joint'),
                                        ])*dt
                
            else:
                right_slide_joint_qpos = self.sim.data.get_joint_qpos(self.which_hand+'_gripper:right_fingertip:slide:control')
                left_slide_joint_qpos = self.sim.data.get_joint_qpos(self.which_hand+'_gripper:left_fingertip:slide:control')
                right_slide_joint_qvel = self.sim.data.get_joint_qvel(self.which_hand+'_gripper:right_fingertip:slide:control')
                left_slide_joint_qvel = self.sim.data.get_joint_qvel(self.which_hand+'_gripper:left_fingertip:slide:control')
                gripper_state = np.array([right_slide_joint_qpos, left_slide_joint_qpos])
                gripper_vel = np.array([right_slide_joint_qvel, left_slide_joint_qvel])*dt

            # ee_xpos= self.sim.data.get_body_xpos(self.which_hand + '_gripper:hand')
            ee_velp = self.sim.data.get_body_xvelp(self.which_hand + '_gripper:hand')
            
            obs = np.concatenate([ee_pos, obj_pos.ravel(), obj_rel_pos.ravel(), gripper_state, obj_rot.ravel(), obj_velp.ravel(), obj_velr.ravel(), ee_velp, gripper_vel], axis =-1)

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
        action = action.copy()
        # print('action in SingleUR3GoalMocapEnv step : ', action)
        # actions of remaning arm will be garbage
        # gripper [-1,1] should be mod to
        self._set_action(action)


        # if self.flat_gripper:
        #     # flat gripper should be welded to follower link of robotq
        #     for i in range(self.sim.model.eq_data.shape[0]):
        #         if self.sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
        #             if 'dummy' in self.sim.model.body_id2name(self.sim.model.eq_obj1id[i]):
        #                 print('dummy weld set')
        #                 self.sim.model.eq_data[i, :] = np.array([0., 0., 0., 1., 0., 0., 0.])
        #     self.sim.forward()

        multi_step = 100
        for i in range(multi_step):
            self.sim.step()
        # self._step_callback() # gripper close related
        

        # action = action.copy()
        # right_joint_ctrl, left_joint_ctrl, right_gripper_ctrl, left_gripper_ctrl = action[:6], action[6:8], action[8:14], action[14:]
        
        # if self.block_gripper: # close gripper
        #     right_gripper_ctrl = 50*np.ones_like(right_gripper_ctrl)
        #     left_gripper_ctrl = 50*np.ones_like(left_gripper_ctrl)

        # action = np.concatenate([right_joint_ctrl, left_joint_ctrl, right_gripper_ctrl, left_gripper_ctrl], axis =-1)            

        # self.do_simulation(action, self.frame_skip)
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
            start_p, end_p = self.ur3_nqpos+self.gripper_nqpos+self.flat_gripper_nqpos, 2*self.ur3_nqpos+2*self.gripper_nqpos+2*self.flat_gripper_nqpos
            start_v, end_v = self.ur3_nqvel+self.gripper_nqvel+self.flat_gripper_nqvel, 2*self.ur3_nqvel+2*self.gripper_nqvel+2*self.flat_gripper_nqvel
            qpos[start_p:end_p] = self.left_get_away_qpos
            qvel[start_v:end_v] = np.zeros(end_v-start_v)
        elif self.which_hand=='left':
            #right arm's qpos,qvel index
            start_p, end_p = 0, self.ur3_nqpos+self.gripper_nqpos+self.flat_gripper_nqpos
            start_v, end_v = 0, self.ur3_nqvel+self.gripper_nqvel+self.flat_gripper_nqvel
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

    def convert_goal_for_reward(self, goal): #needed for TDMWrapper
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

    def convert_goal_for_reward_tf(self, goals): #needed for TDMWrapper
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


class DSCHOSingleUR3PickAndPlaceEnv(DSCHOSingleUR3GoalMocapEnv):
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(has_object=True, block_gripper=False, task='pickandplace', *args, **kwargs)

class DSCHOSingleUR3PushEnv(DSCHOSingleUR3GoalMocapEnv):
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(has_object=True, block_gripper=True, task='push',*args, **kwargs)

class DSCHOSingleUR3ReachEnv(DSCHOSingleUR3GoalMocapEnv):
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(has_object=False, block_gripper=True, task='reach', *args, **kwargs)

class DSCHOSingleUR3AssembleEnv(DSCHOSingleUR3GoalMocapEnv):
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(has_object=True, block_gripper=False, task='assemble', *args, **kwargs)

class DSCHOSingleUR3DrawerOpenEnv(DSCHOSingleUR3GoalMocapEnv):
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(has_object=True, block_gripper=False, task='drawer_open', *args, **kwargs)
