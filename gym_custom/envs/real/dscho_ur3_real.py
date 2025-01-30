import copy
import numpy as np
import os
import traceback
from numpy.core.numeric import full
import gym_custom
from gym_custom.spaces import Box
from gym_custom import utils
# from gym_custom.envs.mujoco import MujocoEnv
from gym_custom.envs.real.ur3_env import UR3RealEnv
from gym_custom.envs.custom.ur_utils import URScriptWrapper, URScriptWrapper_DualUR3
from gym_custom import Wrapper
from gym_custom.envs.custom.ur_utils import SO3Constraint, UprightConstraint, NoConstraint
# import tensorflow as tf
import pickle
import time
import sys
# import mujoco_py
from gym_custom.envs.real.utils import prompt_yes_or_no
# from gym_custom.envs.robotics import rotations #, robot_env, utils
from gym_custom.envs.real import rotations
import pyzed.sl as sl
import cv2
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

def get_se3(R,p):
    T = np.concatenate([np.concatenate([R,  p[:, None]], axis = -1), np.array([[0,0,0,1]])], axis= 0) #[3,4], [1,4] -> [4,4]
    return T
def get_se3_inv(R,p):
    T = np.concatenate([np.concatenate([R.T,  np.matmul(-R.T, p[:, None])], axis = -1), np.array([[0,0,0,1]])], axis= 0) #[3,4], [1,4] -> [4,4]
    return T

def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

class SingleWrapperReal(object):
    def __init__(self,
                env,
                ):
        raise NotImplementedError('you should implement for joint control single wrapper')



class EndEffectorPositionControlSingleWrapperReal(object): #URScriptWrapper_DualUR3
    
    def __init__(self, 
                env,                 
                gripper_action, 
                q_control_type = 'speedj', 
                g_control_type = 'move_gripper_position',
                so3_constraint='vertical_side', 
                action_downscale=0.02,
                gripper_force_scale = 1,
                speedj_args=None, 
                servoj_args = None, 
                g_control_args = None,
                multi_step = 1,
                # dscho added 240807
                rotation_z = False,
                rot_scale = 1/10,
                *args, 
                **kwargs
                ):
        self.env = env

        self.q_control_type = q_control_type
        self.g_control_type = g_control_type
        
        self.ee_xyz_pos_dim = 3
        self.action_downscale = action_downscale
        self.gripper_force_scale = gripper_force_scale
        assert gripper_force_scale == 1
        self.null_obj_func = SO3Constraint(SO3=so3_constraint)

        self.rotation_z = rotation_z
        self.rot_scale = rot_scale

        self.gripper_action = gripper_action
        self.dt = self.env.dt
        self.gripper_max_close_for_pickandplace = 255 # 6cm block
        self.speedj_args = speedj_args
        self.servoj_args = servoj_args
        self.g_control_args = g_control_args
        self.multi_step = multi_step
        if gripper_action:
            if rotation_z:
                self.act_low = act_low = np.array([-1, -1, -1, -1, -1])
                self.act_high = act_high= np.array([1, 1, 1, 1, 1])
            else:
                self.act_low = act_low = np.array([-1, -1, -1, -1])
                self.act_high = act_high= np.array([1, 1, 1, 1])
        else :
            if rotation_z:
                self.act_low = act_low = np.array([-1, -1, -1, -1])
                self.act_high = act_high= np.array([1, 1, 1, 1])
            else:
                self.act_low = act_low = np.array([-1, -1, -1])
                self.act_high = act_high= np.array([1, 1, 1])
    
        
        self.ur3_act_dim = 3 #self.wrapper_right.ndof
        self.gripper_act_dim = 1 # self.wrapper_right.ngripperdof
        assert self.ur3_act_dim==3
        assert self.gripper_act_dim==1
        self.action_space = Box(low=act_low, high=act_high, dtype=np.float32)
        print(colorize('WARNING : CHECK action space boundary : {}'.format(self.action_space), 'green', bold=True))
        
    def reset(self, **kwargs):        
        if self.env.task=='covering' and self.env._episode_step is not None:
            # to prevent collision with object
            for _ in range(10):
                self.step(np.array([0.0, 0, 1,0])) # move to z direction
        elif self.env.task=='peg' and self.env._episode_step is not None:
            # to prevent collision with object
            for _ in range(10):
                self.step(np.array([0.0, -1.0, 0,0])) # move to y direction
        elif self.env.task=='image_pickandplace' and self.env._episode_step is not None:
            # to prevent resetting while grasping the object
            for _ in range(10):
                self.step(np.array([0.0, 0, 1, -1.0])) # move to z direction while opening the gripper

        return self.env.reset(**kwargs)
        # return super().reset(**kwargs)

    def gripper_action_scale(self, gripper_act):
        # Assume gripper act : [-1, 1]
        # Sim : -1 : fully open, 1 : fully closed
        # Real : 0 : fully open, 255 : fully closed
        gripper_act = gripper_act.copy()
        rescaled_act = (gripper_act+1)/(1-(-1)) # 0~1 rescale 
        rescaled_act = self.gripper_max_close_for_pickandplace*rescaled_act # *1.6 # 0~max_close_for_pickandplace rescale
        return rescaled_act 
        
    def step(self, action, wait = False):
        return self._step(action, wait=wait)

    # Wrapper 에서 multistep을 해야 제일 lowlevel에선 매 timestep마다 command 새로계산함으로써 정확도 증가되는데, 내부적으로 IK sol 쓰다보니 이런구조는 아니라 정확도 살짝 떨어질수도
    def _step(self, action, wait = False):
        action = action.copy()
        
        # action = copy.deepcopy(action) # 통째로 *downscale이면 문제 없는데 index로 접근할땐 array가 mutable이라 copy해줘야함, but 매 스텝마다 action을 새로 뽑는상황이라면 굳이 이렇게 안해도 상관없음. 똑같은 action으로 계속 step밟을 때나 문제가 되는거지
        action[:self.ee_xyz_pos_dim] = np.clip(action[:self.ee_xyz_pos_dim], -1, 1)
        action[:self.ee_xyz_pos_dim] = self.action_downscale*action[:self.ee_xyz_pos_dim]
        ur3_act = action[:self.ee_xyz_pos_dim] # delta xyz pos of ee
        
        if self.gripper_action:
            gripper_act = self.gripper_force_scale*action[-self.gripper_act_dim:]
            gripper_act = self.gripper_action_scale(gripper_act)
            # print('gripper act : ', gripper_act)
        else :
            # gripper_act = np.zeros(self.gripper_act_dim) # opened
            gripper_act = np.ones(self.gripper_act_dim) # closed
            gripper_act = self.gripper_action_scale(gripper_act)
        
        
        current_qpos = self.env.interface.get_joint_positions(wait=False)
        R, p, T = self.env.forward_kinematics_ee(current_qpos, self.env.which_hand)
        ee_pos = p
        
        desired_ee_pos = ee_pos + ur3_act
        
        if self.rotation_z:
            
            # TODO: should check which axis is rot_z direction. (maybe z axis)
            delta_rot_z = action[self.ee_xyz_pos_dim:self.ee_xyz_pos_dim+1]
            euler = rotations.mat2euler(R)
            rot_z = euler[-1]
            desired_rot_z = (rot_z+ self.rot_scale*delta_rot_z) % (np.pi*2) # TODO: not sure how to compute the rot_z in UR3 convention
            
            predefined_euler_xy = np.array([-np.pi, 0])
            desired_euler = np.concatenate([predefined_euler_xy, desired_rot_z])
            desired_SO3 = rotations.euler2mat(desired_euler)            
            self.null_obj_func.setSO3_des(desired_SO3)
        
        

        start = time.time()
        # ee_pos, null_obj_func, arm, q_init='current', threshold=0.01, threshold_null=0.001, max_iter=100, epsilon=1e-6
        q_des, iter_taken, err, null_obj = self.env.inverse_kinematics_ee(desired_ee_pos, self.null_obj_func, arm=self.env.which_hand, threshold=0.001, max_iter = 10)
        # print('right_iter taken : {}, left_iter taken : {}, time : {}'.format(right_iter_taken, left_iter_taken, time.time()-start))

        if self.q_control_type =='speedj':
            ur3_action = (q_des-current_qpos)/(self.dt)
            
        elif self.q_control_type =='servoj':
            ur3_action = q_des            
        gripper_action = gripper_act
        
        q_control_args, g_control_args = self._get_control_kwargs(self.q_control_type, self.g_control_type, ur3_action, gripper_action)
        
        q_control_args.update({'wait' : wait})
        
        command = {self.q_control_type : q_control_args,
                   self.g_control_type : g_control_args,
                   }
        for i in range(self.multi_step-1):
            self.env.step(command)
        return self.env.step(command)
        
    def _get_control_kwargs(self, q_control_type, g_control_type, ur3_action, gripper_action):

        if q_control_type=='speedj':
            q_control_args = copy.deepcopy(self.speedj_args)
            q_control_args.update({'qd' : ur3_action})
        elif q_control_type=='servoj':
            q_control_args = copy.deepcopy(self.servoj_args)
            q_control_args.update({'q' : ur3_action})
        
        if g_control_type=='move_gripper_force':
            raise NotImplementedError
            g_control_args.update({'gf' : gripper_action})
        elif g_control_type=='move_gripper_position':            
            g_control_args = copy.deepcopy(self.g_control_args)
            g_control_args.update({'g' : gripper_action})
        elif g_control_type=='move_gripper_velocity':
            raise NotImplementedError
            g_control_args.update({'gd' : gripper_action})
        
        return q_control_args, g_control_args

    def __getattr__(self, name):
        return getattr(self.env, name)

import numpy as np
from scipy.spatial.transform import Rotation as R
def euler_to_rmat(euler, degrees=False):
    return R.from_euler("xyz", euler, degrees=degrees).as_matrix()

def rmat_to_euler(rot_mat, degrees=False):
    euler = R.from_matrix(rot_mat).as_euler("xyz", degrees=degrees)
    return euler

def rotation_matrix_to_axis_angle(R):
    epsilon = 1e-7
    trace = np.trace(R)
    # Clamp value for numerical precision
    cos_theta = max(min((trace - 1) / 2, 1.0), -1.0)
    theta = np.arccos(cos_theta)

    if abs(theta) < epsilon:
        # No significant rotation
        return np.zeros(3, dtype=float)
    elif abs(theta - np.pi) < epsilon:
        # Angle close to pi
        # A robust way is to find which diagonal element of R is largest
        # and use that axis
        # Here is a simple method:
        axis = np.array([R[0,0]+1, R[1,1]+1, R[2,2]+1])
        axis_norm = np.linalg.norm(axis)
        if axis_norm < epsilon:
            # This is a rare numerical edge case
            # fallback to e.g. using off-diagonal elements
            axis = np.array([R[2,1] - R[1,2],
                             R[0,2] - R[2,0],
                             R[1,0] - R[0,1]])
            axis_norm = np.linalg.norm(axis)
        axis = axis / axis_norm
        return axis * theta
    else:
        # General case
        rx = (R[2,1] - R[1,2]) / (2*np.sin(theta))
        ry = (R[0,2] - R[2,0]) / (2*np.sin(theta))
        rz = (R[1,0] - R[0,1]) / (2*np.sin(theta))
        return np.array([rx, ry, rz]) * theta
    


class EndEffectorSE3ControlSingleWrapperReal(object): #URScriptWrapper_DualUR3
    
    def __init__(self, 
                env,                 
                q_control_type = 'servoj', 
                g_control_type = 'move_gripper_position',
                gripper_force_scale = 1,
                speedj_args = None, 
                servoj_args = None, 
                movej_args = None,
                movep_args = None,
                movel_args = None,
                g_control_args = None,
                multi_step = 1,
                *args, 
                **kwargs
                ):
        
        '''
        Assume that the input action (SE(3)) is reasonably close to end-effector. (e.g. generated SE(3) traj by diffusion model)
        '''
        self.env = env

        self.q_control_type = q_control_type
        self.g_control_type = g_control_type
        # assert q_control_type == 'servoj', 'assume action is desired SE(3) of end effector'
        
        self.gripper_force_scale = gripper_force_scale
        assert gripper_force_scale == 1
        
        self.null_obj_func = SO3Constraint()
        
        self.dt = self.env.dt
        print('self.dt in wrapper : ', self.dt)
        self.gripper_max_close_for_pickandplace = 255
        self.speedj_args = speedj_args
        self.servoj_args = servoj_args
        self.movej_args = movej_args
        self.movep_args = movep_args
        self.movel_args = movel_args
        self.g_control_args = g_control_args
        self.multi_step = multi_step
        
        self.ur3_act_dim = 6 #self.wrapper_right.ndof
        self.gripper_act_dim = 1 # self.wrapper_right.ngripperdof

        self.T_gripper_tcp = np.eye(4)
        # NOTE: actual tcp has some offset in z-axis translation and 180 rotation around z-axis
        # self.T_gripper_tcp[:3,:3] = np.array([[-1.0, 0, 0],
        #                                     [0, -1.0, 0],
        #                                     [0, 0, 1.0]])
        self.T_gripper_tcp[:3, 3] = np.array([0, 0, 0]) # NOTE: translation should be manually calibrated
        
    def reset(self, **kwargs):                
        return self.env.reset(**kwargs)
        

    def gripper_action_scale(self, gripper_act):
        # Assume gripper act : [-1, 1]
        # Sim : -1 : fully open, 1 : fully closed
        # Real : 0 : fully open, 255 : fully closed
        gripper_act = gripper_act.copy()
        rescaled_act = (gripper_act+1)/(1-(-1)) # 0~1 rescale 
        rescaled_act = self.gripper_max_close_for_pickandplace*rescaled_act # *1.6 # 0~max_close_for_pickandplace rescale
        return rescaled_act 
        
    def step(self, action, wait = False):
        return self._step(action, wait=wait)

    
    def _step(self, action, wait = False):
        action = action.copy() # action : desired SE(3) of gripper (T_w_gripper)
        
        desired_gripper_pose = action[:self.ur3_act_dim]
        desired_gripper_pos = desired_gripper_pose[:3]
        desired_gripper_euler = desired_gripper_pose[-3:]
        T_w_gripper = np.eye(4)
        T_w_gripper[:3, :3] = euler_to_rmat(desired_gripper_euler)
        T_w_gripper[:3, 3] = desired_gripper_pos
        
        

        T_w_tcp = T_w_gripper @ self.T_gripper_tcp
                
        gripper_act = self.gripper_force_scale*action[-self.gripper_act_dim:]
        gripper_act = self.gripper_action_scale(gripper_act)
                
        if self.q_control_type =='speedj':
            current_qpos = self.env.interface.get_joint_positions(wait=False)
        
        if self.q_control_type in ['speedj', 'servoj']:
            # ee_pos, null_obj_func, arm, q_init='current', threshold=0.01, threshold_null=0.001, max_iter=100, epsilon=1e-6
            self.null_obj_func.setSO3_des(T_w_gripper[:3, :3])
            desired_tcp_pos = T_w_tcp[:3, 3]
            q_des, iter_taken, err, null_obj = self.env.inverse_kinematics_ee(desired_tcp_pos, self.null_obj_func, arm=self.env.which_hand, threshold=0.001, max_iter = 100)
            print('iter taken : {}, err : {} null_obj : {}'.format(iter_taken, err, null_obj))
        
        elif self.q_control_type == 'movej':
            T_b_tcp =  np.linalg.inv(self.env.kinematics_params['T_wb_right']) @ T_w_tcp
            desired_tcp_pos_b = T_b_tcp[:3, 3]
            desired_tcp_R_b = T_b_tcp[:3, :3] # assume rigidbody -> desired_tcp_R == T_w_gripper
            desired_tcp_axis_angle_b = rotation_matrix_to_axis_angle(desired_tcp_R_b)
            desired_tcp_pose = np.concatenate([desired_tcp_pos_b, desired_tcp_axis_angle_b])
            inverse_kin_has_solution = self.env.interface.is_within_safety_limits(desired_tcp_pose)
            if not inverse_kin_has_solution:
                print('desired_tcp_pose is not within safety limits! stay at the current joint positions!')
                q_des = self.env.interface.get_joint_positions()
            else:
                q_des = self.env.interface.get_inverse_kin(desired_tcp_pose, maxPositionError =0.0001, maxOrientationError =0.0001)
            
        if self.q_control_type =='speedj':
            ur3_action = (q_des-current_qpos)/(self.dt)
            
        elif self.q_control_type == 'servoj':
            ur3_action = q_des
            
        elif self.q_control_type == 'movej':
            ur3_action = q_des
            
            
        if self.q_control_type in ['movep', 'movel']:            
            # NOTE: should represent via robot base frame (not world frame), since movep,movec,movel uses inverse_kinematics of UR3, not custom one.
            # i.e. T_btcp = T_bw @ T_wtcp
            T_b_tcp =  np.linalg.inv(self.env.kinematics_params['T_wb_right']) @ T_w_tcp
            desired_tcp_pos_b = T_b_tcp[:3, 3]
            desired_tcp_R_b = T_b_tcp[:3, :3] # assume rigidbody -> desired_tcp_R == T_w_gripper
            desired_tcp_axis_angle_b = rotation_matrix_to_axis_angle(desired_tcp_R_b)
            ur3_action = np.concatenate([desired_tcp_pos_b, desired_tcp_axis_angle_b])

        gripper_action = gripper_act
        
        q_control_args, g_control_args = self._get_control_kwargs(self.q_control_type, self.g_control_type, ur3_action, gripper_action)
        
        q_control_args.update({'wait' : wait})
        # dscho mod
        g_control_args.update({'wait' : wait})
        
        command = {self.q_control_type : q_control_args,
                   self.g_control_type : g_control_args,
                   }
        for i in range(self.multi_step-1):
            self.env.step(command)
        return self.env.step(command)
        
    def _get_control_kwargs(self, q_control_type, g_control_type, ur3_action, gripper_action):

        if q_control_type=='speedj':
            q_control_args = copy.deepcopy(self.speedj_args)
            q_control_args.update({'qd' : ur3_action})
        elif q_control_type=='servoj':
            q_control_args = copy.deepcopy(self.servoj_args)
            q_control_args.update({'q' : ur3_action})
        elif q_control_type=='movej':
            q_control_args = copy.deepcopy(self.movej_args)
            q_control_args.update({'q' : ur3_action})
            
        elif q_control_type=='movep':
            q_control_args = copy.deepcopy(self.movep_args)
            q_control_args.update({'pose' : ur3_action})
        elif q_control_type=='movel':
            q_control_args = copy.deepcopy(self.movel_args)
            q_control_args.update({'pose' : ur3_action})

        if g_control_type=='move_gripper_force':
            raise NotImplementedError
            g_control_args.update({'gf' : gripper_action})
        elif g_control_type=='move_gripper_position':            
            g_control_args = copy.deepcopy(self.g_control_args)
            g_control_args.update({'g' : gripper_action})
        elif g_control_type=='move_gripper_velocity':
            raise NotImplementedError
            g_control_args.update({'gd' : gripper_action})
        
        return q_control_args, g_control_args

    def __getattr__(self, name):
        return getattr(self.env, name)



class DSCHOUR3RealEnv(UR3RealEnv):
    def __init__(self, *args, **kwargs):




        # # self.save_init_params(locals())
        # self.init_qpos_candidates = {}
        # self.init_qpos_type = 'mounted' # 'upright'            
        # self.ur3_nqpos = 6
        # # 양팔 널찍이 벌려있는 상태
        # # default_right_qpos = np.array([[-90.0, -90.0, -90.0, -90.0, -135.0, 180.0]])*np.pi/180.0 #[num_candidate+1, qpos_dim]
        # default_left_qpos = np.array([[90.0, -90.0, 90.0, -90.0, 135.0, -180.0]])*np.pi/180.0 #[num_candidate+1, qpos_dim]
        
        # # right : [0.15, -0.35, 0.9] left : [-0.2, -0.3, 0.8]
        # # NOTE : 주의 ! qpos array rank is 2 !
        # # [0.2, -0.3, 0.8]
        # # default_right_qpos = np.array([[-0.73475149, -1.91237669, -1.78802014, -1.6064106, -2.07919236,  2.16932592]])
        # # [0.15, -0.35, 0.9]
        # # default_right_qpos = np.array([[-0.90259643, -2.24937667, -1.82423119, -1.23998854, -2.15827838,  2.2680261 ]])
        # # [0.15, -0.35, 0.8]
        # if self.init_qpos_type=='upright':
        #     default_right_qpos = np.array([[-90.0, -90.0, -90.0, -90.0, 90.0, 180.0]])*np.pi/180.0 #[num_candidate+1, qpos_dim]
        # elif self.init_qpos_type=='mounted':
        #     default_right_qpos = np.array([109.44559205, -12.49065745, 73.33899229, -138.47907012, 47.67841212,-12.53005398])
        # else:
        #     default_right_qpos = np.array([[-0.76263046, -2.21085609, -1.50821658, -1.57404046, -2.08100962, 2.19369591]])
        # # default_left_qpos = np.array([[0.73490191, -1.22867589, 1.78775333, -1.53617814, 2.07956014, -2.16994491]])
        
        # # add default qpos configuration        
        # self.init_qpos_candidates['q_right_des'] =default_right_qpos
        # self.init_qpos_candidates['q_left_des'] = default_left_qpos
        
        # # for push or reach (closed gripper)
        # if self.task in ['push', 'reach']:
        #     default_gripper_right_qpos = np.array([[0.70005217, 0.01419325, 0.0405478, 0.0134475, 0.74225534, 0.70005207, 0.01402114, 0.04054553, 0.01344841, 0.74224361]])
        #     default_gripper_left_qpos = np.array([[0.70005217, 0.01419325, 0.0405478, 0.0134475, 0.74225534, 0.70005207, 0.01402114, 0.04054553, 0.01344841, 0.74224361]])
        #     # add default qpos configuration        
        #     self.init_qpos_candidates['gripper_q_right_des'] =default_gripper_right_qpos
        #     self.init_qpos_candidates['gripper_q_left_des'] =default_gripper_left_qpos

        self.gripper_min_position = 0
        self.gripper_max_position = 255 # just for _get_obs in _define_class_variables
        super().__init__(*args, **kwargs)


        self.gripper_min_position = self.interface.get_gripper_min_position()
        self.gripper_max_position = self.interface.get_gripper_max_position()

    def _define_class_variables(self):
        '''overridable method'''
        # Initial position/velocity
        self._init_qpos = np.zeros([6])
        self._init_qvel = np.zeros([6])
        self._init_gripperpos = np.zeros([1])
        self._init_grippervel = np.zeros([1])

        # Variables for forward/inverse kinematics
        # https://www.universal-robots.com/articles/ur-articles/parameters-for-calculations-of-kinematics-and-dynamics/
        self.kinematics_params = {}

        # 1. Last frame aligns with (right/left)_ee_link body frame
        # self.kinematics_params['d'] = np.array([0.1519, 0, 0, 0.11235, 0.08535, 0.0819]) # in m
        # 2. Last frame aligns with (right/left)_gripper:hand body frame
        self.kinematics_params['d'] = np.array([0.1519, 0, 0, 0.11235, 0.08535, 0.0819+0.12]) # in m
        self.kinematics_params['a'] = np.array([0, -0.24365, -0.21325, 0, 0, 0]) # in m
        self.kinematics_params['alpha'] =np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]) # in rad
        self.kinematics_params['offset'] = np.array([0, 0, 0, 0, 0, 0])
        self.kinematics_params['ub'] = np.array([2*np.pi for _ in range(6)])
        self.kinematics_params['lb'] = np.array([-2*np.pi for _ in range(6)])
        
        if self.init_qpos_type=='upright':
            print('############################## init qpos type is upright. upright_ur3_kinematics_params is loaded.')
            path_to_pkl = os.path.join(os.path.dirname(__file__), 'ur/upright_ur3_kinematics_params.pkl')
        else: # dual arm posture params       
            path_to_pkl = os.path.join(os.path.dirname(__file__), 'ur/dual_ur3_kinematics_params.pkl')

        if os.path.isfile(path_to_pkl):
            kinematics_params_from_pkl = pickle.load(open(path_to_pkl, 'rb'))
            self.kinematics_params['T_wb_right'] = kinematics_params_from_pkl['T_wb_right']
            if self.init_qpos_type=='upright':
                self.kinematics_params['T_wb_left'] = kinematics_params_from_pkl['T_wb_right'] # dscho NOTE: since pkl file is obtained where right arm is properly set, and left arm is far away yyo use right arm only
            else:
                self.kinematics_params['T_wb_left'] = kinematics_params_from_pkl['T_wb_left']
        else:
            raise FileNotFoundError('No such file: %s. Run MuJoCo-based simulated environment to generate file.'%(path_to_pkl))
        
        # Define spaces
        self.action_space = self._set_action_space()
        obs = self._get_obs()
        self.observation_space = self._set_observation_space(obs)

        # Misc
        self._episode_step = None

        
    def _get_init_qpos(self):
        return self._init_qpos.copy()

    def gripper_rescale(self, gripper_state):
        # assume gripper state is gripper_min ~gripper max(ideally, 0~255 , but auto-calibarted value is around 3~230)
        # Sim action : -1 : fully open, 1 : fully closed
        # Sim flat gripper state : 0 : fully open, 0.05 : fully closed
        # Real : 0 : fully open, 255 : fully closed
        gripper_state = gripper_state[0].copy()
        assert gripper_state >=0 and gripper_state <= 255, 'assume gripper state is raw data from ur interface'
        gripper_state = (gripper_state-self.gripper_min_position)/(self.gripper_max_position - self.gripper_min_position) # 0~1
        # gripper_state = gripper_state/255 # 0~1
        gripper_state *= 0.05 # 0 ~ 0.05
        gripper_state = np.array([gripper_state, gripper_state])
        return gripper_state

    # dscho added for real-world ARL experiments
    def get_joint_positions(self, wait=False):        
        return self.interface.get_joint_positions(wait=wait)
    
    def get_joint_speeds(self, wait=False):        
        return self.interface.get_joint_speeds(wait=wait)
    
    def get_actual_tcp_pose(self, wait=False):
        return self.interface.get_actual_tcp_pose(wait=wait)

    def call_stopj(self):
        self.interface.stopj(a=2.5, wait=True) # prevent protecive stop(invalid setpoints: sudden stop) error
        
    def movej_to_qpos(self, desired_qpos):
        # dscho commented to make same as dual_ur3_env.py
        # self.interface.movej(q=self._init_qpos)

        self.interface.stopj(a=5, wait=True) # prevent protecive stop(invalid setpoints: sudden stop) error

        controller_error = lambda stats: np.any([(stat.safety.StoppedDueToSafety) or (not stat.robot.PowerOn) for stat in stats])
        movej_success = False
        while not movej_success:
            try:
                self.interface.movej(q=desired_qpos)
                for _ in range(2):
                    obs_dict = self.get_obs_dict()
                    movej_success = np.linalg.norm(obs_dict['qpos'] - desired_qpos, np.inf) < np.deg2rad(3)
                    if movej_success: break
                    time.sleep(0.1)
                    self.interface.movej(q=desired_qpos)
                if not movej_success:
                    print('movej of reset_model did not register for some reason..')
                    # beepy.beep('error')
                    if prompt_yes_or_no("Press 'Y' to resend movej command. Press 'n' to terminate program.") is False:
                        print('exiting program!')
                        sys.exit()
            except Exception as e:
                print('hardware error during movej of reset_model')
                traceback.print_exc()
                if controller_error([self.interface.get_controller_status()]):
                    self._recover_from_controller_error()
                # beepy.beep('error')
                if prompt_yes_or_no("Press 'Y' after untangling robot arms. Press 'n' to terminate program.") is False:
                    print('exiting program!')
                    sys.exit()
        
        self.interface.move_gripper(g=self._init_gripperpos)
        # return self._get_obs() # dscho commented (to prevent too frequent calls of get_joint_positions)

    def get_endeff_pos(self, arm, q = None, wait=False, return_with_rotation=False):
        if q is None:
            q= self.interface.get_joint_positions(wait=wait)
            
        else :
            pass
        R, p, T = self.forward_kinematics_ee(q, arm)
        if return_with_rotation:
            return R,p
        else:
            return p


    # state_goal should be defined in child class
    def get_current_goal(self):
        return self._state_goal.copy()
        

    # state_goal, subgoals should be defined in child class
    def set_goal(self, goal):
        self._state_goal = goal
        # self._set_goal_marker(goal)

    

#single ur3 만들고 그 하위로 reachenv 만들수도
class DSCHOSingleUR3GoalRealEnv(DSCHOUR3RealEnv):
    
    # Sholud be used with URScriptWrapper
    
    def __init__(self,
                sparse_reward = False,               
                trigonometry_observation = True, 
                # ur3_random_init=False,
                full_state_goal = False,
                reward_by_ee = False,                                 
                reward_success_criterion='ee_pos',
                distance_threshold = 0.05,
                # initMode='vertical',
                which_hand='right', 
                so3_constraint ='vertical_side',
                has_object = False,
                block_gripper = False,                
                task = 'pickandplace',
                observation_type='joint_q', #'ee_object_object', #'ee_object_all'
                init_qpos_type = None,
                predefined_goal = None,
                optitrack_subscriber = None,
                custom_processed_vel = False,
                # dscho added for ARL
                reset_at_goal = False,
                *args,
                **kwargs
                ):
        # self.save_init_params(locals())

        self.full_state_goal = full_state_goal        
        self.trigonometry_observation = trigonometry_observation
        self._state_goal = None # np.zeros(3)
        self.task = task
        self.observation_type = observation_type
        self.init_qpos_type = init_qpos_type
        # for LEAP(S=G) -> modified for S!=G (20200930)
        self.reward_by_ee = reward_by_ee
        #self.ur3_random_init = ur3_random_init
        
        self.predefined_goal = predefined_goal

        if so3_constraint == 'no' or so3_constraint is None:
            self.so3_constraint = NoConstraint()
        elif so3_constraint == 'upright':
            self.so3_constraint = UprightConstraint()
        else:
            self.so3_constraint = SO3Constraint(SO3=so3_constraint)
        
        self.sparse_reward = sparse_reward
        self.distance_threshold = distance_threshold
        self.custom_processed_vel = custom_processed_vel
        self.block_gripper = block_gripper
        self.has_object = has_object
        if has_object:
            # assert optitrack_subscriber is not None
            self.optitrack_subscriber = optitrack_subscriber
            
        self.table_z_offset = 0.755
        # dscho added for ARL
        self.reset_at_goal = reset_at_goal
        
        # old ver (mounted)
        # peg_insert_bias = np.array([0, -0.03, 0])
        # upright
        peg_insert_bias = np.array([-0.01, 0., 0])
        self.predefined_goal_dict = {
                                    # old ver (tall table)
                                    #  'peg_forward' : np.array([0.15232019, -0.50705489, 0.88629302]) + peg_insert_bias,
                                    #  'peg_backward' : np.array([0.15385023, -0.25421638, 0.80274345]),
                                    #  'sweep_forward' : np.array([[0.15, -0.35, self.table_z_offset+0.05],
                                    #                             [-0.05, -0.35, self.table_z_offset+0.05]]),
                                    #  'sweep_backward' : np.array([0.05, -0.35, self.table_z_offset+0.05]),
                                    # old ver2 (mounted)
                                    #  'peg_forward' : np.array([0.22, -0.4, 0.76]) + peg_insert_bias,
                                    #  'peg_backward' : np.array([0.22, -0.25421638, 0.80274345]),
                                     
                                     # upright
                                    #  'peg_forward' : np.array([-0.34, -0.345, 0.866]) + peg_insert_bias,
                                    #  'peg_backward' : np.array([-0.15, -0.36, 0.9]),
                                     'sweep_forward' : np.array([[0.08, -0.385, 0.8],
                                                                [-0.15, -0.385, 0.8]]),
                                     'sweep_backward' : np.array([-0.039, -0.385, 0.8]),                                     
                                    #  'covering_forward' : np.array([0.178, -0.368, 0.78]),
                                    #  'covering_backward' : np.array([-0.1, -0.34,  0.9]),
                                     # left top : [-0.04018892 -0.25101522  0.79990623] left bottom : [-0.04441312 -0.40260624  0.80429047]
                                     # right top : [ 0.17620757 -0.24521262  0.79909224] right bottom : [ 0.18331614 -0.40192872  0.7892433 ]
                                    #  'image_pickandplace_forward' : np.array([0.067, -0.324, 0.78]),
                                    #  'image_pickandplace_backward' : np.array([[-0.04018892, -0.25101522,  0.79990623],
                                    #                                            [-0.04441312, -0.40260624,  0.80429047],
                                    #                                            [ 0.17620757, -0.24521262,  0.79909224],
                                    #                                            [ 0.18331614, -0.40192872,  0.7892433 ]]),
                                     
                                     # 241106 (after rearrangement of robots in ASRI)(not yet implemented)
                                     'peg_forward' : np.array([0.321, 0.37, 0.899]),
                                     'peg_backward' : np.array([0.316, 0.11, 0.9]),
                                     'covering_forward' : np.array([0.394, 0.29, 0.78]),
                                     'covering_backward' : np.array([0.306, -0.04, 0.9]),
                                     'image_pickandplace_forward' : np.array([-0.367, -0.05, 0.78]),
                                     'image_pickandplace_backward' : np.array([[-0.281, 0.09,  0.78],
                                                                               [-0.459, 0.09,  0.78],
                                                                               [-0.278, -0.165,  0.78],
                                                                               [-0.463, -0.162,  0.78]]),

                                     }

        self.previous_ee_pos = None
        self.previous_obj_pos = None
        
        self.reward_success_criterion = reward_success_criterion
        
        self.which_hand = which_hand
        super().__init__(# initMode=initMode, 
                        # ur3_random_init=ur3_random_init, 
                        *args, 
                        **kwargs
                        )
        
        self._state_goal = self.sample_goal(self.full_state_goal)

        if self.task == 'peg':
            if self.reset_at_goal:
                print('@@@@@@@ You should define initial joint pos correspond to peg goal or define peg_goal correspond to current init qpos!')
                # old ver (tall table)
                # self.set_initial_joint_pos(np.array([92.9005243, -18.10654826, 42.78848504, -110.46895611, 43.94059452, 79.73609282])*np.pi/180.0)
                # old ver2 (mounted)
                # self.set_initial_joint_pos(np.array([1.731179, -0.87071163, 1.14605093, -1.68206484, 0.79754043, 1.33863306]))
                # upright
                # self.set_initial_joint_pos(np.array([0.54606271, -0.68923074, 0.39925241, -1.28086836, -1.56263048, -1.01299936]))
                
                # 241106 (after rearrangement of robots in ASRI)
                # init ee pos : [0.394,  0.29,  0.78]
                self.set_initial_joint_pos(np.array([3.74169946, -1.06351883,  1.05295706, -1.82543356, -1.56358225, 0.56332153])) # right arm
                
            else:
                # old ver2 (mounted)
                # self.set_initial_joint_pos(np.array([107.16792204, -71.10452686, 120.7596843, -121.44078075, 46.43955241, 60.04848235])*np.pi/180.0)
                # upright (x=0)
                # self.set_initial_joint_pos(np.array([1.17900097, -1.34164936, 1.24671173, -1.4816764, -1.5633033, -0.3801921 ]))
                # upright (x=-0.148)
                # self.set_initial_joint_pos(np.array([0.90049171, -1.21185524, 1.07228374, -1.43334037, -1.56223375, -0.65869314]))
                
                # 241106 (after rearrangement of robots in ASRI)
                # init ee pos : [-0.3263,  0.1537,  1.01]
                # self.set_initial_joint_pos(np.array([-0.7485, -1.2905,  0.6997,  -1.0382, -1.5853, -0.66])) # left arm
                
                # init ee pos : [0.306,  -0.04,  0.97]                
                self.set_initial_joint_pos(np.array([3.10677338, -1.56326229, 1.52987003, -1.59008104, -1.50254471, -0.001])) # right arm
                

            self.set_initial_gripper_pos(np.array([255]))
            
        
        elif self.task == 'sweep':
            if self.reset_at_goal:
                # self.set_initial_joint_pos(np.array([98.68409626, -42.59550851, 91.25542542, -132.12103827, 45.30285526, 70.99222322])*np.pi/180.0)
                # upright
                # self.initial_joint_positions = np.array([[1.61340702, -1.12512523,  0.78730154, -1.19159109, -1.60718424, -1.50166256],
                #                                         [0.89749712, -1.01802522, 0.73561239, -1.27307445, -1.64055711, -2.22776157]]
                #                                         )
                
                self.set_initial_joint_pos(np.array([1.15561521, -1.29849226,  0.98778868, -1.28730709, -1.55795795, -1.98243553]))
            else: # vertical front
                # self.set_initial_joint_pos(np.array([98.68409626, -42.59550851, 91.25542542, -132.12103827, 45.30285526, 70.99222322])*np.pi/180.0)
                # upright: init ee pos :[-0.04273568 -0.38191446  0.87700127]
                self.set_initial_joint_pos(np.array([1.15573514, -1.26402408,  1.27845192, -1.5584448 , -1.55817396, -1.98209984]))

            self.set_initial_gripper_pos(np.array([255]))
        elif self.task == 'covering':
            if self.reset_at_goal:
                # upright
                # init ee pos : [ 0.17827251 -0.36799464  0.7791238 ]
                # self.set_initial_joint_pos(np.array([1.71711588, -1.0820306,  1.48655272, -1.94703085, -1.51943809, -1.47388536]))
                
                # 241106 (after rearrangement of robots in ASRI)
                # init ee pos : [0.394,  0.29,  0.78]
                self.set_initial_joint_pos(np.array([ 3.54779124, -0.81062395,  1.08774757, -1.94419986, -1.57904655,  0.42889708])) # right arm
                

            else:
                # upright
                # init ee pos : [-0.13195436 -0.34062076  0.98709191]
                # self.set_initial_joint_pos(np.array([0.89681441, -1.2870949, 0.79797077, -1.11509735, -1.58500892, -2.22543604]))

                # 241106 (after rearrangement of robots in ASRI)
                # init ee pos : [-0.3263,  0.1537,  1.01]
                # self.set_initial_joint_pos(np.array([-0.7485, -1.2905,  0.6997,  -1.0382, -1.5853, -2.3141])) # left arm
                
                # init ee pos : [0.306,  -0.04,  0.97]                
                self.set_initial_joint_pos(np.array([2.62560749, -1.56763441, 1.19621611, -1.2121237, -1.55632431, -0.48850662])) # right arm
                
                


            self.set_initial_gripper_pos(np.array([255]))

        elif self.task == 'arl_push':
            raise NotImplementedError
        elif self.task == 'image_pickandplace':
            if self.reset_at_goal:
                # upright
                # init ee pos : [ 0.06188416 -0.34819109  0.83701621]
                # self.set_initial_joint_pos(np.array([1.43031359, -1.3706773, 1.60599089 ,-1.78116781, -1.58229429 ,-1.73337967]))
                
                # 241106 (after rearrangement of robots in ASRI)
                # init ee pos : [-0.35,  -0.02,  0.9]
                self.set_initial_joint_pos(np.array([-0.2740143,  -1.46871597,  1.41530895 , -1.53634483 , -1.55199987, -1.80799324])) # left arm
            else:
                # upright
                # init ee pos : [ 0.06188416 -0.34819109  0.83701621]
                # self.set_initial_joint_pos(np.array([1.43031359, -1.3706773, 1.60599089 ,-1.78116781, -1.58229429 ,-1.73337967]))
                
                # 241106 (after rearrangement of robots in ASRI)
                # init ee pos : [-0.35,  -0.02,  0.9]
                self.set_initial_joint_pos(np.array([-0.2740143,  -1.46871597,  1.41530895 , -1.53634483 , -1.55199987, -1.80799324])) # left arm

            self.set_initial_gripper_pos(np.array([0]))
        elif self.task == 'moka':
            if self.reset_at_goal:
                # upright
                # init ee pos : [ 0.06188416 -0.34819109  0.83701621]
                # self.set_initial_joint_pos(np.array([1.43031359, -1.3706773, 1.60599089 ,-1.78116781, -1.58229429 ,-1.73337967]))
                raise NotImplementedError
            else:
                # upright
                # init ee pos : [0.2025, -0.0478,  0.1450]
                # self.set_initial_joint_pos(np.array([0.2195821, -1.18186123, -1.88059885, -1.72113067, 1.66767204, np.pi*3/2]))
                self.set_initial_joint_pos(np.array([0.15526275, -0.77878362, -1.95732385, -1.9798978, 1.64737856, 1.70835387]))
                
            self.set_initial_gripper_pos(np.array([0]))
        else:
            self.set_initial_joint_pos(np.array([3.10677338, -1.56326229, 1.52987003, -1.59008104, -1.50254471, -0.001]))
            self.set_initial_gripper_pos(np.array([0]))



        if self.which_hand =='right': 
            ee_low = np.array([-0.1, -0.5, 0.77])
            ee_high = np.array([0.35, -0.2, 0.95])
        
        elif self.which_hand =='left':
            ee_low = np.array([-0.35, -0.5, 0.77])
            ee_high = np.array([0.1, -0.2, 0.95])
        
        if self.init_qpos_type == 'upright': # right 기준
            ee_low = np.array([-0.2, -0.5, 0.77])
            ee_high = np.array([0.2, -0.2, 0.95])

        self.goal_ee_pos_space = Box(low = ee_low, high = ee_high, dtype=np.float32)
        
        # Currently, Set the goal obj space same sa ee pos sapce
        if self.which_hand =='right': 
            goal_obj_low = np.array([0.0, -0.45, self.table_z_offset])
            goal_obj_high = np.array([0.3, -0.3, 0.95])
        
        elif self.which_hand =='left':
            goal_obj_low = np.array([-0.3, -0.45, self.table_z_offset])
            goal_obj_high = np.array([0.0, -0.3, 0.95])
        
        if self.init_qpos_type == 'upright': # right 기준
            goal_obj_low = np.array([-0.15, -0.45, self.table_z_offset])
            goal_obj_high = np.array([0.15, -0.3, 0.95])

        self.goal_obj_pos_space = Box(low = goal_obj_low, high = goal_obj_high, dtype=np.float32)
        
        # TODO : think whether xy space is ouf of range of the table
        floor_z_height = np.array([0.73]) # xml보니 책상높이가 0.73?
        goal_obj_floor_low = np.concatenate([goal_obj_low[:-1], floor_z_height], axis =-1)
        goal_obj_floor_high = np.concatenate([goal_obj_high[:-1], floor_z_height], axis =-1)
        self.goal_obj_floor_space = Box(low = goal_obj_floor_low, high = goal_obj_floor_high, dtype=np.float32)

        # NOTE: should sample goal here if you want to sample it based on the goal_obj_ee_space, etc
        # self._state_goal = self.sample_goal(self.full_state_goal)
        

        # observation, reward, done, _info = self.step(self.action_space.sample()) # goalenv는 return obs_dict
        # assert not done
        
        # self.observation_space = self._set_observation_space(observation) 
    
    def _set_action_space(self):
        act_low = np.array([-1,-1,-1,-1, -1,-1,-1,-1])
        act_high = np.array([1,1,1,1, 1,1,1,1])
        self.action_space = Box(low=act_low, high=act_high, dtype=np.float32)

    # goal space == state space
    # ee_pos 뽑고, 그에 따른 qpos 계산(IK) or qpos뽑고 그에따른 ee_pos 계산(FK)
    # 우선은 후자로 생각(어처피 학습할땐 여기저기 goal 다 뽑고, 쓸때는 제한해서 goal 샘플할꺼니까)
    
    def sample_goal(self, full_state_goal):
        # need to mod for resample if goal is inside the wall
        if self.predefined_goal is not None:
            return self.predefined_goal

        if full_state_goal:
            raise NotImplementedError
        # dscho added for ARL
        elif self.reset_at_goal:
            if self.task in ['peg']:
                goal = self.predefined_goal_dict['peg_backward']
            elif self.task in ['sweep']:
                goal = self.predefined_goal_dict['sweep_backward']
            elif self.task in ['covering']:
                goal = self.predefined_goal_dict['covering_backward']
            elif self.task in ['image_pickandplace']:
                goals = self.predefined_goal_dict['image_pickandplace_backward']
                goal = goals[np.random.randint(goals.shape[0])]
            elif self.task in ['moka']:
                raise NotImplementedError
        else :
            if not self.has_object: # reach or tasks with always closed gripper
                # dscho added for ARL   
                if self.task in ['peg']:
                    goal = self.predefined_goal_dict['peg_forward']
                elif self.task in ['sweep']:
                    goals = self.predefined_goal_dict['sweep_forward']
                    goal = goals[np.random.randint(goals.shape[0])]
                elif self.task in ['covering']:
                    goal = self.predefined_goal_dict['covering_forward']
                elif self.task in ['image_pickandplace']:
                    goal = self.predefined_goal_dict['image_pickandplace_forward']
                elif self.task in ['moka']:
                    goal = np.zeros(3)
                else:
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
                elif self.task in ['reach']:
                    goal_ee_pos = np.random.uniform(
                        self.goal_ee_pos_space.low,
                        self.goal_ee_pos_space.high,
                        size=(self.goal_ee_pos_space.low.size),
                    )
                    goal = goal_ee_pos
                # dscho added for ARL
                elif self.task in ['peg']:
                    goal = self.predefined_goal_dict['peg_forward']
                else:
                    raise NotImplementedError

        return goal
    

    def reset_model(self, init=None, goal=None, prompt = True):
        if prompt:
            if prompt_yes_or_no('Resetting... Init qpos is %s deg. Did you prepare the GRASPING and OBJECT SETTING?'%(np.rad2deg(self._init_qpos))) is False:
                print('exiting program!')
                sys.exit()
        
        
        if self.task=='sweep':
            if self.reset_at_goal:
                # backward goal is only one, so does not need to care it
                self._state_goal = self.sample_goal(full_state_goal = self.full_state_goal)
                if init is not None:
                    if (init == self.predefined_goal_dict['sweep_forward'][0]).all():
                        self.set_initial_joint_pos(np.array([1.54041445, -1.18352491,  1.18829679, -1.5288499, -1.56138164, -1.55215103]))
                    elif (init == self.predefined_goal_dict['sweep_forward'][1]).all():
                        self.set_initial_joint_pos(np.array([0.89652717, -1.10477668,  1.16944218, -1.61763555, -1.65443308, -2.21739132]))
            else:
                if goal is not None:
                    self._state_goal = goal.copy()
                    
                else:
                    self._state_goal = self.sample_goal(full_state_goal = self.full_state_goal)

        elif self.task == 'image_pickandplace':
            if self.reset_at_goal:
                if init is not None:
                    self._state_goal = init.copy()



        else:
            self._state_goal = self.sample_goal(full_state_goal = self.full_state_goal)        
        



        self.previous_ee_pos = None
        self.previous_obj_pos = None
        observation = super().reset_model()

        return observation
        


    # Only ur3 qpos,vel(not include gripper), object pos(achieved_goal), desired_goal
    def _get_obs(self):
        
        obs_dict = super().get_obs_dict()
        # qpos, qvel gripperpos, grippervel
        qpos = obs_dict['qpos']
        qvel = obs_dict['qvel']
        gripperpos = obs_dict['gripperpos']
        gripperpos = self.gripper_rescale(gripper_state=gripperpos)
        # grippervel = obs_dict['grippervel']
        R, ee_pos = self.get_endeff_pos(arm=self.which_hand, q = qpos, return_with_rotation=True)

        
        if self.trigonometry_observation:
            qpos = np.concatenate([np.cos(qpos), np.sin(qpos)], axis = -1)

        # NOTE : e.g. 20Hz인데 20보다 작으면 dt가 부정확할수도(즉 0.05나와야하는데 실제론 0.06일수도)
        dt = self.dt
        if dt==0.1 and self.custom_processed_vel: # For matching sim policy inputs
            print('custom process vel is used!')
            dt = 0.05
            
            
        # print('In get_obs, dt : ', dt)


        if self.has_object:
            obj_pos = self.get_obj_pos(name='block_5cm_red')
            # obj_rot = rotations.quat2euler(self.get_obj_quat(name='obj'))
            # obj_velp = None #self.sim.data.get_site_xvelp('objSite') * dt
            # obj_velr = None #self.sim.data.get_site_xvelr('objSite') * dt
            obj_rel_pos = obj_pos - ee_pos
        else :
            obj_pos, obj_rot, obj_velp, obj_velr, obj_rel_pos = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        gripper_state = gripperpos
        
        if self.observation_type=='joint_q':
            obs = np.concatenate([qpos, qvel, ee_pos, obj_pos])
        elif self.observation_type == 'ee_object_pos':
            obs = np.concatenate([ee_pos, obj_pos, obj_rel_pos])
        elif self.observation_type == 'ee_object_pos_w_grip_custom_vel' or self.observation_type == 'ee_object_pos_rot_w_grip_custom_vel' or self.observation_type=='ee_object_pos_w_grip' or self.observation_type=='ee_object_pos_w_custom_vel':
            if self.previous_ee_pos is None:
                ee_velp = np.zeros_like(ee_pos)
            else:
                if self.custom_processed_vel:
                    ee_velp = (ee_pos - self.previous_ee_pos)/dt*0                
                else:
                    ee_velp = (ee_pos - self.previous_ee_pos)/dt

            if self.previous_obj_pos is None:
                obj_velp = np.zeros_like(obj_pos)
            else:
                if self.custom_processed_vel:
                    obj_velp = (obj_pos - self.previous_obj_pos)/dt*0
                else:
                    obj_velp = (obj_pos - self.previous_obj_pos)/dt
            self.previous_ee_pos = ee_pos.copy()
            self.previous_obj_pos = obj_pos.copy()

            if self.observation_type=='ee_object_pos_w_grip_custom_vel':
                obs = np.concatenate([
                    ee_pos, obj_pos.ravel(), obj_rel_pos.ravel(), gripper_state, 
                    obj_velp.ravel(), ee_velp
                ])
            elif self.observation_type=='ee_object_pos_rot_w_grip_custom_vel':
                ee_euler = rotations.mat2euler(R)
                obs = np.concatenate([
                    ee_pos, obj_pos.ravel(), obj_rel_pos.ravel(), ee_euler, gripper_state, 
                    obj_velp.ravel(), ee_velp
                ])
            elif self.observation_type=='ee_object_pos_w_custom_vel':
                obs = np.concatenate([
                    ee_pos, obj_pos.ravel(), obj_rel_pos.ravel(), obj_velp.ravel(), ee_velp
                ])
            else:
                obs = np.concatenate([ee_pos, obj_pos.ravel(), obj_rel_pos.ravel(), gripper_state])

        elif self.observation_type == 'ee_object_all':
            raise NotImplementedError
            obs = np.concatenate([ee_pos, obj_pos.ravel(), obj_rel_pos.ravel(), gripper_state, obj_rot.ravel(), obj_velp.ravel(), obj_velr.ravel(), ee_velp, gripper_vel], axis =-1)

        if self.full_state_goal:
            achieved_goal = obs
        else :
            if not self.has_object: # reach
                achieved_goal = ee_pos
            else: # pick and place, push, ...
                if self.task in ['reach']:
                    achieved_goal = ee_pos
                else:
                    achieved_goal = obj_pos
        if self._state_goal is None:
            self._state_goal = np.zeros(3)

        return {
            'observation' : obs.copy(),
            'achieved_goal' : achieved_goal.copy(),
            'desired_goal' : self._state_goal.copy(), 
        }    

        
    
    def step(self, action):
        # action example :
        # {
        #     'servoj': {'q': waypoint, 't': 2/real_env.rate._freq, 'wait': False},
        #     'move_gripper_position': {'g' : 0, 'wait' : False} # open
        # }
        action = copy.deepcopy(action)
        
        observation, reward, done, info = super().step(action)


        # observation = self._get_obs()
        done = False
        #TODO: Should consider how to address done
        # done = True if info['is_success'] else False

        info = {
            'is_success': self._is_success(observation['achieved_goal'], self._state_goal),
            # for sync accyracy, do not recompute the ee pos. 
            'ee_pos' : observation['observation'][:3], # self.get_endeff_pos(arm=self.which_hand),            
            # 'null_obj_val' : self._calculate_so3_error().copy(),
            'l2_distance_to_goal' : np.linalg.norm(observation['desired_goal']-observation['achieved_goal'], ord=2, axis = -1), 
            'l1_distance_to_goal' : np.linalg.norm(observation['desired_goal']-observation['achieved_goal'], ord=1, axis = -1),
        }
        
        self.info = copy.deepcopy(info)
        reward = self.compute_reward(observation['achieved_goal'], self._state_goal, info)
       
        return observation, reward, done, info


    # def _calculate_so3_error(self):
    #     if self.which_hand =='right':
    #         ur3_qpos = self._get_ur3_qpos()[:self.ur3_nqpos]    
    #     elif self.which_hand =='left':
    #         ur3_qpos = self._get_ur3_qpos()[self.ur3_nqpos:]

    #     SO3, x, _ = self.forward_kinematics_ee(ur3_qpos, arm=self.which_hand)
    #     null_obj_val = self.so3_constraint.evaluate(SO3)

    #     return null_obj_val
    
    def get_info(self):
        return copy.deepcopy(self.info)

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
        # dscho added for ARL
        if self.task in ['peg']:
            placingDist = np.linalg.norm(achieved_goal[-3:] - desired_goal[-3:])
            if self.sparse_reward:
                # old ver (mounted)
                # obj_tip_ypos = achieved_goal[1] - 0.1 # bias for virtual obj_tip_site
                # upright
                obj_tip_xpos = achieved_goal[0] - 0.05 # bias for virtual obj_tip_site
                if self.reset_at_goal:
                    if placingDist < self.distance_threshold:
                        reward = 1.0
                    else :
                        reward = 0.0    
                else:
                    # old ver (mounted)
                    # if obj_tip_ypos < self._state_goal[1] and placingDist < self.distance_threshold:
                    # upright
                    if obj_tip_xpos < self._state_goal[0] and placingDist < self.distance_threshold:
                        reward = 1.0
                    else :
                        reward = 0.0    
            else:    
                reward = -placingDist
            return reward
        elif self.task in ['sweep', 'covering', 'image_pickandplace']:
            placingDist = np.linalg.norm(achieved_goal[-3:] - desired_goal[-3:])
            if self.sparse_reward:
                if placingDist < self.distance_threshold:
                    reward = 1.0
                else :
                    reward = 0.0    
            else:    
                reward = -placingDist
            return reward
        elif self.task in ['moka']:
            reward = 0
            return reward
        else:
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
                if self.task in ['push']:
                    # print('currently, dense push debug')
                    ee_pos = info['right_ee_pos']
                    from_ee_to_obj = np.linalg.norm(achieved_goal[-3:] - ee_pos[-3:])
                    reward = -from_ee_to_obj -placingDist
                else:
                    reward = -placingDist

            return reward

    
    # obj related

    def get_calibration_info(self, name):
        obs_dict = super().get_obs_dict()
        # qpos, qvel gripperpos, grippervel
        qpos = obs_dict['qpos']
        qvel = obs_dict['qvel']
        gripperpos = obs_dict['gripperpos']
        gripperpos = self.gripper_rescale(gripper_state=gripperpos)
        # grippervel = obs_dict['grippervel']
        ee_pos = self.get_endeff_pos(arm=self.which_hand, q = qpos)

        
        data = self.optitrack_subscriber.get_optitrack_data()
        table_pos_w = np.array([data['ur3_table']['x'], data['ur3_table']['y'], data['ur3_table']['z']])
        table_quat_w = np.array([data['ur3_table']['quat_x'], data['ur3_table']['quat_y'], data['ur3_table']['quat_z'],data['ur3_table']['quat_w']])
        object_pos_w = np.array([data[name]['x'], data[name]['y'], data[name]['z']])
        
        R_wt = rotations.quat2mat(table_quat_w)
        P_wt = table_pos_w
        T_wt = get_se3(R_wt, P_wt)
        T_wt_inv = get_se3_inv(R_wt, P_wt)
        
        T_gb = self.kinematics_params['T_wb_right'] # ground to base of robot
        T_gb[:, -1] = np.array([0, 0, 0.71, 1], dtype=np.float64)
        
        # NOTE : This is assumption!!
        R_bt = np.array([[-1,0,0], [0,-1,0], [0,0,1]], dtype=np.float64)
        P_bt = np.array([0.135, -0.105, 0.03], dtype=np.float64)

        T_bt = get_se3(R_bt, P_bt)
        
        obj_pos = np.matmul(np.matmul(np.matmul(T_gb, T_bt), T_wt_inv), np.concatenate([object_pos_w, np.array([1])])[:, None]).squeeze()[:3]
        
        gripper_state = gripperpos
        

            
        obs = np.concatenate([
            ee_pos, obj_pos.ravel()
        ])
        return obs
    
    def get_multiple_obj_pos(self, name=None):
        # return np.zeros(3)
        if self.optitrack_subscriber is None:
            print('optitrack is None')
            return np.zeros(3)
        data = self.optitrack_subscriber.get_optitrack_data()
        if name is not None:
            # obj_x = data[name]['x']
            # obj_y = data[name]['y']
            # obj_z = data[name]['z']
            # quat_x = data[name]['quat_x']
            # quat_y = data[name]['quat_y']
            # quat_z = data[name]['quat_z']
            # sequence = data[name]['sequence'] 
            table_pos_w = np.array([data['ur3_table']['x'], data['ur3_table']['y'], data['ur3_table']['z']])
            table_quat_w = np.array([data['ur3_table']['quat_x'], data['ur3_table']['quat_y'], data['ur3_table']['quat_z'],data['ur3_table']['quat_w']])
            assert name[0] == 'ur3_table'
            object_pos_w_list = []
            for id in name[1:]: 
                object_pos_w_list.append(np.array([data[id]['x'], data[id]['y'], data[id]['z']]))
            
            
            R_wt = rotations.quat2mat(table_quat_w)
            P_wt = table_pos_w
            T_wt = get_se3(R_wt, P_wt)
            T_wt_inv = get_se3_inv(R_wt, P_wt)
            
            T_gb = self.kinematics_params['T_wb_right'] # ground to base of robot
            T_gb[:, -1] = np.array([0, 0, 0.71, 1], dtype=np.float64)
            
            # NOTE : This is assumption!!
            R_bt = np.array([[-1,0,0], [0,-1,0], [0,0,1]], dtype=np.float64)
            P_bt = np.array([0.135, -0.105, 0.03], dtype=np.float64)


            T_bt = get_se3(R_bt, P_bt)
            obj_pos_list = []
            for i in range(len(name[1:])):
                obj_pos_list.append(np.matmul(np.matmul(np.matmul(T_gb, T_bt), T_wt_inv), np.concatenate([object_pos_w_list[i], np.array([1])])[:, None]).squeeze()[:3])
            obj_pos = np.stack(obj_pos_list, axis = 0) #[num_obj, dim]
            return obj_pos

    def get_obj_pos(self, name=None):
        # return np.zeros(3)
        if self.optitrack_subscriber is None:
            print('optitrack is None')
            return np.zeros(3)
        data = self.optitrack_subscriber.get_optitrack_data()
        if name is not None:
            # obj_x = data[name]['x']
            # obj_y = data[name]['y']
            # obj_z = data[name]['z']
            # quat_x = data[name]['quat_x']
            # quat_y = data[name]['quat_y']
            # quat_z = data[name]['quat_z']
            # sequence = data[name]['sequence'] 
            table_pos_w = np.array([data['ur3_table']['x'], data['ur3_table']['y'], data['ur3_table']['z']])
            table_quat_w = np.array([data['ur3_table']['quat_x'], data['ur3_table']['quat_y'], data['ur3_table']['quat_z'],data['ur3_table']['quat_w']])
            object_pos_w = np.array([data[name]['x'], data[name]['y'], data[name]['z']])
            
            R_wt = rotations.quat2mat(table_quat_w)
            P_wt = table_pos_w
            T_wt = get_se3(R_wt, P_wt)
            T_wt_inv = get_se3_inv(R_wt, P_wt)
            
            T_gb = self.kinematics_params['T_wb_right'] # ground to base of robot
            T_gb[:, -1] = np.array([0, 0, 0.71, 1], dtype=np.float64)
            
            # NOTE : This is assumption!!
            R_bt = np.array([[-1,0,0], [0,-1,0], [0,0,1]], dtype=np.float64)
            P_bt = np.array([0.135, -0.105, 0.03], dtype=np.float64)


            T_bt = get_se3(R_bt, P_bt)
            
            obj_pos = np.matmul(np.matmul(np.matmul(T_gb, T_bt), T_wt_inv), np.concatenate([object_pos_w, np.array([1])])[:, None]).squeeze()[:3]
            
            return obj_pos
            
        else:
            obj_x = data['x']
            obj_y = data['y']
            obj_z = data['z']
            quat_x = data['quat_x']
            quat_y = data['quat_y']
            quat_z = data['quat_z']
            sequence = data['sequence']
            return np.array([obj_x, obj_y, obj_z])
        # raise NotImplementedError('should be implemented with optitrack or vicon')

        # if not self.has_object:
        #     raise NotImplementedError
        # if name is None:
        #     return self.data.get_body_xpos('obj')
        # else :
        #     return self.data.get_body_xpos(name) 

    def get_obj_quat(self, name=None):
        raise NotImplementedError('should be implemented with optitrack or vicon')
        if not self.has_object:
            raise NotImplementedError
        if name is None:
            return self.data.get_body_xquat('obj')
        else :
            return self.data.get_body_xquat(name)
    

class DSCHOSingleUR3PickAndPlaceMultiObjectRealEnv(DSCHOSingleUR3GoalRealEnv):
    
    # Sholud be used with URScriptWrapper
    
    def __init__(self,                
                num_objects = 1,
                weight_dim = None,
                sequential_rl = False,
                predefined_sequential_goal_list = None,
                *args,
                **kwargs
                ):
        # self.save_init_params(locals())
        self.num_objects = num_objects
        self.goal_object_idx = 0
        self.goal_weight_is_set = False
        self.weight_dim = weight_dim
        self.sequential_rl = sequential_rl
        self.predefined_sequential_goal_list = predefined_sequential_goal_list
        
        super().__init__(has_object=True,  block_gripper=False,  task='pickandplace', *args, **kwargs)
        # if sequential_rl:
        #     self._state_goal = np.concatenate(predefined_sequential_goal_list)

    def _get_obs(self):
        obs_dict = super().get_obs_dict()
        # qpos, qvel gripperpos, grippervel
        qpos = obs_dict['qpos']
        qvel = obs_dict['qvel']
        gripperpos = obs_dict['gripperpos']
        gripperpos = self.gripper_rescale(gripper_state=gripperpos)
        # grippervel = obs_dict['grippervel']
        ee_pos = self.get_endeff_pos(arm=self.which_hand, q = qpos)

        self.ee_pos = ee_pos.copy()
        if self.trigonometry_observation:
            qpos = np.concatenate([np.cos(qpos), np.sin(qpos)], axis = -1)
        
        if self.has_object:
            object_pos_list = []
            object_rot_list = []
            object_velp_list = []
            object_velr_list = []
            object_rel_pos_list = []
            
            obj_pos = self.get_multiple_obj_pos(self.optitrack_subscriber.id_list)
            obj_rel_pos = obj_pos - ee_pos[None, :]
            # NOTE : e.g. 20Hz인데 20보다 작으면 dt가 부정확할수도(즉 0.05나와야하는데 실제론 0.06일수도)
            dt = self.dt
            if dt==0.1 and self.custom_processed_vel: # For matching sim policy inputs
                print('custom process vel is used!')
                dt = 0.05
                
                
            # print('In get_obs, dt : ', dt)
            # for i in range(self.num_objects):
            #     obj_pos = self.get_multiple_obj_pos(self.optitrack_subscriber.optitrack_id_list)

            #     obj_rot = rotations.mat2euler(self.sim.data.get_site_xmat('objSite_'+str(i)))
            #     # velocities
            #     dt = self.sim.nsubsteps * self.sim.model.opt.timestep # same as self.dt
            #     obj_velp = self.sim.data.get_site_xvelp('objSite_'+str(i)) * dt
            #     obj_velr = self.sim.data.get_site_xvelr('objSite_'+str(i)) * dt
            #     obj_rel_pos = obj_pos - ee_pos

            #     object_pos_list.append(obj_pos)
            #     object_rot_list.append(obj_rot)
            #     object_velp_list.append(obj_velp)
            #     object_velr_list.append(obj_velr)
            #     object_rel_pos_list.append(obj_rel_pos)
            
            # obj_pos = np.stack(object_pos_list, axis =0) #[num_obj, dim]
            # obj_rot = np.stack(object_rot_list, axis =0) #[num_obj, dim]
            # obj_velp = np.stack(object_velp_list, axis =0) #[num_obj, dim]
            # obj_velr = np.stack(object_velr_list, axis =0) #[num_obj, dim]
            # obj_rel_pos = np.stack(object_rel_pos_list, axis =0) #[num_obj, dim]
        else :
            raise NotImplementedError            
        if self.sequential_rl:
            self.object_pos = obj_pos.copy()

        gripper_state = gripperpos


        if self.observation_type=='joint_q':
            obs = np.concatenate([qpos, qvel, ee_pos, obj_pos])
        elif self.observation_type == 'ee_object_pos':
            obs = np.concatenate([ee_pos, obj_pos, obj_rel_pos])
        elif self.observation_type == 'ee_object_pos_w_grip_custom_vel' or self.observation_type=='ee_object_pos_w_grip' or self.observation_type=='ee_object_pos_w_custom_vel':
            if self.previous_ee_pos is None:
                ee_velp = np.zeros_like(ee_pos)
            else:
                if self.custom_processed_vel:
                    ee_velp = (ee_pos - self.previous_ee_pos)/dt*0
                else:
                    ee_velp = (ee_pos - self.previous_ee_pos)/dt                


            if self.previous_obj_pos is None:
                obj_velp = np.zeros_like(obj_pos)
            else:
                if self.custom_processed_vel:
                    obj_velp = (obj_pos - self.previous_obj_pos)/dt*0
                else:
                    obj_velp = (obj_pos - self.previous_obj_pos)/dt

            self.previous_ee_pos = ee_pos.copy()
            self.previous_obj_pos = obj_pos.copy()

            if self.observation_type=='ee_object_pos_w_grip_custom_vel':
                obs = np.concatenate([
                    ee_pos, obj_pos.ravel(), obj_rel_pos.ravel(), gripper_state, 
                    obj_velp.ravel(), ee_velp
                ])
            elif self.observation_type=='ee_object_pos_w_custom_vel':
                obs = np.concatenate([
                    ee_pos, obj_pos.ravel(), obj_rel_pos.ravel(), obj_velp.ravel(), ee_velp
                ])
            else:
                obs = np.concatenate([ee_pos, obj_pos.ravel(), obj_rel_pos.ravel(), gripper_state])

        elif self.observation_type == 'ee_object_all':
            raise NotImplementedError()                
            obs = np.concatenate([ee_pos, obj_pos.ravel(), obj_rel_pos.ravel(), gripper_state, obj_rot.ravel(), obj_velp.ravel(), obj_velr.ravel(), ee_velp, gripper_vel], axis =-1)

        if self.full_state_goal:
            achieved_goal = obs
        else :
            if not self.has_object:
                raise NotImplementedError()                
            else:
                print('object idx : ', self.goal_object_idx)
                achieved_goal = obj_pos[self.goal_object_idx]
        
        if self.sequential_rl:
            if self._state_goal is None:
                self._state_goal = np.concatenate(self.predefined_sequential_goal_list)
            desired_goal = np.reshape(self._state_goal.copy(), (self.num_objects, -1)) #[num_obj, dim]
            desired_goal = desired_goal[self.goal_object_idx]            
        else:
            if self._state_goal is None:
                self._state_goal = np.zeros(3)
            desired_goal = self._state_goal.copy()


        return {
            'observation' : obs.copy(),
            'achieved_goal' : achieved_goal.copy(),
            'desired_goal' : desired_goal.copy(), 
        }    

    def sample_goal(self, full_state_goal):
        if full_state_goal:
            raise NotImplementedError
        else:
            if self.sequential_rl:
                goal = np.concatenate(self.predefined_sequential_goal_list)
                print(goal)
                # time.sleep(3)
            else:
                goal = super().sample_goal(full_state_goal)
                if self.goal_weight_is_set:
                    self.goal_object_idx = np.argmax(self.goal_weight)
                    self.goal_weight_is_set = False
                else:
                    self.goal_object_idx = np.random.randint(self.num_objects)
            return goal
    def set_goal_weight(self, goal_weight):
        self.goal_weight = goal_weight
        self.goal_weight_is_set = True


    def get_weight_for_multigoal_rl(self):
        # obs_dict = self._get_obs()
        # observation = obs_dict['observation']
        
        object_pos = self.object_pos.copy()
                
        # Assume object_pos : [num_obj, dim]
        desired_goal = self._state_goal.copy() #[num_obj*dim]
        desired_goal = np.reshape(desired_goal, (self.num_objects, -1)) #[num_obj, dim]
        
        # 첫번째부터 순서대로 desired goal에 도달했는지 보고서 도달안한 obj 중 제일 앞순서대로 w 설정하게끔.
        # 즉, Multi Goal HRL에서 w를 내뱉는 high level policy를 naive하게 대체하는 역할
        weight = np.zeros(self.weight_dim)
        for idx, obj_pos, goal in zip(range(self.num_objects), object_pos, desired_goal):
            d = np.linalg.norm(obj_pos-goal, axis =-1)            
            if (d > self.distance_threshold):
                weight[idx] = 1.0
                self.goal_object_idx = idx
                break
        if (weight==np.zeros(self.weight_dim)).all(): # all goals are achieved
            weight = np.zeros(self.weight_dim)
            # 그냥 일단은 마지막 object기준으로 남겨두기
            idx = self.num_objects-1
            weight[idx] = 1.0
            self.goal_object_idx = idx
            # weight = None

        return weight

class DSCHOSingleUR3PickAndPlaceRealEnv(DSCHOSingleUR3GoalRealEnv):
    def __init__(self, *args, **kwargs):        
        super().__init__(has_object=True, block_gripper=False, task='pickandplace', *args, **kwargs)

class DSCHOSingleUR3PushRealEnv(DSCHOSingleUR3GoalRealEnv):
    def __init__(self, *args, **kwargs):
        
        super().__init__(has_object=True, block_gripper=True, task='push',*args, **kwargs)

class DSCHOSingleUR3ReachRealEnv(DSCHOSingleUR3GoalRealEnv):
    def __init__(self, *args, **kwargs):
        print('@@@@@@@ Currently debugging reach env where has_object is True!')
        super().__init__(has_object=True, block_gripper=True, task='reach', *args, **kwargs)

class DSCHOSingleUR3AssembleRealEnv(DSCHOSingleUR3GoalRealEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(has_object=True, block_gripper=False, task='assemble', *args, **kwargs)

class DSCHOSingleUR3DrawerOpenRealEnv(DSCHOSingleUR3GoalRealEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(has_object=True, block_gripper=False, task='drawer_open', *args, **kwargs)


# dscho added for ARL
class DSCHOSingleUR3PegRealEnv(DSCHOSingleUR3GoalRealEnv):
    def __init__(self, *args, **kwargs):        
        # assert kwargs.get('so3_constraint')=='vertical_side-180'
        # assert kwargs.get('so3_constraint')=='vertical_front'
        assert kwargs.get('so3_constraint')=='vertical_front-180'
        super().__init__(has_object=False, block_gripper=True, task='peg', *args, **kwargs)

class DSCHOSingleUR3SweepRealEnv(DSCHOSingleUR3GoalRealEnv):
    def __init__(self, *args, **kwargs):
        assert kwargs.get('so3_constraint')=='vertical_front-180'
        super().__init__(has_object=False, block_gripper=True, task='sweep', *args, **kwargs)

class DSCHOSingleUR3CoveringRealEnv(DSCHOSingleUR3GoalRealEnv):
    def __init__(self, *args, **kwargs):
        # assert kwargs.get('so3_constraint')=='vertical_front-180'
        assert kwargs.get('so3_constraint')=='vertical_front-180'
        super().__init__(has_object=False, block_gripper=True, task='covering', *args, **kwargs)


class DSCHOSingleUR3ImagePickAndPlaceRealEnv(DSCHOSingleUR3GoalRealEnv):
    def __init__(self, *args, **kwargs):
        # assert kwargs.get('so3_constraint')=='vertical_front-180'
        assert kwargs.get('so3_constraint')=='vertical_side-180'
        # since 
        super().__init__(has_object=False, block_gripper=False, task='image_pickandplace', *args, **kwargs)

class DSCHOSingleUR3MOKARealEnv(DSCHOSingleUR3GoalRealEnv):
    def __init__(self, *args, **kwargs):
        # assert kwargs.get('so3_constraint')=='vertical_front-180'
        # since 
        super().__init__(has_object=False, block_gripper=False, task='moka', *args, **kwargs)


class DSCHOSingleUR3RealEnv(DSCHOSingleUR3GoalRealEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(has_object=False, block_gripper=False, task=None, *args, **kwargs)



















def get_default_env_kwargs():
    # host_ip_right = '192.168.5.102'
    host_ip_left = '192.168.5.101'
    # host_ip_right = '192.168.2.4' # assume optitrack wifi
    env_kwargs = dict(
        host_ip=host_ip_left, # But which hand should be right in current simulation xml setting! (In sim, only right arm is used as upright)
        # host_ip=host_ip_left, # But which hand should be right in current simulation xml setting! (In sim, only right arm is used as upright)
        rate=10, # 너무 느리면 20Hz정도까지 늘려보기                  
        #################### env kwargs
        sparse_reward = True,               
        trigonometry_observation = False,         
        full_state_goal = False,
        reward_by_ee = False,                                 
        reward_success_criterion='ee_pos',
        distance_threshold = 0.05,
        # initMode='vertical',
        which_hand='right', 
        so3_constraint ='vertical_side-180',
        # task = 'pickandplace',
        observation_type='ee_object_pos_w_grip_custom_vel', #'joint_q', #'ee_object_object', #'ee_object_all'
        init_qpos_type = 'upright',
        predefined_goal = None,
    )
    return env_kwargs

def get_default_wrapper_kwargs(env):
    # real은 PID, scale factor등은 필요가없음(HW built in이니까)
    return dict(env=env,
                gripper_action = True,
                g_control_type='move_gripper_position',
                q_control_type= 'speedj',
                action_downscale=0.01, # something wrong... (waving motion) -> servoj? or smaller scale or lower Hz?
                speedj_args = {'a': 5, 't': 2/env.rate._freq, 'wait': False},
                servoj_args = {'t': 2/env.rate._freq, 'wait': False},
                movej_args = {'t': 2/env.rate._freq, 'a' : 1.0, 'v' : 0.5, 'wait': False},
                g_control_args = {'wait' : False},
                so3_constraint='vertical_side-180', 
                )

def test_single_ur3_real():

    env_kwargs = get_default_env_kwargs()
    rate = 10
    env_kwargs.update(dict(rate = rate))

    from gym_custom.envs.real.optitrack_subscriber import OptitrackSubscriber
    env_kwargs.update({'optitrack_subscriber' : OptitrackSubscriber(rate=rate, id_list = ['ur3_table', 'block_5cm_red'])})
    time.sleep(1) # wait for optitrack init
    env = DSCHOSingleUR3PickAndPlaceRealEnv(**env_kwargs)
    # time.sleep(1) # wait for optitrack inittest_single_ur3_real
    # env.set_initial_joint_pos('current')
    # env.set_initial_gripper_pos('current')
    
    # NOTE : action downscale : 0.001, 50Hz -> maximum 5cm per second
    wrapper_kwargs = get_default_wrapper_kwargs(env)
    q_control_type_list = ['speedj', 'servoj']
    q_control_type = q_control_type_list[0]
    if q_control_type=='speedj': # tested with rate 10, action downscale 0.01 is stable, 0.015 is so so.. / rate 15, action downscale 0.01 (sometimes break)
        wrapper_kwargs.update({'action_downscale' : 0.01, 'speedj_args' : {'a': 5, 't': 2/env.rate._freq, 'wait': False}, 
                                'multi_step' : 1, 'q_control_type' : 'speedj',
                                })     
    elif q_control_type=='servoj': # tested with rate 10, action downscale 0.03 is stable, 0.05 is so so.. & slower than speedj
        wrapper_kwargs.update({'action_downscale' : 0.03, 'servoj_args' : {'t': 2/env.rate._freq, 'wait': False},
                                'multi_step' : 1, 'q_control_type' : 'servoj',  
                                })     

    env = EndEffectorPositionControlSingleWrapperReal(**wrapper_kwargs)
    
    
    print('env init done')
    # if prompt_yes_or_no('reset model would use movej to init qpos, where endeffector pos is \r\n right: %s \r\n left: %s \r\n?'
    #     %(self.get_endeff_pos('right'), self.get_endeff_pos('left'))) is False:
    #     print('exiting program!')
    #     sys.exit()

    # NOTE : Should check initial qpos & end effector pos

    obs = env.reset()
    print('reset, obs : {}'.format(obs))
    if prompt_yes_or_no('moving to x axis, while closing gripper?') is False:
        print('exiting program!')
        sys.exit()
    # if env.rate._freq==20 :
    #     num_iter = 20 # 20Hz : 1s 
    # elif env.rate._freq==50 :
    #     num_iter = 50 # 50Hz : 1s 
    num_iter = env.rate._freq*2
    for i in range(num_iter): 
        action = np.array([1.0, 0, 0, 1.0]) # moving x axis, closing gripper
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        ee_pos, obj_pos, obj_rel_pos, gripper_state, obj_velp, ee_velp, _ = np.split(obs['observation'], [3,6,9,11,14,17], axis =-1)
        print('ee pos : {} obj pos : {}, gripper state : {}, obj vel : {}, ee vel : {}'.format(ee_pos, obj_pos, gripper_state, obj_velp, ee_velp))
    time.sleep(3)

    if prompt_yes_or_no('moving to -x axis, while opening gripper?') is False:
        print('exiting program!')
        sys.exit()
    for i in range(num_iter):
        action = np.array([-1.0, 0, 0, -1.0]) # moving x axis, opening gripper
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        ee_pos, obj_pos, obj_rel_pos, gripper_state, obj_velp, ee_velp, _ = np.split(obs['observation'], [3,6,9,11,14,17], axis =-1)
        print('ee pos : {} obj pos : {}, gripper state : {}, obj vel : {}, ee vel : {}'.format(ee_pos, obj_pos, gripper_state, obj_velp, ee_velp))
    time.sleep(3)
    
    print('done')




def test_single_ur3_real_for_calibration_optitrack():
    rate = 15
    env_kwargs = get_default_env_kwargs()
    env_kwargs.update(dict(rate = rate))
    from gym_custom.envs.real.optitrack_subscriber import OptitrackSubscriber
    env_kwargs.update({'optitrack_subscriber' : OptitrackSubscriber(rate=rate, id_list = ['ur3_table', 'block_5cm_red'])})
    time.sleep(1)
    env = DSCHOSingleUR3PickAndPlaceRealEnv(**env_kwargs)
    env.set_initial_joint_pos('current')
    env.set_initial_gripper_pos('current')
    wrapper_kwargs = get_default_wrapper_kwargs(env)     
    env = EndEffectorPositionControlSingleWrapperReal(**wrapper_kwargs)
    
    
    print('done')
    # if prompt_yes_or_no('reset model would use movej to init qpos, where endeffector pos is \r\n right: %s \r\n left: %s \r\n?'
    #     %(self.get_endeff_pos('right'), self.get_endeff_pos('left'))) is False:
    #     print('exiting program!')
    #     sys.exit()

    # NOTE : Should check initial qpos & end effector pos

    obs = env.reset()
    print('reset, obs : {}'.format(obs))
    if prompt_yes_or_no('prepared with freedrive mode for optitrack calibration?') is False:
        print('exiting program!')
        sys.exit()

    for i in range(50):
        obs = env.get_calibration_info(name='block_5cm_red')
        # ee_pos, obj_pos, obj_rel_pos, gripper_state, obj_velp, ee_velp, _ = np.split(obs['observation'], [3,6,9,11,14,17], axis =-1)
        ee_pos, obj_pos, _ = np.split(obs, [3,6], axis =-1)
        print('ee pos : {} obj pos : {}'.format(ee_pos, obj_pos))
        time.sleep(1)

    # for i in range(10): # 20Hz 기준 0.5초
    #     action = np.array([1.0, 0, 0, 1.0]) # moving x axis, closing gripper
    #     next_obs, reward, done, info = env.step(action)
    #     obs = next_obs
    #     ee_pos, obj_pos, obj_rel_pos, gripper_state, obj_velp, ee_velp, _ = np.split(obs['observation'], [3,6,9,11,14,17], axis =-1)
    #     print('ee pos : {} obj pos : {}, gripper state : {}, obj vel : {}, ee vel : {}'.format(ee_pos, obj_pos, gripper_state, obj_velp, ee_velp))
    # time.sleep(3)

    # if prompt_yes_or_no('moving to -x axis, while opening gripper?') is False:
    #     print('exiting program!')
    #     sys.exit()
    # for i in range(10):
    #     action = np.array([-1.0, 0, 0, -1.0]) # moving x axis, opening gripper
    #     next_obs, reward, done, info = env.step(action)
    #     obs = next_obs
    #     ee_pos, obj_pos, obj_rel_pos, gripper_state, obj_velp, ee_velp, _ = np.split(obs['observation'], [3,6,9,11,14,17], axis =-1)
    #     print('ee pos : {} obj pos : {}, gripper state : {}, obj vel : {}, ee vel : {}'.format(ee_pos, obj_pos, gripper_state, obj_velp, ee_velp))
    # time.sleep(3)
    
    print('done')



def test_single_ur3_real_pickandplace():

    env_kwargs = get_default_env_kwargs()
    rate = 10
    env_kwargs.update(dict(rate = rate))
    from gym_custom.envs.real.optitrack_subscriber import OptitrackSubscriber
    env_kwargs.update({'optitrack_subscriber' : OptitrackSubscriber(rate=rate, id_list = ['ur3_table', 'block_5cm_red'])})
    time.sleep(1)
    env = DSCHOSingleUR3PickAndPlaceRealEnv(**env_kwargs)
        
    wrapper_kwargs = get_default_wrapper_kwargs(env)     
    wrapper_kwargs.update({'action_downscale' : 0.01, 'speedj_args' : {'a': 5, 't': 2/env.rate._freq, 'wait': False}, 
                                'multi_step' : 1, 'q_control_type' : 'speedj',
                                })    
    env = EndEffectorPositionControlSingleWrapperReal(**wrapper_kwargs)
    
    
    print('done')
    # if prompt_yes_or_no('reset model would use movej to init qpos, where endeffector pos is \r\n right: %s \r\n left: %s \r\n?'
    #     %(self.get_endeff_pos('right'), self.get_endeff_pos('left'))) is False:
    #     print('exiting program!')
    #     sys.exit()

    # NOTE : Should check initial qpos & end effector pos


    obs = env.reset()
    print('reset, obs : {}'.format(obs))
    
    print('NOTE : You\'d better use BaselineHERGoalEnvWrapper for conveniency in GCRL')
    time.sleep(1)
    pure_obs = obs['observation']
    achieved_goal = obs['achieved_goal']
    # desired_goal = obs['desired_goal']
    # desired_goal = achieved_goal + np.array([0,0, 0.1])
    desired_goal = np.array([-0.1, -0.3, 0.78])
    ee_pos = pure_obs[:3]
    obj_pos = pure_obs[3:6]
    assert (obj_pos==achieved_goal).all()
    use_sleep = False
    duration = 4.0  #4초씩 움직임
    dt = env.dt
    act_scale = 10
    for i in range(5):
        for t in range(int(duration/dt)):
            if i ==0 :
                action_xyz = np.tanh(act_scale*(obj_pos+np.array([0,0,0.1])- ee_pos))
                action = np.concatenate([action_xyz, np.array([-1.0])], axis =-1)
            elif i ==1: 
                action_xyz = np.tanh(act_scale*(obj_pos - ee_pos))
                action = np.concatenate([action_xyz, np.array([-1.0])], axis =-1)
            elif i ==2: 
                action_xyz = np.tanh(act_scale*(obj_pos - ee_pos))
                action = np.concatenate([action_xyz, np.array([1.0])], axis =-1)
            elif i ==3: 
                action_xyz = np.tanh(act_scale*(desired_goal - ee_pos))
                action = np.concatenate([action_xyz, np.array([1.0])], axis =-1)
            elif i ==4: 
                action_xyz = np.tanh(act_scale*(desired_goal +np.array([0, 0, 0.1])- ee_pos))
                action = np.concatenate([action_xyz, np.array([-1.0])], axis =-1)

            if ee_pos[2] < 0.77:
                action[2] = np.clip(action[2], 0, 1) # to prevent collision with table

            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            pure_obs = obs['observation']
            ee_pos = pure_obs[:3]
            obj_pos = pure_obs[3:6]
            print('step : {} act : {} ee pos ; {} obj pos : {} goal : {}'.format(t, action, ee_pos, obj_pos, desired_goal))
        
        if use_sleep:
            time.sleep(1)
    

    time.sleep(1)
    
    print('done')



def test_single_ur3_real_peg():

    env_kwargs = get_default_env_kwargs()
    rate = 20
    env_kwargs.update(dict(rate = rate,
                           so3_constraint='vertical_side-180',
                           auto_calibrate=False,
                           ))
    
    env = DSCHOSingleUR3PegRealEnv(**env_kwargs)
        
    wrapper_kwargs = get_default_wrapper_kwargs(env)     
    wrapper_kwargs.update(dict(so3_constraint='vertical_side-180'))
    wrapper_kwargs.update({'action_downscale' : 0.005, 'speedj_args' : {'a': 5, 't': 2/env.rate._freq, 'wait': False}, 
                                'multi_step' : 1, 'q_control_type' : 'speedj',
                                'gripper_action' : False,
                                })    
    env = EndEffectorPositionControlSingleWrapperReal(**wrapper_kwargs)
    
    
    print('done')

    obs = env.reset()
    print('reset, obs : {}'.format(obs))
    

    if prompt_yes_or_no('Run goal reaching behavior?') is False:
        print('exiting program!')
        sys.exit()

    pure_obs = obs['observation']
    achieved_goal = obs['achieved_goal']
    
    # desired_goal = obs['desired_goal']
    desired_goal = achieved_goal + np.array([0,0,0.05])
    # desired_goal = np.array([-0.1, -0.3, 0.78])
    ee_pos = pure_obs[:3]
    use_sleep = False
    duration = 4.0  #4초씩 움직임
    dt = env.dt
    act_scale = 10
    for i in range(2):
        for t in range(int(duration/dt)):
            
            action_xyz = np.tanh(act_scale*(desired_goal - ee_pos))
            # (20240507) NOTE: only -1~-0.8, 0.8~1.0 works. Intermediate values are meaningless (do not know reason..)
            if i ==0:
                action = np.concatenate([action_xyz, np.array([1.0])], axis =-1)
            elif i==1:
                action = np.concatenate([action_xyz, np.array([-1.0])], axis =-1)
            
            if ee_pos[2] < 0.75:
                action[2] = np.clip(action[2], 0, 1) # to prevent collision with table

            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            pure_obs = obs['observation']
            ee_pos = pure_obs[:3]
            # obj_pos = pure_obs[3:6]
            print('step : {} act : {} ee pos ; {}  goal : {}'.format(t, action, ee_pos, desired_goal))
        
        if use_sleep:
            time.sleep(1)
    

    time.sleep(1)
    
    print('done')


def test_single_ur3_real_sweep():

    env_kwargs = get_default_env_kwargs()
    rate = 10
    env_kwargs.update(dict(rate = rate,
                           so3_constraint='vertical_front',
                           auto_calibrate=False,
                           ))
    env = DSCHOSingleUR3SweepRealEnv(**env_kwargs)
        
    wrapper_kwargs = get_default_wrapper_kwargs(env)     
    wrapper_kwargs.update(dict(so3_constraint='vertical_front'))
    wrapper_kwargs.update({'action_downscale' : 0.005, 'speedj_args' : {'a': 5, 't': 2/env.rate._freq, 'wait': False}, 
                                'multi_step' : 1, 'q_control_type' : 'speedj',
                                'gripper_action' : False,
                                })    
    env = EndEffectorPositionControlSingleWrapperReal(**wrapper_kwargs)
    
    
    print('done')

    obs = env.reset()
    print('reset, obs : {}'.format(obs))
    

    if prompt_yes_or_no('Run goal reaching behavior?') is False:
        print('exiting program!')
        sys.exit()

    pure_obs = obs['observation']
    achieved_goal = obs['achieved_goal']
    
    desired_goal = obs['desired_goal']
    # desired_goal = achieved_goal + np.array([0,0,0.05])
    # desired_goal = np.array([-0.1, -0.3, 0.78])
    ee_pos = pure_obs[:3]
    use_sleep = False
    duration = 4.0  #4초씩 움직임
    dt = env.dt
    act_scale = 10
    for i in range(2):
        for t in range(int(duration/dt)):
            
            action_xyz = np.tanh(act_scale*(desired_goal - ee_pos))
            # (20240507) NOTE: only -1~-0.8, 0.8~1.0 works. Intermediate values are meaningless (do not know reason..)
            if i ==0:
                action = np.concatenate([action_xyz, np.array([1.0])], axis =-1)
            elif i==1:
                action = np.concatenate([action_xyz, np.array([-1.0])], axis =-1)
            
            if ee_pos[2] < 0.77:
                action[2] = np.clip(action[2], 0, 1) # to prevent collision with table

            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            pure_obs = obs['observation']
            ee_pos = pure_obs[:3]
            # obj_pos = pure_obs[3:6]
            print('step : {} act : {} ee pos ; {}  goal : {}'.format(t, action, ee_pos, desired_goal))
        
        if use_sleep:
            time.sleep(1)
    

    time.sleep(1)
    
    print('done')

def zed_render(zed, runtime_parameters, image, height, width):
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # A new image is available if grab() returns SUCCESS
        zed.retrieve_image(image, sl.VIEW.LEFT)
                
        raw_data = image.get_data()
        # Only for PickAndPlace
        # raw_data = raw_data[100:, 250:550]
        raw_data = raw_data[100:, 200:500] # left arm 241106 (after rearrangement of robots in ASRI)(not yet implemented)
    
        cv2.imshow("ZED", raw_data)
        cv2.waitKey(1)
        
        img = cv2.cvtColor(raw_data, cv2.COLOR_BGRA2RGB)
        img = cv2.resize(img, (height, width))
        assert img.shape==(height, width, 3)    
        return img
    else:
        raise NotImplementedError



def test_single_ur3_real_se3_calibration():
    # task='peg'
    # task='sweep'
    # task='covering'
    # task='image_pickandplace'
    task='moka'

    if task=='peg':
        so3_constraint ='vertical_side-180'
        rotation_z = False
        observation_type='ee_object_pos_w_grip_custom_vel'
    elif task in ['sweep','covering', 'image_pickandplace']:
        so3_constraint ='vertical_front-180'
        rotation_z = False
        observation_type='ee_object_pos_w_grip_custom_vel'
    elif task in ['moka']:
        so3_constraint ='vertical_side-180'
        rotation_z = True
        observation_type='ee_object_pos_rot_w_grip_custom_vel'
    env_kwargs = get_default_env_kwargs()
    rate = 10
    

    env_kwargs.update(dict(rate = rate,
                           so3_constraint =so3_constraint,
                           reset_at_goal = False,
                           auto_calibrate = True, # False,
                           observation_type=observation_type,
                           ))
    
    if task=='peg':
        env = DSCHOSingleUR3PegRealEnv(**env_kwargs)
    elif task=='sweep':
        env = DSCHOSingleUR3SweepRealEnv(**env_kwargs)
    elif task=='covering':
        env = DSCHOSingleUR3CoveringRealEnv(**env_kwargs)
    elif task=='image_pickandplace':
        env = DSCHOSingleUR3ImagePickAndPlaceRealEnv(**env_kwargs)
    elif task=='moka':
        env = DSCHOSingleUR3MOKARealEnv(**env_kwargs)
    wrapper_kwargs = get_default_wrapper_kwargs(env)         
    wrapper_kwargs.update({'action_downscale' : 0.01, 'speedj_args' : {'a': 5, 't': 2/env.rate._freq, 'wait': False}, 
                                'multi_step' : 1, 'q_control_type' : 'speedj',
                                'so3_constraint' : so3_constraint,
                                'rotation_z' : rotation_z,
                                })    
    env = EndEffectorPositionControlSingleWrapperReal(**wrapper_kwargs)
    print(env.get_obs_dict()['qpos'])    
    obs = env.reset()
    print('done')

    print(f"endeff pos: {env.get_endeff_pos(arm='right')}")
    
    # Move robot to desired position
    while True:
        current_pos = env.get_endeff_pos(arm='right')
        delta_pos_x = 0.2 - current_pos[0]

        if np.linalg.norm(delta_pos_x) < 0.05:  # A small threshold to check if the position is reached
            print(f"Reached desired position: {current_pos}")
            break

        # Create action to move towards the desired position
        action = np.zeros(5)  # Assuming the action has 5 dimensions: [dx, dy, dz, d_rot_z, gripper]
        action[0] = 0.5

        for _ in range(5):
            env.step(action)

    while True:
        current_pos = env.get_endeff_pos(arm='right')
        delta_pos_z = 0.72 - current_pos[2]

        if np.linalg.norm(delta_pos_z) < 0.04:  # A small threshold to check if the position is reached
            print(f"Reached desired position: {current_pos}")
            break

        # Create action to move towards the desired position
        action = np.zeros(5)  # Assuming the action has 5 dimensions: [dx, dy, dz, d_rot_z, gripper]
        action[2] = -0.5

        for _ in range(5):
            env.step(action)

    print(f"Final endeff pos: {env.get_endeff_pos(arm='right')}")


    while True:
        if task=='moka':
            # if prompt_yes_or_no("Press 'Y' to move gripper. Press 'n' to terminate program.") is False:
            #     print('exiting program!')
            #     sys.exit()
            env.interface.move_gripper_position(g=10.0, wait=True)
            q = env.get_obs_dict()['qpos']
            R, p, T = env.forward_kinematics_ee(q=q, arm='right')

            time.sleep(1)
            env.interface.move_gripper_position(g=255.0, wait=True)
            q = env.get_obs_dict()['qpos']
            R, p, T = env.forward_kinematics_ee(q=q, arm='right')
    
        else:
            env.step(np.array([0,0,0,0]))

    obs = env.reset()
    print('reset, obs : {}'.format(obs))
    
#==================================================
    # while True:
    # # for i in range(3):
    #     q = env.get_obs_dict()['qpos']
    #     R, p, T = env.forward_kinematics_ee(q=q, arm='right')
    #     obs = env._get_obs()
    #     print(obs)
    #     # print(R)
    #     # print(rotations.mat2euler(R))
    #     # print(f'{p} {q}')
    #     time.sleep(0.1)
    
def test_single_ur3_real_se3_calibration_dscho_custom():
    task='peg'
    # task='sweep'
    # task='covering'
    # task='image_pickandplace'
    # task='moka'

    if task=='peg':
        so3_constraint ='vertical_front-180' # right arm (after rearrange ASRI)
        rotation_z = False
        observation_type='ee_object_pos_w_grip_custom_vel'
    elif task == 'covering':
        so3_constraint ='vertical_front-180' # right arm (after rearrange ASRI)
        rotation_z = False
        observation_type='ee_object_pos_w_grip_custom_vel'
    elif task in ['sweep','image_pickandplace']:
        so3_constraint ='vertical_side-180'
        rotation_z = False
        observation_type='ee_object_pos_w_grip_custom_vel'
    elif task in ['moka']:
        so3_constraint ='vertical_side-180'
        rotation_z = True
        observation_type='ee_object_pos_rot_w_grip_custom_vel'
    env_kwargs = get_default_env_kwargs()
    
    rate = 10
    

    env_kwargs.update(dict(rate = rate,
                           so3_constraint =so3_constraint,
                           reset_at_goal = False,
                           auto_calibrate = True, # False,
                           observation_type=observation_type,
                           ))
    
    if task=='peg':
        host_ip_right = '192.168.5.102'
        env_kwargs.update({'host_ip' : host_ip_right,
                           'auto_calibrate' : False,
                           })
                           
        env = DSCHOSingleUR3PegRealEnv(**env_kwargs)
    elif task=='sweep':
        env = DSCHOSingleUR3SweepRealEnv(**env_kwargs)
    elif task=='covering':
        host_ip_right = '192.168.5.102'
        env_kwargs.update({'host_ip' : host_ip_right,
                           'auto_calibrate' : False,
                           })
        env = DSCHOSingleUR3CoveringRealEnv(**env_kwargs)
    elif task=='image_pickandplace':
        host_ip_left = '192.168.5.101'
        env_kwargs.update({'host_ip' : host_ip_left})
        env = DSCHOSingleUR3ImagePickAndPlaceRealEnv(**env_kwargs)
    elif task=='moka':
        env = DSCHOSingleUR3MOKARealEnv(**env_kwargs)
    wrapper_kwargs = get_default_wrapper_kwargs(env)         
    wrapper_kwargs.update({'action_downscale' : 0.01, 'speedj_args' : {'a': 5, 't': 2/env.rate._freq, 'wait': False}, 
                                'multi_step' : 1, 'q_control_type' : 'speedj',
                                'so3_constraint' : so3_constraint,
                                'rotation_z' : rotation_z,
                                })    
    env = EndEffectorPositionControlSingleWrapperReal(**wrapper_kwargs)
    print(env.get_obs_dict()['qpos'])    
    obs = env.reset()
    print('done')

    print(f"endeff pos: {env.get_endeff_pos(arm='right')}")
    
    # for i in range(100):
    #     env.step(np.array([0,0,0,1]))

    # obs = env.reset()
    # print('reset, obs : {}'.format(obs))
    
    # import numpy as np
    # from scipy.spatial.transform import Rotation as R
    # def euler_to_rmat(euler, degrees=False):
    #     return R.from_euler("xyz", euler, degrees=degrees).as_matrix()


    while True:
    # for i in range(3):
        q = env.get_obs_dict()['qpos']
        R, p, T = env.forward_kinematics_ee(q=q, arm='right')
        
        print(R)
        print(f'{p} {q}')
        time.sleep(0.1)
    
    
def zed_camera_streaming():
    
    height = width = 84
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    
    # for multiple camera case
    # cameras = sl.Camera.get_device_list()
    # print('cameras : ', cameras)
    # serial_1 = cameras[0].serial_number
    # serial_2 = cameras[1].serial_number
    # print(serial_1, serial_2)
    
    serial_1 = 13426 # camera 1
    serial_2 = 26236181 # camera 2
    init_params.set_from_serial_number(serial_2)



    # init_params.camera_resolution = sl.RESOLUTION.AUTO # Use HD720 opr HD1200 video mode, depending on camera type.
    init_params.camera_resolution = sl.RESOLUTION.VGA # Use HD720 opr HD1200 video mode, depending on camera type.
    init_params.camera_fps = 30  # Set fps at 30

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : "+repr(err)+". Exit program.")
        exit()
    
    # Set brightness (acceptable range is typically 0 to 8)
    brightness_value = 8 # max brightness
    res = zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, brightness_value)
    if res != sl.ERROR_CODE.SUCCESS:
        print("Failed to set brightness.")
    else:
        print(f"Brightness set to: {brightness_value}")

    # getDeviceList(), and provide their serial number for identification. Once you know which ZED should be opened, you can request a specific serial number with InitParameters.input.setFromSerialNumber(1010)
    
    image = sl.Mat()
    
    runtime_parameters = sl.RuntimeParameters()

    while True:
        img = zed_render(zed, runtime_parameters, image, height, width)

    
def zed_image_pixel_difference_test():
    import numpy as np
    from PIL import Image
    height = width = 84
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    # init_params.camera_resolution = sl.RESOLUTION.AUTO # Use HD720 opr HD1200 video mode, depending on camera type.
    init_params.camera_resolution = sl.RESOLUTION.VGA # Use HD720 opr HD1200 video mode, depending on camera type.
    init_params.camera_fps = 30  # Set fps at 30

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : "+repr(err)+". Exit program.")
        exit()

    
    image = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()

    
    img_1 = zed_render(zed, runtime_parameters, image, height, width)
    img_2 = zed_render(zed, runtime_parameters, image, height, width)
    
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY).astype(np.float32)
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gap_rgb = np.abs(img_1-img_2)
    max_v = gap_rgb.max()
    min_v = gap_rgb.min()
    gap_rgb = (gap_rgb - min_v)/ (max_v - min_v)*255
    # gap_gray = cv2.cvtColor(np.abs(img_1-img_2),cv2.COLOR_RGB2GRAY)

    Image.fromarray(img_1.astype(np.uint8)).save('./temp_eval_debug/img_1.png')
    Image.fromarray(img_2.astype(np.uint8)).save('./temp_eval_debug/img_2.png')
    Image.fromarray(gap_rgb.astype(np.uint8)).save('./temp_eval_debug/img_gap_rgb.png')
    # Image.fromarray(gap_gray).save('./temp_eval_debug/img_gap_gray.png')

def ur3_urscript_inverse_kinematics_test():
    import numpy as np
    from scipy.spatial.transform import Rotation

    def rmat_to_euler(rot_mat, degrees=False):
        euler = Rotation.from_matrix(rot_mat).as_euler("xyz", degrees=degrees)
        return euler
    
    

    

    from gym_custom.envs.real.dscho_ur3_real import get_default_env_kwargs, get_default_wrapper_kwargs, DSCHOSingleUR3RealEnv, EndEffectorSE3ControlSingleWrapperReal

    env_kwargs = get_default_env_kwargs()
    rate = 5 # 10
    host_ip_right = '192.168.5.102'
    
    env_kwargs.update(dict(rate = rate,
                           host_ip = host_ip_right,
                            so3_constraint= None, 
                           reset_at_goal=False, # reset_at_goal,
                           auto_calibrate=False,
                           predefined_goal=np.zeros(3), # dummy
                        ))
    
    env = DSCHOSingleUR3RealEnv(**env_kwargs)
        
    wrapper_kwargs = get_default_wrapper_kwargs(env)     
    
    
    # # # speedj
    # wrapper_kwargs.update({'action_downscale' : 0.005, 
    #                         'multi_step' : 1, 'q_control_type' : 'speedj', 'speedj_args': {'a': 0.1, 't': 2/env.rate._freq, 'wait': False},
    #                         'gripper_action' : False
    #                         })    
    
    # # servoj (non-blocking)
    # wrapper_kwargs.update({'action_downscale' : 0.01, # 0.03, 
    #                             'multi_step' : 1, 'q_control_type' : 'servoj', 'servoj_args' : {'t': 3, 'wait': False},
    #                             })    
    
    # movej (blocking)
    wrapper_kwargs.update({'action_downscale' : 0.01, # 0.03, y
                                'multi_step' : 1, 
                                'q_control_type' : 'movej', 
                                'movej_args': {'t':1, 'a' : 1.4, 'v' : 1.0, 'wait': False}, # 
                                # 'q_control_type' : 'movep', 
                                # 'movep_args': {'a' : 1.0, 'v' : 0.1, 'wait': False},
                                # 'q_control_type' : 'movel', 
                                # 'movel_args': {'t' : 5, 'a' : 1.0, 'v' : 0.3, 'wait': False},
                                })        
    
    env = EndEffectorSE3ControlSingleWrapperReal(**wrapper_kwargs)
    obs = env.reset()
    
    for i in range(3):
        print(f'{i}th action')

        # # temp for ur3 test
        T_w_gripper = np.array([[0, -1, 0, 0.35],
                                [-1, 0, 0, -0.1],
                                [0, 0, -1, 0.8+i*0.05],
                                [ 0.,          0.,          0. ,         1.        ]])

        
        print('T_w_gripper : ', T_w_gripper)


        R = T_w_gripper[:3, :3]
        p = T_w_gripper[:3, 3]
        euler = rmat_to_euler(R)
        gripper_act = -np.ones(1) # open
        action = np.concatenate([p, euler, gripper_act])
        env.step(action, wait=True)
    

    tcp_pose = env.interface.get_actual_tcp_pose(wait=True)
    print('tcp_pose : ', tcp_pose)
    



if __name__ == "__main__":
    # test_single_ur3_real()
    # test_single_ur3_real_for_calibration_optitrack()
    # test_single_ur3_real_pickandplace()
    # test_single_ur3_real_peg()
    # test_single_ur3_real_sweep()
    # test_single_ur3_real_se3_calibration()
    # test_single_ur3_real_se3_calibration_dscho_custom()
    # zed_camera_streaming()
    # zed_image_pixel_difference_test()
    ur3_urscript_inverse_kinematics_test()
    