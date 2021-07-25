import copy
import numpy as np
import os

from numpy.core.numeric import full
import gym_custom
from gym_custom.spaces import Box
from gym_custom import utils
# from gym_custom.envs.mujoco import MujocoEnv
from gym_custom.envs.real.ur3_env import UR3RealEnv
from gym_custom.envs.custom.ur_utils import URScriptWrapper, URScriptWrapper_DualUR3
from gym_custom import Wrapper
from gym_custom.envs.custom.ur_utils import SO3Constraint, UprightConstraint, NoConstraint
import tensorflow as tf
import pickle
import joblib
import time
import sys
# import mujoco_py
from gym_custom.envs.real.utils import prompt_yes_or_no
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
     
        self.gripper_action = gripper_action
        self.dt = self.env.dt

        self.speedj_args = speedj_args
        self.servoj_args = servoj_args
        self.g_control_args = g_control_args
       
        if gripper_action:
            self.act_low = act_low = np.array([-1, -1, -1, -1])
            self.act_high = act_high= np.array([1, 1, 1, 1])
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

    def gripper_action_scale(self, gripper_act):
        # Assume gripper act : [-1, 1]
        # Sim : -1 : fully open, 1 : fully closed
        # Real : 0 : fully open, 255 : fully closed
        gripper_act = gripper_act.copy()
        rescaled_act = (gripper_act+1)/(1-(-1)) # 0~1 rescale 
        rescaled_act = 255*rescaled_act # 0~255 rescale
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
            gripper_act = np.zeros(self.gripper_act_dim) #opened
        
        
        current_qpos = self.env.interface.get_joint_positions(wait=False)
        R, p, T = self.env.forward_kinematics_ee(current_qpos, self.env.which_hand)
        ee_pos = p
        
        desired_ee_pos = ee_pos + ur3_act
        
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


class DSCHOUR3RealEnv(UR3RealEnv):
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
        if self.init_qpos_type=='upright':
            default_right_qpos = np.array([[-90.0, -90.0, -90.0, -90.0, 90.0, 180.0]])*np.pi/180.0 #[num_candidate+1, qpos_dim]
        else:
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
        
        
    def _get_init_qpos(self):
        return self._init_qpos.copy()



    def step(self, action):
        action = action.copy()
        raise NotImplementedError('Currently, Not implemented for dual arm. We just overrided it in sigle arm env')
        
        

    def get_endeff_pos(self, arm, q = None, wait=False):
        if q is None:
            q= self.interface.get_joint_positions(wait=wait)
            
        else :
            pass
        R, p, T = self.forward_kinematics_ee(q, arm)
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
                initMode='vertical',
                which_hand='right', 
                so3_constraint ='vertical_side',
                has_object = False,
                block_gripper = False,                
                task = 'pickandplace',
                observation_type='joint_q', #'ee_object_object', #'ee_object_all'
                init_qpos_type = None,
                predefined_goal = None,
                *args,
                **kwargs
                ):
        self.save_init_params(locals())

        self.full_state_goal = full_state_goal        
        self.trigonometry_observation = trigonometry_observation
        
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
        
        self.block_gripper = block_gripper
        self.has_object = has_object
        

        self.previous_ee_pos = None
        self.previous_obj_pos = None
        
        self.reward_success_criterion = reward_success_criterion
        
        self.which_hand = which_hand
        super().__init__(initMode=initMode, 
                        # ur3_random_init=ur3_random_init, 
                        *args, 
                        **kwargs
                        )
        
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
            goal_obj_low = np.array([0.0, -0.45, 0.77])
            goal_obj_high = np.array([0.3, -0.3, 0.95])
        
        elif self.which_hand =='left':
            goal_obj_low = np.array([-0.3, -0.45, 0.77])
            goal_obj_high = np.array([0.0, -0.3, 0.95])
        
        if self.init_qpos_type == 'upright': # right 기준
            goal_obj_low = np.array([-0.15, -0.45, 0.77])
            goal_obj_high = np.array([0.15, -0.3, 0.95])

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
        if self.predefined_goal is not None:
            return self.predefined_goal

        if full_state_goal:
            raise NotImplementedError
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
                elif self.task in ['reach']:
                    goal_ee_pos = np.random.uniform(
                        self.goal_ee_pos_space.low,
                        self.goal_ee_pos_space.high,
                        size=(self.goal_ee_pos_space.low.size),
                    )
                    goal = goal_ee_pos
                else:
                    raise NotImplementedError

        return goal
    
    # def reset(self):
    #     self.sim.reset()
    #     ob = self.reset_model()
    #     return ob

    def reset_model(self):
        if prompt_yes_or_no('Resetting... Did you prepare the object setting?') is False:
            print('exiting program!')
            sys.exit()
        self._state_goal = self.sample_goal(full_state_goal = self.full_state_goal)
        
        # if self.has_object: # reach인 경우엔 필요x            
        #     if self.task in ['reach']:
        #         raise NotImplementedError
        #         # if debug_opt==1:
        #         #     pass
        #         # elif debug_opt==2:
        #         #     self._state_goal = object_pos.copy()
        #     else:
        #         while np.linalg.norm(object_pos - self._state_goal) < 0.05:
        #             self._state_goal = self.sample_goal(full_state_goal = self.full_state_goal)
            
        

        self.previous_ee_pos = None
        self.previous_obj_pos = None
        observation = super().reset_model()

        return observation
        
    # Only ur3 qpos,vel(not include gripper), object pos(achieved_goal), desired_goal
    def _get_obs(self):
        
        obs_dict = super.get_obs_dict()
        # qpos, qvel gripperpos, grippervel
        qpos = obs_dict['qpos']
        qvel = obs_dict['qvel']
        gripperpos = obs_dict['gripperpos']
        grippervel = obs_dict['grippervel']
        ee_pos = self.get_endeff_pos(arm=self.which_hand, q = qpos)

        
        if self.trigonometry_observation:
            qpos = np.concatenate([np.cos(qpos), np.sin(qpos)], axis = -1)

        # NOTE : e.g. 20Hz인데 20보다 작으면 dt가 부정확할수도(즉 0.05나와야하는데 실제론 0.06일수도)
        dt = self.dt

        if self.has_object:
            obj_pos = self.get_obj_pos(name='obj')
            obj_rot = rotations.quat2euler(self.get_obj_quat(name='obj'))
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
        elif self.observation_type == 'ee_object_pos_w_grip_custom_vel' or self.observation_type=='ee_object_pos_w_grip' or self.observation_type=='ee_object_pos_w_custom_vel':
            if self.previous_ee_pos is None:
                ee_velp = np.zeros_like(ee_pos)
            else:
                ee_velp = (ee_pos - self.previous_ee_pos)/dt                


            if self.previous_obj_pos is None:
                obj_velp = np.zeros_like(obj_pos)
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

    # def convert_goal_for_reward(self, goal): #needed for TDMWrapper
    #     if goal.ndim==1:
    #         if not self.full_state_goal:
    #             return goal
    #         elif self.reward_by_ee:
    #             return goal[-3:]
    #         else: #exclude qvel in reward computation in outer wrapper
    #             return np.concatenate([goal[:self.obs_nqpos], goal[-3:]], axis =-1)
    #     elif goal.ndim==2:
    #         if not self.full_state_goal:
    #             return goal
    #         elif self.reward_by_ee:
    #             return goal[:, -3:]
    #         else: #exclude qvel in reward computation in outer wrapper
    #             return np.concatenate([goal[:, :self.obs_nqpos], goal[:, -3:]], axis =-1)
    #     else :
    #         raise NotImplementedError

    # def convert_goal_for_reward_tf(self, goals): #needed for TDMWrapper
    #     #Caution : Assume batch data is given.
    #     if not self.full_state_goal:
    #         return goals
    #     elif self.reward_by_ee:
    #         return goals[:, -3:]
    #     else: #exclude qvel in reward computation in outer wrapper
    #         return tf.concat([goals[:, :self.obs_nqpos], goals[:, -3:]], axis =-1)

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
            if self.task in ['push']:
                # print('currently, dense push debug')
                ee_pos = info['right_ee_pos']
                from_ee_to_obj = np.linalg.norm(achieved_goal[-3:] - ee_pos[-3:])
                reward = -from_ee_to_obj -placingDist
            else:
                reward = -placingDist

        return reward


    # obj related
    def get_obj_pos(self, name=None):
        raise NotImplementedError('should be implemented with optitrack or vicon')

        if not self.has_object:
            raise NotImplementedError
        if name is None:
            return self.data.get_body_xpos('obj')
        else :
            return self.data.get_body_xpos(name) 

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
                *args,
                **kwargs
                ):
        self.save_init_params(locals())
        self.num_objects = num_objects
        self.goal_object_idx = 0
        self.goal_weight_is_set = False
        super().__init__(has_object=True,  block_gripper=False,  task='pickandplace', *args, **kwargs)
    
    def _get_obs(self):
        
        obs_dict = super.get_obs_dict()
        # qpos, qvel gripperpos, grippervel
        qpos = obs_dict['qpos']
        qvel = obs_dict['qvel']
        gripperpos = obs_dict['gripperpos']
        grippervel = obs_dict['grippervel']
        ee_pos = self.get_endeff_pos(arm=self.which_hand, q = qpos)
        
        if self.trigonometry_observation:
            qpos = np.concatenate([np.cos(qpos), np.sin(qpos)], axis = -1)
        
        if self.has_object:
            object_pos_list = []
            object_rot_list = []
            object_velp_list = []
            object_velr_list = []
            object_rel_pos_list = []
            
            for i in range(self.num_objects):
                obj_pos = self.get_obj_pos(name='obj_'+str(i))

                obj_rot = rotations.mat2euler(self.sim.data.get_site_xmat('objSite_'+str(i)))
                # velocities
                dt = self.sim.nsubsteps * self.sim.model.opt.timestep # same as self.dt
                obj_velp = self.sim.data.get_site_xvelp('objSite_'+str(i)) * dt
                obj_velr = self.sim.data.get_site_xvelr('objSite_'+str(i)) * dt
                obj_rel_pos = obj_pos - ee_pos

                object_pos_list.append(obj_pos)
                object_rot_list.append(obj_rot)
                object_velp_list.append(obj_velp)
                object_velr_list.append(obj_velr)
                object_rel_pos_list.append(obj_rel_pos)
            
            obj_pos = np.stack(object_pos_list, axis =0) #[num_obj, dim]
            obj_rot = np.stack(object_rot_list, axis =0) #[num_obj, dim]
            obj_velp = np.stack(object_velp_list, axis =0) #[num_obj, dim]
            obj_velr = np.stack(object_velr_list, axis =0) #[num_obj, dim]
            obj_rel_pos = np.stack(object_rel_pos_list, axis =0) #[num_obj, dim]



        else :
            raise NotImplementedError            
        gripper_state = gripperpos


        if self.observation_type=='joint_q':
            obs = np.concatenate([qpos, qvel, ee_pos, obj_pos])
        elif self.observation_type == 'ee_object_pos':
            obs = np.concatenate([ee_pos, obj_pos, obj_rel_pos])
        elif self.observation_type == 'ee_object_pos_w_grip_custom_vel' or self.observation_type=='ee_object_pos_w_grip' or self.observation_type=='ee_object_pos_w_custom_vel':
            if self.previous_ee_pos is None:
                ee_velp = np.zeros_like(ee_pos)
            else:
                ee_velp = (ee_pos - self.previous_ee_pos)/dt                


            if self.previous_obj_pos is None:
                obj_velp = np.zeros_like(obj_pos)
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
                achieved_goal = obj_pos[self.goal_object_idx]
            
            
        return {
            'observation' : obs.copy(),
            'achieved_goal' : achieved_goal.copy(),
            'desired_goal' : self._state_goal.copy(), 
        }    

    def sample_goal(self, full_state_goal):
        if full_state_goal:
            raise NotImplementedError
        else:
            goal = super().sample_goal(full_state_goal)
            if self.goal_weight_is_set:
                self.goal_object_idx = np.argmax(self.goal_weight)
                self.goal_weight_is_set = False
            else:
                self.goal_object_idx = np.random.randint(self.num_objects)

    def set_goal_weight(self, goal_weight):
        self.goal_weight = goal_weight
        self.goal_weight_is_set = True


    

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



def get_default_env_kwargs():
    host_ip_right = '192.168.5.102'
    host_ip_left = '192.168.5.101'
    env_kwargs = dict(
        host_ip=host_ip_left, # But which hand should be right in current simulation xml setting! (In sim, only right arm is used as upright)
        rate=20, # 너무 느리면 25Hz정도까지 늘려보기                  
        #################### env kwargs
        sparse_reward = False,               
        trigonometry_observation = False,         
        full_state_goal = False,
        reward_by_ee = False,                                 
        reward_success_criterion='ee_pos',
        distance_threshold = 0.05,
        initMode='vertical',
        which_hand='right', 
        so3_constraint ='vertical_side',
        task = 'pickandplace',
        observation_type='ee_object_pos_w_grip_custom_vel', #'joint_q', #'ee_object_object', #'ee_object_all'
        init_qpos_type = 'upright',
        predefined_goal = None,
    )
    return env_kwargs

def get_default_wrapper_kwargs(env):
    # real은 PID, scale factor등은 필요가없음(HW built in이니까)
    return dict(env=env,
                g_control_type='move_gripper_position',
                q_control_type='speedj',
                speedj_args = {'a': 5, 't': 2/env.rate._freq, 'wait': False},
                servoj_args = {'t': 2/env.rate._freq, 'wait': False},
                g_control_args = {'wait' : False},
                )

def test_single_ur3_real():

    env_kwargs = get_default_env_kwargs()
    env = DSCHOSingleUR3PickAndPlaceRealEnv(**env_kwargs)
        
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
    
    for i in range(10): # 20Hz 기준 0.5초
        action = np.array([1.0, 0, 0, 1.0]) # moving x axis, closing gripper
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        print('obs : ', obs)
    time.sleep(3)

    for i in range(10):
        action = np.array([-1.0, 0, 0, -1.0]) # moving x axis, closing gripper
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        print('obs : ', obs)
    time.sleep(3)
    
    print('done')


def test_single_ur3_real_pickandplace():

    env_kwargs = get_default_env_kwargs()
    env = DSCHOSingleUR3PickAndPlaceRealEnv(**env_kwargs)
        
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
    
    print('NOTE : You\'d better use BaselineHERGoalEnvWrapper for conveniency in GCRL')
    
    pure_obs = obs['observation']
    achieved_goal = obs['achieved_goal']
    desired_goal = obs['desired_goal']
    ee_pos = pure_obs[:3]
    obj_pos = pure_obs[3:6]
    assert obj_pos==achieved_goal
    use_sleep = False
    duration = 1.0  #1초씩 움직임
    dt = env.dt
    for i in range(5):
        for t in range(int(duration/dt)):
            if i ==0: 
                action_xyz = np.tanh(obj_pos+np.array([0,0,0.1])- ee_pos)
                action = np.concatenate([action_xyz, np.array([-1.0])], axis =-1)
            elif i ==1: 
                action_xyz = np.tanh(obj_pos - ee_pos)
                action = np.concatenate([action_xyz, np.array([-1.0])], axis =-1)
            elif i ==2: 
                action_xyz = np.tanh(obj_pos - ee_pos)
                action = np.concatenate([action_xyz, np.array([1.0])], axis =-1)
            elif i ==3 or i ==4: 
                action_xyz = np.tanh(desired_goal - ee_pos)
                action = np.concatenate([action_xyz, np.array([1.0])], axis =-1)
            
            next_obs, reward, done, info = env.step(action)

            print('step : {} ee pos ; {} obj pos : {} rew : {}'.format(t, ee_pos, obj_pos, reward))
        
        if use_sleep:
            time.sleep(1)
    

    time.sleep(1)
    
    print('done')


if __name__ == "__main__":
    test_single_ur3_real()
    # test_single_ur3_real_pickandplace()