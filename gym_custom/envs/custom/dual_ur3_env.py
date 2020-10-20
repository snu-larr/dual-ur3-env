import copy
import numpy as np
import pickle
import os
import warnings

import gym_custom
from gym_custom import utils
from gym_custom.envs.mujoco import MujocoEnv

#dscho mod
from gym_custom.core import Serializable
no_object_xmls = ['dscho_dual_ur3.xml', 'dscho_dual_ur3_obstacle_v0.xml', 'dscho_dual_ur3_obstacle_v1.xml', 'dscho_dual_ur3_obstacle_v2.xml', 'dscho_dual_ur3_obstacle_v3.xml', 'dscho_dual_ur3_obstacle_v4.xml', 'dscho_dual_ur3_obstacle_v5.xml']
lift_xmls = ['dscho_dual_ur3_bar.xml', 'dscho_dual_ur3_bar_obstacle_v0.xml', 'dscho_dual_ur3_bar_obstacle_v1.xml', 'dscho_dual_ur3_bar_obstacle_v2.xml','dscho_dual_ur3_bar_obstacle_v3.xml','dscho_dual_ur3_bar_obstacle_v4.xml','dscho_dual_ur3_bar_obstacle_v5.xml',\
            'dscho_dual_ur3_cylinder.xml', 'dscho_dual_ur3_cylinder_obstacle_v0.xml', 'dscho_dual_ur3_cylinder_obstacle_v1.xml', 'dscho_dual_ur3_cylinder_obstacle_v2.xml', 'dscho_dual_ur3_cylinder_obstacle_v3.xml','dscho_dual_ur3_cylinder_obstacle_v4.xml','dscho_dual_ur3_cylinder_obstacle_v5.xml',\
            ]
pickandplace_xmls = ['dscho_dual_ur3_bar_pickandplace.xml', 'dscho_dual_ur3_bar_pickandplace_obstacle_v0.xml', 'dscho_dual_ur3_bar_pickandplace_obstacle_v1.xml','dscho_dual_ur3_bar_pickandplace_obstacle_v2.xml', 'dscho_dual_ur3_bar_pickandplace_obstacle_v3.xml','dscho_dual_ur3_bar_pickandplace_obstacle_v4.xml','dscho_dual_ur3_bar_pickandplace_obstacle_v5.xml',\
                    'dscho_dual_ur3_cylinder_pickandplace.xml', 'dscho_dual_ur3_cylinder_pickandplace_obstacle_v0.xml', 'dscho_dual_ur3_cylinder_pickandplace_obstacle_v1.xml', 'dscho_dual_ur3_cylinder_pickandplace_obstacle_v2.xml', 'dscho_dual_ur3_cylinder_pickandplace_obstacle_v3.xml','dscho_dual_ur3_cylinder_pickandplace_obstacle_v4.xml','dscho_dual_ur3_cylinder_pickandplace_obstacle_v5.xml',\
                    ]
stick_pull_xmls = ['dscho_dual_ur3_stick_pull.xml', 'dscho_dual_ur3_stick_pull_obstacle_v0.xml', 'dscho_dual_ur3_stick_pull_obstacle_v2.xml', 'dscho_dual_ur3_stick_pull_obstacle_v3.xml','dscho_dual_ur3_stick_pull_obstacle_v4.xml','dscho_dual_ur3_stick_pull_obstacle_v5.xml']

class DualUR3Env(MujocoEnv, Serializable): #, utils.EzPickle

    # class variables
    # mujoco_xml_full_path = os.path.join(os.path.dirname(__file__), 'assets/ur3/dual_ur3_base.xml')
    mujocoenv_frame_skip = 1
    ur3_nqpos, gripper_nqpos = 6, 10 # per ur3/gripper joint pos dim
    ur3_nqvel, gripper_nqvel = 6, 10 # per ur3/gripper joint vel dim
    ur3_nact, gripper_nact = 6, 2 # per ur3/gripper action dim
    objects_nqpos = [7, 7, 7, 7]
    objects_nqvel = [6, 6, 6, 6]
    
    
    def __init__(self,
                xml_filename = None, 
                initMode = None,
                automatically_set_spaces=True,
                ur3_random_init = False,
                ):
        #dscho mod
        self.save_init_params(locals())
        self.xml_filename = xml_filename
        self.initMode = initMode
        self.automatically_set_spaces = automatically_set_spaces
        self.ur3_random_init = ur3_random_init
        if xml_filename is None :
            self.mujoco_xml_full_path = os.path.join(os.path.dirname(__file__), 'assets/ur3/dual_ur3_base.xml')
            self.ur3_nqpos, self.gripper_nqpos, self.objects_nqpos = 6, 10, [7,7,7,7] 
            self.ur3_nqvel, self.gripper_nqvel, self.objects_nqvel = 6, 10, [6,6,6,6]
            self.num_objects = 4
        elif xml_filename in lift_xmls or xml_filename in pickandplace_xmls:
            self.mujoco_xml_full_path = os.path.join(os.path.dirname(__file__), 'assets/ur3/'+xml_filename)
            self.ur3_nqpos, self.gripper_nqpos, self.objects_nqpos = 6, 10, [7] # long box or cylinder
            self.ur3_nqvel, self.gripper_nqvel, self.objects_nqvel = 6, 10, [6]
            self.num_objects = 1
        elif xml_filename in no_object_xmls:
            self.mujoco_xml_full_path = os.path.join(os.path.dirname(__file__), 'assets/ur3/'+xml_filename)
            self.ur3_nqpos, self.gripper_nqpos, self.objects_nqpos = 6, 10, [0]
            self.ur3_nqvel, self.gripper_nqvel, self.objects_nqvel = 6, 10, [0]
            self.num_objects = 0
        elif xml_filename in stick_pull_xmls:
            self.mujoco_xml_full_path = os.path.join(os.path.dirname(__file__), 'assets/ur3/'+xml_filename)
            self.ur3_nqpos, self.gripper_nqpos, self.objects_nqpos = 6, 10, [7,2] # long box(stick), pull_object
            self.ur3_nqvel, self.gripper_nqvel, self.objects_nqvel = 6, 10, [6,2]
            self.num_objects = 2
        
        #self._ezpickle_init()
        self._mujocoenv_init()
        self._check_model_parameter_dimensions()
        self._define_class_variables()

    def save_init_params(self, locals):
        """
        Should call this FIRST THING in the __init__ method if you ever want
        to serialize or clone this network.

        Usage:
        ```
        def __init__(self, ...):
            self.init_serialization(locals())
            ...
        ```
        :param locals:
        :return:
        """
        Serializable.quick_init(self, locals)

    # def _ezpickle_init(self):
    #    '''overridable method'''
    #    utils.EzPickle.__init__(self)

    def _mujocoenv_init(self):
        '''overridable method'''
        #dscho mod
        MujocoEnv.__init__(self, self.mujoco_xml_full_path, self.mujocoenv_frame_skip, automatically_set_spaces = self.automatically_set_spaces)
        if not self.automatically_set_spaces:
            self._set_action_space()
            self.do_simulation(self.action_space.sample(), self.frame_skip)

    def _check_model_parameter_dimensions(self):
        '''overridable method'''
        assert 2*self.ur3_nqpos + 2*self.gripper_nqpos + sum(self.objects_nqpos) == self.model.nq, 'Number of qpos elements mismatch'
        assert 2*self.ur3_nqvel + 2*self.gripper_nqvel + sum(self.objects_nqvel) == self.model.nv, 'Number of qvel elements mismatch'
        assert 2*self.ur3_nact + 2*self.gripper_nact == self.model.nu, 'Number of action elements mismatch'

    def _define_class_variables(self):
        '''overridable method'''
        # Initial position for UR3
        # dscho mod
        if self.ur3_random_init:
            pass
        if self.initMode is None :
            # self.init_qpos[0:self.ur3_nqpos] = \
            #     np.array([-90.0, -90.0, -90.0, -90.0, -135.0, 90.0])*np.pi/180.0 # right arm
            # self.init_qpos[self.ur3_nqpos+self.gripper_nqpos:2*self.ur3_nqpos+self.gripper_nqpos] = \
            #     np.array([90.0, -90.0, 90.0, -90.0, 135.0, -90.0])*np.pi/180.0 # left arm
            self.init_qpos[0:self.ur3_nqpos] = \
                np.array([-90.0, -90.0, -90.0, -90.0, -135.0, 180.0])*np.pi/180.0 # right arm
            self.init_qpos[self.ur3_nqpos+self.gripper_nqpos:2*self.ur3_nqpos+self.gripper_nqpos] = \
                np.array([90.0, -90.0, 90.0, -90.0, 135.0, -180.0])*np.pi/180.0 # left arm
        elif self.initMode =='vertical':
            
            # vertical init(high)
            # self.init_qpos[0:self.ur3_nqpos] = \
            #     np.array([-1.54849013, -2.45489269, -2.41625398,  0.0827262,  -2.35112646,  3.07500069]) # right arm
            # self.init_qpos[self.ur3_nqpos+self.gripper_nqpos:2*self.ur3_nqpos+self.gripper_nqpos] = \
            #     np.array([ 1.55129035, -0.68582331,  2.40896034, -3.22916226,  2.3560944,  -3.06996474])# left arm
            # vertical init(mid, wide)
            # self.init_qpos[0:self.ur3_nqpos] = \
            #     np.array([-1.27263236, -2.4677664,  -0.88920031, -1.6874239, -2.29199926, 1.57222008]) # right arm
            # self.init_qpos[self.ur3_nqpos+self.gripper_nqpos:2*self.ur3_nqpos+self.gripper_nqpos] = \
            #     np.array([1.27265907, -0.67387192, 0.88853694, -1.45401963, 2.29304517, -1.57573228])# left arm
            # vertical init(obstacle)
            self.init_qpos[0:self.ur3_nqpos] = \
                np.array([-0.32213427, -1.81002217, -1.87559869, -1.72603011, -1.79932887,  1.82011286]) # right arm
            self.init_qpos[self.ur3_nqpos+self.gripper_nqpos:2*self.ur3_nqpos+self.gripper_nqpos] = \
                np.array([ 0.3209594,  -1.33282653,  1.87653391, -1.41410399, 1.79674747, -1.81847637])# left arm
            
            

        elif self.initMode =='horizontal':
            # horizontal init
            self.init_qpos[0:self.ur3_nqpos] = \
                np.array([ 1.82496873, -1.78037016,  1.86075417,  4.40278818,  5.47660708, -2.8826006]) # right arm
            self.init_qpos[self.ur3_nqpos+self.gripper_nqpos:2*self.ur3_nqpos+self.gripper_nqpos] = \
                np.array([-1.85786483, -1.3540493,  -1.89351501, -1.18579177,  0.82976128, -0.50789828])# left arm
        else :
            raise NotImplementedError
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
        
        self.kinematics_params['T_wb_right'] = np.eye(4)
        self.kinematics_params['T_wb_right'][0:3,0:3] = self.sim.data.get_body_xmat('right_arm_rotz').reshape([3,3]).copy()
        self.kinematics_params['T_wb_right'][0:3,3] = self.sim.data.get_body_xpos('right_arm_rotz').copy()
        
        self.kinematics_params['T_wb_left'] = np.eye(4)
        self.kinematics_params['T_wb_left'][0:3,0:3] = self.sim.data.get_body_xmat('left_arm_rotz').reshape([3,3]).copy()
        self.kinematics_params['T_wb_left'][0:3,3] = self.sim.data.get_body_xpos('left_arm_rotz').copy()

        path_to_pkl = os.path.join(os.path.dirname(__file__), '../real/ur/dual_ur3_kinematics_params.pkl')
        if not os.path.isfile(path_to_pkl):
            pickle.dump(self.kinematics_params, open(path_to_pkl, 'wb'))

    #
    # Utilities (general)

    def forward_kinematics_DH(self, q, arm):
        assert len(q) == self.ur3_nqpos

        if arm == 'right':
            T_0_i = self.kinematics_params['T_wb_right']
        elif arm == 'left':
            T_0_i = self.kinematics_params['T_wb_left']
        else:
            raise ValueError('Invalid arm type!')
        T = np.zeros([self.ur3_nqpos+1, 4, 4])
        R = np.zeros([self.ur3_nqpos+1, 3, 3])
        p = np.zeros([self.ur3_nqpos+1, 3])
        # Base frame
        T[0,:,:] = T_0_i
        R[0,:,:] = T_0_i[0:3,0:3]
        p[0,:] = T_0_i[0:3,3]

        for i in range(self.ur3_nqpos):
            ct = np.cos(q[i] + self.kinematics_params['offset'][i])
            st = np.sin(q[i] + self.kinematics_params['offset'][i])
            ca = np.cos(self.kinematics_params['alpha'][i])
            sa = np.sin(self.kinematics_params['alpha'][i])

            T_i_iplus1 = np.array([[ct, -st*ca, st*sa, self.kinematics_params['a'][i]*ct],
                                   [st, ct*ca, -ct*sa, self.kinematics_params['a'][i]*st],
                                   [0, sa, ca, self.kinematics_params['d'][i]],
                                   [0, 0, 0, 1]])
            T_0_i = np.matmul(T_0_i, T_i_iplus1)
            # cf. base frame at i=0
            T[i+1, :, :] = T_0_i
            R[i+1, :, :] = T_0_i[0:3,0:3]
            p[i+1, :] = T_0_i[0:3,3]

        return R, p, T

    #dscho mod
    def forward_kinematics_DH_parallel(self, q, arm):
        # assert len(q) == self.ur3_nqpos
        assert q.ndim ==2
        assert q.shape[-1] == self.ur3_nqpos
        num_parallel = q.shape[0]

        if arm == 'right':
            T_0_i = np.tile(self.kinematics_params['T_wb_right'], (num_parallel,1,1)) #[parallel, 4,4]
        elif arm == 'left':
            T_0_i = np.tile(self.kinematics_params['T_wb_left'], (num_parallel,1,1))
        else:
            raise ValueError('Invalid arm type!')
        
        T = np.zeros([num_parallel, self.ur3_nqpos+1, 4, 4])
        R = np.zeros([num_parallel, self.ur3_nqpos+1, 3, 3])
        p = np.zeros([num_parallel, self.ur3_nqpos+1, 3])
        # Base frame
        T[:, 0,:,:] = T_0_i
        R[:, 0,:,:] = T_0_i[:, 0:3,0:3]
        p[:, 0,:] = T_0_i[:, 0:3,3]

        for i in range(self.ur3_nqpos):
            ct = np.cos(q[:, i] + np.tile(self.kinematics_params['offset'][i], num_parallel))
            st = np.sin(q[:, i] + np.tile(self.kinematics_params['offset'][i], num_parallel))
            ca = np.cos(np.tile(self.kinematics_params['alpha'][i], num_parallel))
            sa = np.sin(np.tile(self.kinematics_params['alpha'][i], num_parallel)) #[parallel]
            first_row = np.stack([ct, -st*ca, st*sa, np.tile(self.kinematics_params['a'][i], num_parallel)*ct], axis = -1) #[parallel, 4]
            second_row = np.stack([st, ct*ca, -ct*sa, np.tile(self.kinematics_params['a'][i], num_parallel)*st], axis = -1)
            third_row = np.stack([np.zeros_like(sa), sa, ca, np.tile(self.kinematics_params['d'][i], num_parallel)], axis = -1)
            fourth_row = np.stack([np.zeros_like(sa), np.zeros_like(sa), np.zeros_like(sa), np.ones_like(sa)], axis = -1)
            T_i_iplus1 = np.stack([first_row, second_row, third_row, fourth_row], axis = 1) #[parallel, 4, 4]

            T_0_i = np.matmul(T_0_i, T_i_iplus1) #[paralell, 4,4]
            # cf. base frame at i=0
            T[:, i+1, :, :] = T_0_i
            R[:, i+1, :, :] = T_0_i[:, 0:3,0:3]
            p[:, i+1, :] = T_0_i[:, 0:3,3]

        return R, p, T

    def forward_kinematics_ee(self, q, arm):
        R, p, T = self.forward_kinematics_DH(q, arm)
        return R[-1,:,:], p[-1,:], T[-1,:,:]

    def forward_kinematics_ee_parallel(self, q, arm):
        R, p, T = self.forward_kinematics_DH_parallel(q, arm)
        return R[:,-1,:,:], p[:,-1,:], T[:,-1,:,:]

    def _jacobian_DH(self, q, arm):
        assert len(q) == self.ur3_nqpos
        epsilon = 1e-6
        epsilon_inv = 1/epsilon
        _, ps, _ = self.forward_kinematics_DH(q, arm)
        p = ps[-1,:] # unperturbed position

        jac = np.zeros([3, self.ur3_nqpos])
        for i in range(self.ur3_nqpos):
            q_ = q.copy()
            q_[i] = q_[i] + epsilon
            _, ps_, _ = self.forward_kinematics_DH(q_, arm)
            p_ = ps_[-1,:] # perturbed position
            jac[:, i] = (p_ - p)*epsilon_inv

        return jac

    def inverse_kinematics_ee(self, ee_pos, null_obj_func, arm,
            q_init='current', threshold=0.01, threshold_null=0.001, max_iter=100, epsilon=1e-6
        ):
        '''
        inverse kinematics with forward_kinematics_DH() and _jacobian_DH()
        '''
        # Set initial guess
        if arm == 'right':
            if type(q_init).__name__ == 'ndarray': q = q_init.copy()
            elif q_init == 'current': q = self._get_ur3_qpos()[:self.ur3_nqpos]
            elif q_init == 'zero': q = np.zeros([self.ur3_nqpos])
            else: raise ValueError("q_init must be one of the following: ['current', 'zero', numpy.ndarray]")
        elif arm == 'left':
            if type(q_init).__name__ == 'ndarray': q = q_init.copy()
            elif q_init == 'current': q = self._get_ur3_qpos()[self.ur3_nqpos:]
            elif q_init == 'zero': q = np.zeros([self.ur3_nqpos])
            else: raise ValueError("q_init must be one of the following: ['current', 'zero', numpy.ndarray]")
        else:
            raise ValueError('Invalid arm type!')
        
        SO3, x, _ = self.forward_kinematics_ee(q, arm)
        jac = self._jacobian_DH(q, arm)
        delta_x = ee_pos - x
        err = np.linalg.norm(delta_x)
        null_obj_val = null_obj_func.evaluate(SO3)
        iter_taken = 0

        while True:
            if (err < threshold and null_obj_val < threshold_null) or iter_taken >= max_iter: break
            else: iter_taken += 1

            # pseudo-inverse + null-space approach
            jac_dagger = np.linalg.pinv(jac)
            jac_null = np.eye(self.ur3_nqpos) - np.matmul(jac_dagger, jac) # null space of Jacobian
            phi = np.zeros(self.ur3_nqpos) # find phi (null objective derivative)
            for i in range(self.ur3_nqpos):
                q_perturb = q.copy()
                q_perturb[i] += epsilon
                SO3_perturb, _, _ = self.forward_kinematics_ee(q_perturb, arm)
                null_obj_val_perturb = null_obj_func.evaluate(SO3_perturb)
                phi[i] = (null_obj_val_perturb - null_obj_val)/epsilon
            # update
            delta_x = ee_pos - x
            delta_q = np.matmul(jac_dagger, delta_x) - np.matmul(jac_null, phi)
            q += delta_q
            q = np.minimum(self.kinematics_params['ub'], np.maximum(q, self.kinematics_params['lb'])) # clip within theta bounds
            SO3, x, _ = self.forward_kinematics_ee(q, arm)
            jac = self._jacobian_DH(q, arm)
            null_obj_val = null_obj_func.evaluate(SO3)
            # evaluate
            err = np.linalg.norm(delta_x)
        
        if iter_taken == max_iter:
            pass
            # warnings.warn('Max iteration limit reached! err: %f (threshold: %f), null_obj_err: %f (threshold: %f)'%(err, threshold, null_obj_val, threshold_null),
            #     RuntimeWarning)
        
        return q, iter_taken, err, null_obj_val

    #
    # Utilities (MujocoEnv related)

    def get_body_se3(self, body_name):
        R = self.sim.data.get_body_xmat(body_name).reshape([3,3]).copy()
        p = self.sim.data.get_body_xpos(body_name).copy()
        T = np.eye(4)
        T[0:3,0:3] = R
        T[0:3,3] = p

        return R, p, T

    def _get_ur3_qpos(self):
        return np.concatenate([self.sim.data.qpos[0:self.ur3_nqpos], 
            self.sim.data.qpos[self.ur3_nqpos+self.gripper_nqpos:2*self.ur3_nqpos+self.gripper_nqpos]]).ravel()

    def _get_gripper_qpos(self):
        return np.concatenate([self.sim.data.qpos[self.ur3_nqpos:self.ur3_nqpos+self.gripper_nqpos], 
            self.sim.data.qpos[2*self.ur3_nqpos+self.gripper_nqpos:2*self.ur3_nqpos+2*self.gripper_nqpos]]).ravel()

    def _get_ur3_qvel(self):
        return np.concatenate([self.sim.data.qvel[0:self.ur3_nqvel], 
            self.sim.data.qvel[self.ur3_nqvel+self.gripper_nqvel:2*self.ur3_nqvel+self.gripper_nqvel]]).ravel()

    def _get_gripper_qvel(self):
        return np.concatenate([self.sim.data.qvel[self.ur3_nqvel:self.ur3_nqvel+self.gripper_nqvel], 
            self.sim.data.qvel[2*self.ur3_nqvel+self.gripper_nqvel:2*self.ur3_nqvel+2*self.gripper_nqvel]]).ravel()

    def _get_ur3_bias(self):
        return np.concatenate([self.sim.data.qfrc_bias[0:self.ur3_nqvel], 
            self.sim.data.qfrc_bias[self.ur3_nqvel+self.gripper_nqvel:2*self.ur3_nqvel+self.gripper_nqvel]]).ravel()

    def _get_gripper_bias(self):
        return np.concatenate([self.sim.data.qfrc_bias[self.ur3_nqvel:self.ur3_nqvel+self.gripper_nqvel], 
            self.sim.data.qfrc_bias[2*self.ur3_nqvel+self.gripper_nqvel:2*self.ur3_nqvel+2*self.gripper_nqvel]]).ravel()

    def _get_ur3_constraint(self):
        return np.concatenate([self.sim.data.qfrc_constraint[0:self.ur3_nqvel], 
            self.sim.data.qfrc_constraint[self.ur3_nqvel+self.gripper_nqvel:2*self.ur3_nqvel+self.gripper_nqvel]]).ravel()

    def _get_ur3_actuator(self):
        return np.concatenate([self.sim.data.qfrc_actuator[0:self.ur3_nqvel], 
            self.sim.data.qfrc_actuator[self.ur3_nqvel+self.gripper_nqvel:2*self.ur3_nqvel+self.gripper_nqvel]]).ravel()

    def _get_obs(self):
        '''overridable method'''
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def get_obs_dict(self):
        '''overridable method'''
        return {'right': {
                'qpos': self._get_ur3_qpos()[:self.ur3_nqpos],
                'qvel': self._get_ur3_qvel()[:self.ur3_nqvel],
                'gripperpos': self._get_gripper_qpos()[:self.gripper_nqpos],
                'grippervel': self._get_gripper_qvel()[:self.gripper_nqvel]
            },
            'left': {
                'qpos': self._get_ur3_qpos()[self.ur3_nqpos:],
                'qvel': self._get_ur3_qvel()[self.ur3_nqvel:],
                'gripperpos': self._get_gripper_qpos()[self.gripper_nqpos:],
                'grippervel':self._get_gripper_qvel()[self.gripper_nqvel:]
            }
        }

    #
    # Overrided MujocoEnv methods

    def step(self, a):
        '''overridable method'''
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, {}

    def reset_model(self):
        '''overridable method'''
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        '''overridable method'''
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent


def test_video_record(env):
    import time
    from gym_custom.wrappers.monitoring.video_recorder import VideoRecorder
    rec = VideoRecorder(env, enabled=True)
    stime = time.time()
    env.reset()
    rec.capture_frame()
    for i in range(int(2*rec.frames_per_sec)):
        action = env.action_space.sample()
        env.step(action)
        rec.capture_frame()
        print('step: %d'%(i))
    ftime = time.time()

    print('recording %f seconds of video took %f seconds'%(2, ftime-stime))
    rec.close()
    
    assert not rec.empty
    assert not rec.broken
    assert os.path.exists(rec.path)
    with open(rec.path) as f:
        print('path to file is %s'%(f.name))
        assert os.fstat(f.fileno()).st_size > 100


if __name__ == '__main__':
    env = gym_custom.make('dual-ur3-larr-v0')
    test_video_record(env)