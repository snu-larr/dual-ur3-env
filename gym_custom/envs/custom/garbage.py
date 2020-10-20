
# should be inherited by child class
class DSCHODualUR3GoalEnv(DualUR3Env):

    def __init__(self,
                # random_init=False, #goal [0.25, -0.3, 0.8]
                # tasks = [{'goal': np.array([0.0, -0.4, 0.8]), 'goal_quat' : np.array([1., 0., 0., 0.]), 'obj_init_pos':np.array([0, -0.4, 0.65]), 'obj_init_angle': 0.0}], 
                sparse_reward = False,
                reduced_observation = False,
                trigonometry_observation = False, 
                random_init=False,
                full_state_goal = True,
                reward_by_ee = False, 
                automatically_set_spaces=False,
                reward_success_criterion='ee_pos',
                distance_threshold = 0.05,
                initMode='vertical',
                xml_filename = 'dscho_dual_ur3.xml',
                **kwargs
                ):
        self.full_state_goal = full_state_goal
        self.reduced_observation = reduced_observation
        self.trigonometry_observation = trigonometry_observation
        # for LEAP(S=G)
        assert (reduced_observation and not full_state_goal) or (not reduced_observation and full_state_goal)
        assert (reduced_observation and not trigonometry_observation) or (not reduced_observation and trigonometry_observation)
        self.reward_by_ee = reward_by_ee
        self.random_init = random_init
        self.curr_path_length = 0
        
        self.sparse_reward = sparse_reward
        self.distance_threshold = distance_threshold
        self.initMode = initMode
        self.reward_success_criterion = reward_success_criterion
        self.automatically_set_spaces = automatically_set_spaces
        
        super().__init__(xml_filename, initMode, automatically_set_spaces=automatically_set_spaces)
        
        if not automatically_set_spaces:
            self._set_action_space()

        self.obj_init_pos = self.get_obj_pos()
        self.obj_names = ['obj']
    
    def step(self, action):
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
        return self._state_goal.copy()
        
    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def set_goal(self, goal):
        self._state_goal = goal
        self._set_goal_marker(goal)

    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        # print('set_goal_marker : {}'.format(goal))
        self.data.site_xpos[self.model.site_name2id('goal')] = (
            goal[-3:]
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


class DSCHOSingleUR3GoalEnv(DSCHODualUR3GoalEnv):
    
    # Sholud be used with URScriptWrapper
    
    def __init__(self, which_hand='right', *args,**kwargs):
        self.which_hand = which_hand
        super().__init__(*args, **kwargs)
        self.left_get_away_qpos = np.concatenate([np.array([-90.0, -90.0, 90.0, -90.0, 135.0, 0.0])*np.pi/180.0, np.zeros(self.gripper_nqpos)]) # it was self.gripper_nqpos
        self.right_get_away_qpos = np.concatenate([np.array([-90.0, -90.0, 90.0, -90.0, 135.0, 0.0])*np.pi/180.0, np.zeros(self.gripper_nqpos)])
        
        if not self.reduced_observation:
            if self.trigonometry_observation:
                self.obs_nqpos = self.ur3_nqpos*2
            else :
                self.obs_nqpos = self.ur3_nqpos
        else :
            self.obs_nqpos = 3 # ee_pos

        if self.which_hand =='right':
            ee_low = np.array([-0.2, -0.7, 0.65])
            ee_high = np.array([0.7, -0.2, 0.95])
        
        elif self.which_hand =='left':
            ee_low = np.array([-0.7, -0.7, 0.65])
            ee_high = np.array([0.2, -0.2, 0.95])
        self.goal_ee_pos_space=Box(low = ee_low, high = ee_high, dtype=np.float32)
        
        if self.full_state_goal :
            qpos_low = -np.ones(int(self.ur3_nqpos))*2*np.pi
            qpos_high = np.ones(int(self.ur3_nqpos))*2*np.pi
            self.goal_qpos_space = Box(low=qpos_low, high=qpos_high, dtype=np.float32)
            

        self._state_goal = self.sample_goal(self.full_state_goal)

        observation, reward, done, _info = self.step(self.action_space.sample()) # goalenv는 return obs_dict
        assert not done
        
        self.observation_space = self._set_observation_space(observation) #[qpos, qvel, ee_pos]


    # goal space == state space
    # ee_pos 뽑고, 그에 따른 qpos 계산(IK) or qpos뽑고 그에따른 ee_pos 계산(FK)
    # 우선은 후자로 생각(어처피 학습할땐 여기저기 goal 다 뽑고, 쓸때는 제한해서 goal 샘플할꺼니까)
    
    def sample_goal(self, full_state_goal):
        if full_state_goal:
            t=0
            p = np.array([np.inf, np.inf, np.inf])
            while not (p <= self.goal_ee_pos_space.high).all() and (p >= self.goal_ee_pos_space.low).all():
                goal_qpos = np.random.uniform(
                    self.goal_qpos_space.low,
                    self.goal_qpos_space.high,
                    size=(self.goal_qpos_space.low.size),
                )
                # p = self.get_endeff_pos(arm=self.which_hand) 이걸로 구하면 샘플한 qpos가 아닌 현재 qpos기준 ee나오니까 안돼!
                R, p, _ = self.forward_kinematics_ee(goal_qpos, arm=self.which_hand)
                t+=1
            print('{} hand resample num : {}'.format(self.which_hand, t))
            goal_qvel = np.zeros(int(self.ur3_nqvel))
            if self.trigonometry_observation:
                goal_qpos = np.concatenate([np.cos(goal_qpos), np.sin(goal_qpos)], axis = -1)

            goal = np.concatenate([goal_qpos, goal_qvel, p], axis =-1) #[qpos, qvel, ee_pos]
        else :
            goal_ee_pos = np.random.uniform(
                self.goal_ee_pos_space.low,
                self.goal_ee_pos_space.high,
                size=(self.goal_ee_pos_space.low.size),
            )
            goal = goal_ee_pos
            
        return goal

    def reset_model(self):
        # original_ob = super().reset_model() # init qpos,qvel set_state and get_obs
        # assert not self.random_init, 'it is only for reaching env'
        # 이렇게 하면 obs에 샘플한 state_goal이 반영이 안됨. -> 밑에서 별도 설정!
        while True:
            self._state_goal = self.sample_goal(full_state_goal = self.full_state_goal)
            original_ob = super().reset_model() # init qpos,qvel set_state and get_obs
            current_ee_pos = self.get_endeff_pos(arm=self.which_hand)
            if np.linalg.norm(self._state_goal[-3:] - current_ee_pos) > 0.2:
                break
            
        self._set_goal_marker(self._state_goal)
        self.curr_path_length = 0
        self._set_obj_xyz(np.array([0.0, -0.8, 0.65]))
        # original_ob['desired_goal'] = self._state_goal
        return original_ob
        
    # Only ur3 qpos,vel(not include gripper), object pos(achieved_goal), desired_goal
    def _get_obs(self):
        
        if self.which_hand=='right':
            qpos = self._get_ur3_qpos()[:self.ur3_nqpos]
            qvel = self._get_ur3_qvel()[:self.ur3_nqvel]
            achieved_goal = ee_pos = self.get_endeff_pos(arm='right') # qpos idx찾아서 써야

        elif self.which_hand=='left':
            qpos = self._get_ur3_qpos()[self.ur3_nqpos:]
            qvel = self._get_ur3_qvel()[self.ur3_nqvel:]
            achieved_goal = ee_pos = self.get_endeff_pos(arm='left')
        
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
            'object_pos': self.get_obj_pos(name='obj'),
            'object_qpos': self.get_obj_qpos(name='obj'),
            'object_vel': self.get_obj_qvel(name='obj'),
            'object_quat': self.get_obj_quat(name='obj'),
            'l2_distance_to_goal' : np.linalg.norm(observation['desired_goal']-observation['achieved_goal'], ord=2, axis = -1), 
            'l1_distance_to_goal' : np.linalg.norm(observation['desired_goal']-observation['achieved_goal'], ord=1, axis = -1),
            'l2_distance_to_goal_of_interest' : np.linalg.norm(observation['desired_goal'][-3:]-observation['achieved_goal'][-3:], ord=2, axis = -1), # diffrent from reward_dim 
            'l1_distance_to_goal_of_interest' : np.linalg.norm(observation['desired_goal'][-3:]-observation['achieved_goal'][-3:], ord=1, axis = -1),
        }
        if self.full_state_goal:
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
        elif self.reward_by_ee:
            info.update({'l2_distance_to_goal_for_reward' : np.linalg.norm(observation['desired_goal'][-3:]-observation['achieved_goal'][-3:], ord=2, axis=-1),  
                         'l1_distance_to_goal_for_reward' : np.linalg.norm(observation['desired_goal'][-3:]-observation['achieved_goal'][-3:], ord=1, axis=-1),
                         })
        else :
            info.update({'l2_distance_to_goal_for_reward' : info['l2_distance_to_goal'],
                         'l1_distance_to_goal_for_reward' : info['l1_distance_to_goal']})

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
        # print('env state goal : {}'.format(self.env._state_goal))
        
        return observation, reward, done, info

    
    def convert_goal_for_reward(self, goal):
        if not self.full_state_goal:
            return goal
        elif self.reward_by_ee:
            return goal[-3:]
        else: #exclude qvel in reward computation in outer wrapper
            return np.concatenate([goal[:self.obs_nqpos], goal[-3:]], axis =-1)

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