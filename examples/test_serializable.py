def serializable_env_test():
    
    import pickle
    import numpy as np
    import gym_custom
    env = gym_custom.make('dscho-single-ur3-reach-v0', fixed_goal_qvel = False) #default fixed_goal_qvel is True
    
    q_control_type = 'speedj'
    if q_control_type == 'servoj':
        PID_gains = {'servoj': {'P': 1.0, 'I': 0.5, 'D': 0.2}}
    elif q_control_type == 'speedj':
        PID_gains = {'speedj': {'P': 0.2, 'I': 10.0}} # was 0.2, 5.0
    ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
    gripper_scale_factor = np.array([1.0])
    g_control_type='move_gripper_force'
    #multi_step=5 # 1step : 0.01s -> 100Hz
    multi_step=20 # 1step : 0.04s -> 25Hz 
    gripper_action = False
    from gym_custom.envs.custom.dscho_dual_ur3_goal_env import SingleWrapper
    wrapped_env = SingleWrapper(env=env,
                                q_control_type=q_control_type, 
                                g_control_type=g_control_type, 
                                multi_step=multi_step, 
                                gripper_action=gripper_action,
                                PID_gains=PID_gains, 
                                ur3_scale_factor=ur3_scale_factor, 
                                gripper_scale_factor=gripper_scale_factor
                                )

    print(env._state_goal)
    print(env.fixed_goal_qvel)
    print(wrapped_env.fixed_goal_qvel)
    print(wrapped_env._state_goal)
    
    # for check getattr
    # from gym_custom.envs.custom.dscho_dual_ur3_goal_env import DummyWrapper
    # dummy_env = DummyWrapper(wrapped_env)
    

    pickle.dump(env, open('serializable_env.pkl', 'wb'))
    pickle.dump(wrapped_env, open('serializable_wrapped_env.pkl', 'wb'))
    import joblib
    loaded_env = joblib.load('serializable_env.pkl')
    loaded_wrapped_env = joblib.load('serializable_wrapped_env.pkl')
    print(loaded_env._state_goal)
    print(loaded_env.fixed_goal_qvel)
    print(env._state_goal==loaded_env._state_goal)
    print(loaded_wrapped_env.fixed_goal_qvel)
    print(loaded_wrapped_env._state_goal)




if __name__ == "__main__":
    serializable_env_test()