import os
from gym import utils
from gym.envs.robotics import fetch_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'reach.xml')


class FetchReachEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', obj_range=0.15, target_range=0.15):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=obj_range, target_range=target_range, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)


if __name__ == '__main__':
    import numpy as np
    env = FetchReachEnv()
    dict_obs = env.reset()
    ag = dict_obs['achieved_goal']
    list_ag_diff = []
    for i in range(50):
        action = env.action_space.sample()
        dict_obs_new, *_ = env.step(action)
        ag_diff = np.linalg.norm(np.array(dict_obs_new['achieved_goal']) - np.array(ag))
        ag = dict_obs_new['achieved_goal']
        list_ag_diff.append(ag_diff)
        print(ag_diff)
    print()
    print(np.mean(list_ag_diff))
