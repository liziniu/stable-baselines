import gym
import numpy as np
import argparse
import datetime
import os
import os.path as osp
import pickle

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC
from stable_baselines import logger
from stable_baselines.common.cmd_util import make_mujoco_env


class MujocoSaver(gym.core.Wrapper):
    def __init__(self, env, savepath):
        self.savepath = savepath
        super().__init__(env)

        self.list_variables = []
        self.nb_traj = 0

    def _save_variables(self):
        variables = self.unwrapped.get_save_variables()
        self.list_variables.append(variables)

    def reset(self, **kwargs):
        self.list_variables = []
        obs = self.env.reset(**kwargs)
        self._save_variables()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._save_variables()
        if done:
            self._dump()
        return obs, reward, done, info

    def _dump(self):
        self.nb_traj += 1
        with open(self.savepath, "ab+") as f:
            pickle.dump(self.list_variables, f)
            if self.nb_traj % 100 == 0 or self.nb_traj == 1:
                print('MuJoCoSaver save {} trajectory into :{}'.format(self.nb_traj, self.savepath))


def main(args):
    path = os.path.join("logs", "{}-{}-{}".format(
        "sac", args.env, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")))
    logger.configure(folder=path)

    env = make_mujoco_env(env_id=args.env, seed=args.seed)
    mujocosaver_dir = osp.join(logger.get_dir(), "mujoco_saver")
    os.makedirs(mujocosaver_dir, exist_ok=True)
    mujocosaver_path = osp.join(mujocosaver_dir, "mujoco_saver_0.pkl")
    env = MujocoSaver(env, savepath=mujocosaver_path)

    model = SAC(MlpPolicy, env, verbose=1, seed=args.seed, n_cpu_tf_sess=1, learning_starts=0)

    model.learn(total_timesteps=int(args.total_timesteps), log_interval=10, save_samples=True,
                model_save_interval=args.save_interval)
    modelsave_dir = osp.join(logger.get_dir(), "checkpoints")
    os.makedirs(modelsave_dir, exist_ok=True)
    modelsave_path = osp.join(modelsave_dir, "final")
    model.save(modelsave_path)
    logger.info("save SAC model into: {}".format(modelsave_path))

    # del model  # remove to demonstrate saving and loading
    #
    # model = SAC.load("sac_pendulum")
    #
    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v2')
    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--total_timesteps', type=float, default=1e5)
    parser.add_argument('--save_interval', type=float, default=5e5)
    main(parser.parse_args())
