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

from train_online import MujocoSaver


def main(args):
    path = os.path.join("logs", "{}-{}-{}".format(
        "sac", args.env, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")))
    logger.configure(folder=path)

    env = make_mujoco_env(env_id=args.env, seed=args.seed)
    mujocosaver_dir = osp.join(logger.get_dir(), "mujoco_saver")
    os.makedirs(mujocosaver_dir, exist_ok=True)
    mujocosaver_path = osp.join(mujocosaver_dir, "mujoco_saver_0.pkl")
    env = MujocoSaver(env, savepath=mujocosaver_path)

    model = SAC(MlpPolicy, env, verbose=1, seed=args.seed, n_cpu_tf_sess=1)

    load_path = osp.join(args.load_dir, args.env, 'checkpoints', 'final.zip')
    logger.info("*********Trying load model params from: {}*********".format(load_path))
    model.load_parameters(load_path)

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
    parser.add_argument('--load_dir', type=str, default='dataset')
    parser.add_argument('--total_timesteps', type=float, default=1e3)
    parser.add_argument('--save_interval', type=float, default=5e5)
    main(parser.parse_args())
