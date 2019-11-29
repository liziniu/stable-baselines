import numpy as np
import argparse
import datetime
import os
import os.path as osp
from scipy import stats
import pickle

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC
from stable_baselines import logger
from stable_baselines.common.cmd_util import make_mujoco_env

from train_online import MujocoSaver


class NoiseModel(object):
    def __init__(self, ac_space, model, noise_type, seed=2019):
        self.ac_space = ac_space
        self.ac_space.seed(seed)
        self.model = model
        self.noise_type = str(noise_type)

        self.multi_variate_norm_pd = stats.multivariate_normal

    def step(self, obs):
        if 'random' in self.noise_type:
            acs = self.ac_space.sample()
            logps = self.multi_variate_norm_pd.logpdf(acs, mean=np.zeros_like(acs))
        else:
            acs, logps = self.model.policy_tf.step(obs[None])
            acs, logps = acs.flatten(), logps.flatten()
            if 'eps' in self.noise_type:
                eps = float(self.noise_type.split('_')[-1])
                randomized = np.random.uniform() < eps
                if randomized:
                    acs = self.ac_space.sample()
                    logps = self.model.policy_tf.evaluate_logp(obs[None], acs[None]).flatten()
            elif 'normal' in self.noise_type:
                std = float(self.noise_type.split('_')[-1])
                normal_noise = np.random.normal(loc=0, scale=std, size=np.shape(acs))
                acs += normal_noise
                logps = self.multi_variate_norm_pd.logpdf(normal_noise, mean=np.zeros_like(normal_noise))
            elif self.noise_type == 'none':
                pass
            else:
                raise ValueError('{} is not recognized'.format(self.noise_type))
        assert acs.ndim == 1  and logps.ndim == 1
        return acs, logps


def collect_trajectories(env, model, num_transitions, savedir):
    nb_samples = 0

    obs = env.reset()
    trajs, rets, lens = [], [], []
    mb_obs, mb_acs, mb_rs, mb_dones, mb_mus = [], [], [], [], []
    while nb_samples < num_transitions:
        acs, logps = model.step(obs)
        mb_obs.append(obs)
        mb_acs.append(acs)
        mb_mus.append(logps)
        obs, rewards, dones, infos = env.step(acs)
        mb_rs.append(rewards)
        mb_dones.append(dones)
        if dones:
            traj = dict(obs=np.asarray(mb_obs, dtype=env.observation_space.dtype),
                        acs=np.asarray(mb_acs, dtype=env.action_space.dtype),
                        rs=np.asarray(mb_rs, dtype=np.float32),
                        dones=np.asarray(mb_dones, dtype=np.bool),
                        mus=np.exp(np.asarray(mb_mus, dtype=np.float32)))
            trajs.append(traj)
            rets.append(np.sum(mb_rs))
            lens.append(len(mb_obs))
            nb_samples += len(mb_obs)
            with open(osp.join(savedir, 'samples_0.pkl'), 'ab+') as f:
                pickle.dump(traj, f)
            mb_obs, mb_acs, mb_rs, mb_dones, mb_mus = [], [], [], [], []
            obs = env.reset()
    logger.info("*******************************")
    logger.info("Total Trajectory: {}".format(len(trajs)))
    logger.info("Average Return: {}".format(np.mean(rets)))
    logger.info("Average Length: {}".format(np.mean(lens)))
    logger.info("*******************************")
    return trajs


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

    model = NoiseModel(ac_space=env.action_space, model=model,
                       noise_type=args.noise_type, seed=args.seed)

    savedir = osp.join(logger.get_dir(), "runner")
    os.makedirs(savedir)
    collect_trajectories(env, model, num_transitions=args.total_timesteps, savedir=savedir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v2')
    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--load_dir', type=str, default='dataset')
    parser.add_argument('--total_timesteps', type=float, default=1e4)
    parser.add_argument('--noise_type', type=str, default='none')
    main(parser.parse_args())
