import os
os.environ["LD_LIBRARY_PATH"] = ":/home/ztan/.mujoco/mujoco200/bin"
os.environ.get("LD_LIBRARY_PATH", "")

import argparse
from datetime import datetime

import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv

from slac_torch.slac.algo import SlacAlgorithm
from slac_torch.slac.env import make_dmc
from slac_torch.slac.trainer import Trainer

import torchvision.models as models

from absl import app, flags
from typing import Sequence
import sys
from dm_control import viewer
from dm_robotics.moma import action_spaces


import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import dmc2gym
from rgb_stacking import environment

from rnd import RND_CNN

ptu.set_gpu_mode(True)

def main(args):

    '''
    env = make_dmc(
        domain_name=args.domain_name,
        task_name=args.task_name,
        action_repeat=args.action_repeat,
        image_size=64,
    )
    env_test = make_dmc(
        domain_name=args.domain_name,
        task_name=args.task_name,
        action_repeat=args.action_repeat,
        image_size=64,
    )
    '''

    env = NormalizedBoxEnv(dmc2gym.make(domain_name="rgb_stacking", task_name='rgb_test_triplet1'))
    env_test = NormalizedBoxEnv(dmc2gym.make(domain_name="rgb_stacking", task_name='rgb_test_triplet1'))

    log_dir = os.path.join(
        "logs",
        f"{args.domain_name}-{args.task_name}",
        f'slac-seed{args.seed}-{datetime.now().strftime("%Y%m%d-%H%M")}',
    )
    input_width, input_height, input_channels = env.observation_space.shape
    action_dim, = env.action_space.shape

    print(env.observation_space.shape)
    print(env.action_space.shape)

    rnd = RND_CNN(input_width, input_height, input_channels, action_dim)

    algo = SlacAlgorithm(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        action_repeat=args.action_repeat,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        rnd_net=rnd,
    )
    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        rnd_net=rnd,
        seed=args.seed,
    )
    trainer.train()

    env.close()
    env_test.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--domain_name", type=str, default="cheetah")
    parser.add_argument("--task_name", type=str, default="run")
    parser.add_argument("--action_repeat", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    main(args)
