import os
os.environ["LD_LIBRARY_PATH"] = ":/home/ztan/.mujoco/mujoco200/bin"
os.environ.get("LD_LIBRARY_PATH", "")


from gym.envs.mujoco import HalfCheetahEnv
from gym.envs.box2d import CarRacing

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import (
    TanhGaussianPolicy,
    MakeDeterministic,
    TanhCNNGaussianPolicy,
    GaussianCNNPolicy,
)
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import ConcatMlp, PretrainedCNN, CNN
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import torch
import torchvision.models as models

from absl import app, flags
from typing import Sequence
import sys
from absl import app
from dm_control import viewer
from dm_robotics.moma import action_spaces


import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import dmc2gym
from gym import core, spaces
from dm_control import suite
from dm_env import specs
import numpy as np
from rgb_stacking import environment

ptu.set_gpu_mode(True)

def experiment(variant):
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(2e4),
        algorithm_kwargs=dict(
            num_epochs=int(1e6),
            # num_eval_steps_per_epoch=2000,
            # num_trains_per_train_loop=2000,
            # num_expl_steps_per_train_loop=2000,
            # min_num_steps_before_training=2000,
            num_eval_steps_per_epoch=4,
            num_trains_per_train_loop=4,
            num_expl_steps_per_train_loop=4,
            min_num_steps_before_training=4,
            max_path_length=400,
            batch_size=12,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3e-4,
            qf_lr=3e-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )

    expl_env = NormalizedBoxEnv(dmc2gym.make(domain_name="rgb_stacking", task_name='rgb_test_triplet1'))
    eval_env = NormalizedBoxEnv(dmc2gym.make(domain_name="rgb_stacking", task_name='rgb_test_triplet1'))
    
    action_dim = eval_env.action_space.low.size
    M = variant["layer_size"]

    # "size" will return the desired product of dimensions

    #(input_width,input_height,input_channels) = expl_env.observation_space.shape
    n = 2

    #obs_dim = input_width
    #if input_width > input_height:
    #    obs_dim = input_height

    #obs_dim *= n * 3
    obs_dim = 64 * n * 3

    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

def main(_argv):
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    setup_logger('name-of-experiment', variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass 
