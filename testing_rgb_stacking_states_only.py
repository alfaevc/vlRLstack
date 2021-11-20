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

variant = dict(
    algorithm="SAC",
    version="normal",
    layer_size=256,
    replay_buffer_size=int(50000),
    algorithm_kwargs=dict(
        num_epochs=3000,
        num_eval_steps_per_epoch=1200,
        num_trains_per_train_loop=2000,
        num_expl_steps_per_train_loop=2000,
        min_num_steps_before_training=2400,
        max_path_length=400,
        batch_size=4096,
        # num_eval_steps_per_epoch=1,
        # num_trains_per_train_loop=2000,
        # num_expl_steps_per_train_loop=1,
        # min_num_steps_before_training=1,
        # max_path_length=1,
        # batch_size=1,
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

expl_env = NormalizedBoxEnv(
    dmc2gym.make(domain_name="rgb_stacking_state_only", task_name="rgb_test_triplet5")
)
eval_env = NormalizedBoxEnv(
    dmc2gym.make(domain_name="rgb_stacking_state_only", task_name="rgb_test_triplet5")
)
obs_dim = expl_env.observation_space.low.size
action_dim = eval_env.action_space.low.size
pretrained_model = models.resnet18
input_width, input_height, input_channels = 128, 64, 3  # channel last!!!
q_additional_dim = action_dim + obs_dim - input_width * input_height * input_channels
M = variant["layer_size"]

# "size" will return the desired product of dimensions
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

# now: also use pretrainedCNN
# self.conv_output_flat_size: 1280 is the CNN output (effnet for example!)
eval_policy = MakeDeterministic(policy)
eval_path_collector = MdpPathCollector(eval_env, eval_policy,)
expl_path_collector = MdpPathCollector(expl_env, policy,)
replay_buffer = EnvReplayBuffer(variant["replay_buffer_size"], expl_env,)

trainer = SACTrainer(
    env=eval_env,
    policy=policy,
    qf1=qf1,
    qf2=qf2,
    target_qf1=target_qf1,
    target_qf2=target_qf2,
    **variant["trainer_kwargs"],
)
algorithm = TorchBatchRLAlgorithm(
    trainer=trainer,
    exploration_env=expl_env,
    evaluation_env=eval_env,
    exploration_data_collector=expl_path_collector,
    evaluation_data_collector=eval_path_collector,
    replay_buffer=replay_buffer,
    min_random_exploration_ratio=0.01,
    max_random_exploration_ratio=0.99,
    random_exploration_steps=100000,
    **variant["algorithm_kwargs"],
)

setup_logger("rgb_stacking_states_only", variant=variant)
algorithm.to(ptu.device)
algorithm.train()
