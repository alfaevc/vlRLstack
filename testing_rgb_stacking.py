
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

ptu.set_gpu_mode(True)

variant = dict(
    algorithm="SAC",
    version="normal",
    layer_size=256,
    replay_buffer_size=int(2e4),
    algorithm_kwargs=dict(
        num_epochs=666,
        num_eval_steps_per_epoch=500,
        num_trains_per_train_loop=100,
        num_expl_steps_per_train_loop=100,
        min_num_steps_before_training=100,
        max_path_length=1000,
        batch_size=16,
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

expl_env = NormalizedBoxEnv(CarRacing())
eval_env = NormalizedBoxEnv(CarRacing())
obs_dim = expl_env.observation_space.low.size
action_dim = eval_env.action_space.low.size
M = variant["layer_size"]

# "size" will return the desired product of dimensions

(
    input_width,
    input_height,
    input_channels,
) = expl_env.observation_space.shape  # channel last!!!
qf1 = PretrainedCNN(
    input_width,
    input_height,
    input_channels,
    output_size=1,
    hidden_sizes=[128, 64],  # this is the hidden sizes of FC layers after the CNN
    added_fc_input_size=action_dim,  # layer used to merge image output and action input
    batch_norm_fc=False,
    init_w=1e-4,
    # hidden_init=nn.init.xavier_uniform_,
    # hidden_activation=nn.ReLU(),
    # output_activation=identity,
    output_conv_channels=False,
    model_architecture=models.efficientnet_b0,
    model_pretrained=True,
    model_freeze=False,
)
qf2 = PretrainedCNN(
    input_width,
    input_height,
    input_channels,
    output_size=1,
    hidden_sizes=[128, 64],  # this is the hidden sizes of FC layers after the CNN
    added_fc_input_size=action_dim,  # layer used to merge image output and action input
    batch_norm_fc=False,
    init_w=1e-4,
    # hidden_init=nn.init.xavier_uniform_,
    # hidden_activation=nn.ReLU(),
    # output_activation=identity,
    output_conv_channels=False,
    model_architecture=models.efficientnet_b0,
    model_pretrained=True,
    model_freeze=False,
)
target_qf1 = PretrainedCNN(
    input_width,
    input_height,
    input_channels,
    output_size=1,
    hidden_sizes=[128, 64],  # this is the hidden sizes of FC layers after the CNN
    added_fc_input_size=action_dim,  # layer used to merge image output and action input
    batch_norm_fc=False,
    init_w=1e-4,
    # hidden_init=nn.init.xavier_uniform_,
    # hidden_activation=nn.ReLU(),
    # output_activation=identity,
    output_conv_channels=False,
    model_architecture=models.efficientnet_b0,
    model_pretrained=True,
    model_freeze=False,
)
target_qf2 = PretrainedCNN(
    input_width,
    input_height,
    input_channels,
    output_size=1,
    hidden_sizes=[128, 64],  # this is the hidden sizes of FC layers after the CNN
    added_fc_input_size=action_dim,  # layer used to merge image output and action input
    batch_norm_fc=False,
    init_w=1e-4,
    # hidden_init=nn.init.xavier_uniform_,
    # hidden_activation=nn.ReLU(),
    # output_activation=identity,
    output_conv_channels=False,
    model_architecture=models.efficientnet_b0,
    model_pretrained=True,
    model_freeze=False,
)
policy = GaussianCNNPolicy(
    hidden_sizes=[128, 64],  # hidden size of FC after CNN; it uses "return_last_activations" to skip the last FC
    obs_dim=obs_dim,
    action_dim=action_dim,
    std=None,
    init_w=1e-3,
    min_log_std=-20,
    max_log_std=2,
    std_architecture="shared",
    **{
        "input_width": input_width,
        "input_height": input_height,
        "input_channels": input_channels,
        "kernel_sizes": [5, 5, 5],
        "n_channels": [32, 64, 128],
        "strides": [1] * 3,
        "paddings": ["same"] * 3,
    },
)

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
        **variant["trainer_kwargs"]
    )
algorithm = TorchBatchRLAlgorithm(
    trainer=trainer,
    exploration_env=expl_env,
    evaluation_env=eval_env,
    exploration_data_collector=expl_path_collector,
    evaluation_data_collector=eval_path_collector,
    replay_buffer=replay_buffer,
    **variant["algorithm_kwargs"]
)

setup_logger("carRace_testing", variant=variant)
algorithm.to(ptu.device)
algorithm.train()