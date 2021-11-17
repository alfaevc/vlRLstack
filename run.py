from absl import app, flags
from typing import Sequence
import sys
from absl import app
from dm_control import viewer
from dm_robotics.moma import action_spaces

from rgb_stack.rgb_stacking import environment

import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import GaussianCNNPolicy, TanhCNNGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks.pretrained_cnn import PretrainedCNN
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import itertools

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# flags.DEFINE_string("debug_specs", None, "NONE")
from torchinfo import summary

def experiment(variant):
    expl_env = environment.rgb_stacking(
        observation_set=environment.ObservationSet.VISION_ONLY, object_triplet='rgb_test_triplet1')
    eval_env = environment.rgb_stacking(
        observation_set=environment.ObservationSet.VISION_ONLY, object_triplet='rgb_test_triplet1')
    #obs_dim = expl_env.observation_space.low.size
    #action_dim = eval_env.action_space.low.size
    n_layers = 1
    kernel_size = list(itertools.repeat(3, n_layers))
    n_channels = list(itertools.repeat(5, n_layers))
    strides = list(itertools.repeat(1, n_layers))
    paddings = list(itertools.repeat("same", n_layers))

    step_type, reward, discount, obs = expl_env.reset() # throws an error..?????
    action_dim = 5
    # state = np.concatenate(
    #     (obs["basket_front_left/pixels"], obs["basket_front_right/pixels"]), axis=1)

    state_shape = (128, 256, 3)

    M = variant['layer_size']
    qf1 = PretrainedCNN(
        input_width=state_shape[0],
        input_height=state_shape[1],
        input_channels=state_shape[2],
        output_size=1,
        hidden_sizes=[M, M],
    )
    summary(qf1)
    qf2 = PretrainedCNN(
        input_width=state_shape[0],
        input_height=state_shape[1],
        input_channels=state_shape[2],
        output_size=1,
        hidden_sizes=[M, M],
    )
    summary(qf2)
    target_qf1 = PretrainedCNN(
        input_width=state_shape[0],
        input_height=state_shape[1],
        input_channels=state_shape[2],
        output_size=1,
        hidden_sizes=[M, M],
    )
    summary(target_qf1)
    target_qf2 = PretrainedCNN(
        input_width=state_shape[0],
        input_height=state_shape[1],
        input_channels=state_shape[2],
        output_size=1,
        hidden_sizes=[M, M],
    )
    summary(target_qf2)
    policy = GaussianCNNPolicy(
        obs_dim=1,
        action_dim=action_dim,
        hidden_sizes=[M, M],
        min_log_std=-20,
        max_log_std=2,
        **{'input_width': state_shape[0],
                'input_height': state_shape[1],
                'input_channels': state_shape[2],
                'output_size': action_dim,
                'kernel_sizes': kernel_size,
                'n_channels': n_channels,
                'strides': strides,
                'paddings': paddings},
    )
    summary(policy)

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

    expl_env.close()
    eval_env.close()


def main():
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=10000,
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=400,
            min_num_steps_before_training=400,
            max_path_length=400,
            batch_size=4,
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
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)

    # Launch the viewer application.
    # viewer.launch(env)

main()