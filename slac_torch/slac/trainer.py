import os
from collections import deque
from datetime import timedelta
from time import sleep, time

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class SlacObservation:
    """
    Observation for SLAC.
    """

    def __init__(self, state_shape, action_shape, num_sequences):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.num_sequences = num_sequences

    def reset_episode(self, state):
        self._state = deque(maxlen=self.num_sequences)
        self._action = deque(maxlen=self.num_sequences - 1)
        for _ in range(self.num_sequences - 1):
            self._state.append(np.zeros(self.state_shape, dtype=np.uint8))
            self._action.append(np.zeros(self.action_shape, dtype=np.float32))
        self._state.append(state)

    def append(self, state, action):
        self._state.append(state)
        self._action.append(action)

    @property
    def state(self):
        return np.array(self._state)[None, ...]

    @property
    def action(self):
        return np.array(self._action).reshape(1, -1)


class Trainer:
    """
    Trainer for SLAC.
    """

    def __init__(
        self,
        env,
        env_test,
        algo,
        log_dir,
        # rnd_net,
        seed=0,
        num_steps=10 ** 7,
        initial_collection_steps=10 ** 4,
        initial_learning_steps=10 ** 4,
        num_sequences=8,
        eval_interval=10 ** 3,
        num_eval_episodes=5,
    ):
        # Env to collect samples.
        self.env = env
        self.env.seed(seed)

        # Env for evaluation.
        self.env_test = env_test
        self.env_test.seed(2 ** 31 - seed)

        # Observations for training and evaluation.
        self.ob = SlacObservation(env.observation_space.shape, env.action_space.shape, num_sequences)
        self.ob_test = SlacObservation(env.observation_space.shape, env.action_space.shape, num_sequences)

        # Algorithm to learn.
        self.algo = algo

        # Log setting.
        self.log = {"step": [], "return": []}
        self.csv_path = os.path.join(log_dir, "log.csv")
        self.log_dir = log_dir
        self.summary_dir = os.path.join(log_dir, "summary")
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, "model")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        # self.action_repeat = self.env.action_repeat
        self.num_steps = num_steps
        self.initial_collection_steps = initial_collection_steps
        self.initial_learning_steps = initial_learning_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

        # self.rnd_net = rnd_net

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        state = self.env.reset()
        # print(state.shape)
        self.ob.reset_episode(state)
        self.algo.buffer.reset_episode(state)

        # Collect trajectories using random policy.
        for step in range(1, self.initial_collection_steps + 1):
            t = self.algo.step(self.env, self.ob, t, step <= self.initial_collection_steps)
        
        # Update latent variable model first so that SLAC can learn well using (learned) latent dynamics.
        bar = tqdm(range(self.initial_learning_steps))

        print("random pretrain done!")

        for _ in bar:
            bar.set_description("Updating latent variable model.")
            self.algo.update_latent(self.writer)

        # Iterate collection, update and evaluation.
        # for step in range(self.initial_collection_steps + 1, self.num_steps // self.action_repeat + 1):
        iters = tqdm(range(self.initial_collection_steps + 1, self.num_steps + 1))

        for step in iters:
            iters.set_description("Training policy")
            t = self.algo.step(self.env, self.ob, t, False)

            # Update the algorithm.
            self.algo.update_latent(self.writer)
            self.algo.update_sac(self.writer)

            # Evaluate regularly.
            # step_env = step * self.action_repeat
            step_env = step
            if step_env % self.eval_interval == 0:
                self.evaluate(step_env)
            if step_env % (self.eval_interval*10) == 0:
                print("Save model at step {}".format(step_env))
                self.algo.save_model(os.path.join(self.model_dir, f"step{step_env}"))

        print("Logging done!")
        # Wait for logging to be finished.
        sleep(10)

    def evaluate(self, step_env):
        mean_return = 0.0
        max_step = 1000

        for i in range(self.num_eval_episodes):
            state = self.env_test.reset()
            self.ob_test.reset_episode(state)
            episode_return = 0.0
            done = False
            j = 0

            while not done and j < max_step:
                action = self.algo.exploit(self.ob_test)
                state, reward, done, _ = self.env_test.step(action)
                self.ob_test.append(state, action)
                episode_return += reward
                j += 1

            mean_return += episode_return / self.num_eval_episodes

        # Log to CSV.
        self.log["step"].append(step_env)
        self.log["return"].append(mean_return)
        pd.DataFrame(self.log).to_csv(self.csv_path, index=False)

        # Log to TensorBoard.
        self.writer.add_scalar("return/test", mean_return, step_env)
        print(f"Steps: {step_env:<6}   " f"Return: {mean_return:<5.1f}   " f"Time: {self.time}")

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
