import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector


class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
        self,
        trainer,
        exploration_env,
        evaluation_env,
        exploration_data_collector: PathCollector,
        evaluation_data_collector: PathCollector,
        replay_buffer: ReplayBuffer,
        batch_size,
        max_path_length,
        num_epochs,
        num_eval_steps_per_epoch,
        num_expl_steps_per_train_loop,
        num_trains_per_train_loop,
        num_train_loops_per_epoch=1,
        min_num_steps_before_training=0,
        start_epoch=0,  # negative epochs are offline, positive epochs are online
        min_random_exploration_ratio=0.01,
        max_random_exploration_ratio=0.99,
        random_exploration_steps=100000,
        test_per_epoch = 5
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self._start_epoch = start_epoch
        self.min_random_exploration_ratio = min_random_exploration_ratio
        self.max_random_exploration_ratio = max_random_exploration_ratio

        ## step as random exploration rate decrease
        self.random_exploration_steps = random_exploration_steps
        self.step_count = 0
        self.random_exploration_ratio = max_random_exploration_ratio


        ####
        self.test_per_epoch = test_per_epoch

    def train(self):
        """Negative epochs are offline, positive epochs are online"""
        for self.epoch in gt.timed_for(
            range(self._start_epoch, self.num_epochs), save_itrs=True,
        ):
            print(f"self.random_exploration_ratio: {self.random_exploration_ratio}")
            self.offline_rl = self.epoch < 0
            self._begin_epoch(self.epoch)
            self._train()
            self._end_epoch(self.epoch)

    def _train(self):
        if self.epoch == 0 and self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
                eval=False,
                random_exploration_ratio=self.random_exploration_ratio,
            )
            if not self.offline_rl:
                self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        # if self.epoch % self.test_per_epoch == 0:
        self.eval_data_collector.collect_new_paths(
            self.max_path_length,
            self.num_eval_steps_per_epoch,
            discard_incomplete_paths=True,
            eval=True,
            random_exploration_ratio=0,
        )
        gt.stamp("evaluation sampling")

        for _ in range(self.num_train_loops_per_epoch): # num_train_loops_per_epoch == 1; this is used for increase train / explore ratio
            # print(f"calling collect_new_paths from train with {self.random_exploration_ratio}")
            ### in these step we should add more uniform sampling of exploration.
            new_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_expl_steps_per_train_loop,
                discard_incomplete_paths=False,
                eval=False,
                random_exploration_ratio=self.random_exploration_ratio,
            )
            gt.stamp("exploration sampling", unique=False)

            if not self.offline_rl:
                self.replay_buffer.add_paths(new_expl_paths)
            gt.stamp("data storing", unique=False)

            self.training_mode(True)
            for _ in range(self.num_trains_per_train_loop):
                train_data = self.replay_buffer.random_batch(self.batch_size)
                self.trainer.train(train_data)

            self.step_count += self.num_trains_per_train_loop
            self.random_exploration_ratio = self.min_random_exploration_ratio + (
                self.max_random_exploration_ratio - self.min_random_exploration_ratio
            ) * max(
                0,
                (self.random_exploration_steps - self.step_count)
                / self.random_exploration_steps,
            )

        ### step the exploration scheduler
        gt.stamp("training", unique=False)
        self.training_mode(False)


