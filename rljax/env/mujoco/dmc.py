# NOTE: this code was mainly taken from:
# https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py
import gym
import numpy as np
from dm_control import suite
from dm_env import specs
from gym import core, spaces

from rljax.env.atari import FrameStack

from rgb_stacking import environment
import cv2

gym.logger.set_level(40)


def make_dmc_env(domain_name, task_name, action_repeat=1, n_frames=3, image_size=64):
    env = make(
        domain_name=domain_name,
        task_name=task_name,
        visualize_reward=False,
        from_pixels=True,
        height=image_size,
        width=image_size,
        frame_skip=action_repeat,
    )
    if n_frames != 1:
        env = FrameStack(env, n_frames=n_frames)
    if not hasattr(env, "_max_episode_steps"):
        setattr(env, "_max_episode_steps", env.env._max_episode_steps)
    return env


def _spec_to_box(spec):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0)
    high = np.concatenate(maxs, axis=0)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


def make(
    domain_name,
    task_name,
    seed=1,
    visualize_reward=True,
    from_pixels=False,
    height=84,
    width=84,
    camera_id=0,
    frame_skip=1,
    episode_length=1000,
    environment_kwargs=None,
    time_limit=None,
):
    env_id = "dmc_%s_%s_%s_%s_%s-v1" % (domain_name, task_name, seed, height, width)

    if from_pixels:
        assert not visualize_reward, "cannot use visualize reward when learning from pixels"

    # shorten episode length
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

    if env_id not in gym.envs.registry.env_specs:
        task_kwargs = {}
        if seed is not None:
            task_kwargs["random"] = seed
        if time_limit is not None:
            task_kwargs["time_limit"] = time_limit
        gym.envs.registration.register(
            id=env_id,
            entry_point=DMCWrapper,
            kwargs=dict(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                environment_kwargs=environment_kwargs,
                visualize_reward=visualize_reward,
                from_pixels=from_pixels,
                height=height,
                width=width,
                camera_id=camera_id,
                frame_skip=frame_skip,
            ),
            max_episode_steps=max_episode_steps,
        )
    return gym.make(env_id)

class DMCWrapper(core.Env):
    def __init__(
        self,
        domain_name="rgb_stacking",
        task_name="rgb_test_triplet1",
        task_kwargs=None,
        visualize_reward={},
        from_pixels=False,
        height=128,
        width=256,
        camera_id=0,
        frame_skip=1,
        environment_kwargs=None,
        channels_first=False,
    ):
        assert (
            "random" in task_kwargs
        ), "please specify a seed, for deterministic behaviour"
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._channels_first = channels_first
        self._using_rgb_stacking = False

        
        # create task
        if domain_name == "rgb_stacking":
            self._using_rgb_stacking = True
            print(f"self._using_rgb_stacking {self._using_rgb_stacking}")
            self._env = environment.rgb_stacking(
                observation_set=environment.ObservationSet.VISION_ONLY,
                object_triplet=task_name,
            )
        else:
            self._env = suite.load(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                visualize_reward=visualize_reward,
                environment_kwargs=environment_kwargs,
            )

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()])
        self._norm_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=self._true_action_space.shape, dtype=np.float32
        )

        # print("?????")
        # create observation space

        if self._using_rgb_stacking: # obs
            shape = [64, 64, 3] # this WILL be used for indexing c,w,h. Reduce image size pls.
            self._observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        elif from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self._observation_space = _spec_to_box(
                self._env.observation_spec().values()
            )

        if self._using_rgb_stacking:
            shape = [256, 128, 3]
            self._state_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            ) # this will not be used for indexing h,w,c
        else:
            self._state_space = _spec_to_box(self._env.observation_spec().values())

        self.current_state = None

        # set seed
        self.seed(seed=task_kwargs.get("random", 1))

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._using_rgb_stacking:
            # obs = time_step.observation['basket_front_left/pixels'],  time_step.observation['basket_front_right/pixels']
            # we hardcoded out the observations!
            obs = np.concatenate(list(time_step.observation.values()), axis=1)
            # print(obs.shape)
            obs = cv2.resize(obs, dsize=(64, 64)).reshape((3,64,64))
            # print(obs.shape)
            # obs = _flatten_obs(obs)
        elif self._from_pixels:
            obs = self.render(
                height=self._height, width=self._width, camera_id=self._camera_id
            )
            if self._channels_first:
                obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = _flatten_obs(time_step.observation)
        # obs = pca_reduce_state(obs)
        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    def seed(self, seed):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        extra = {"internal_state": self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra["discount"] = time_step.discount
        return obs, reward, done, extra
    
    def step_rnd(self, action, rnd_net):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        extra = {"internal_state": self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra["discount"] = time_step.discount
        return obs, reward+rnd_net.get_reward(obs), done, extra

    def reset(self):
        time_step = self._env.reset()
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        return obs

    def render(self, mode="rgb_array", height=None, width=None, camera_id=0):
        assert mode == "rgb_array", "only support rgb_array mode, given %s" % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)
