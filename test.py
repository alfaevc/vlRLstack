from typing import Sequence

from absl import app
from dm_control import viewer
from dm_robotics.moma import action_spaces

from rgb_stack.rgb_stacking import environment

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def something():
    expl_env = environment.rgb_stacking(
        observation_set=environment.ObservationSet.VISION_ONLY, object_triplet='rgb_test_triplet1')
    step_type, reward, discount, obs = expl_env.reset()
    expl_env.close()

def main(argv: Sequence[str]) -> None:

    # del argv

    # Load the rgb stacking environment.
    env = environment.rgb_stacking(
        observation_set=environment.ObservationSet.VISION_ONLY, object_triplet='rgb_test_triplet1')

    something()
    
    step_type, reward, discount, obs = env.reset()
    for i in obs:
        print("The dimension of {0} is {1}.".format(i, obs[i].shape))

    state = np.concatenate(
        (obs["basket_front_left/pixels"], obs["basket_front_right/pixels"]), axis=1)
    action = np.array([7.00e-02, 7.00e-02, 7.00e-02, 1.00e+00, 2.55e+02])/2


    print("shape of the state", state.shape)
    
    _, next_reward, _, next_obs = env.step(action)
    for i in next_obs:
        print("The dimension of {0} is {1}.".format(i, obs[i].shape))

    # Launch the viewer application.
    # viewer.launch(env)
    env.close()



if __name__ == '__main__':
    app.run(main)
