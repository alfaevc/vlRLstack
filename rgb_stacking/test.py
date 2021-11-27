from typing import Sequence

from absl import app
from dm_control import viewer
from dm_robotics.moma import action_spaces

from rgb_stacking import environment

import numpy as np

from sklearn.decomposition import PCA

from PIL import Image


def main(argv: Sequence[str]) -> None:

    del argv

    pca = PCA(n_components=2, svd_solver='arpack')

    # Load the rgb stacking environment.
    env = environment.rgb_stacking(observation_set=environment.ObservationSet.VISION_ONLY, object_triplet='rgb_test_triplet1')
    step_type, reward, discount, obs = env.reset()
    print(step_type)
    print(reward)
    print(discount)
    for i in obs:
        print("The dimension of {0} is {1}.".format(i, obs[i].shape))

    
    state = np.concatenate((obs["basket_front_left/pixels"], obs["basket_front_right/pixels"]), axis=1)

    pil_image = Image.fromarray(state)

    cols = pil_image.split()

    n = 2

    pca_list = []
    for col in cols:
        pca = PCA(n_components=n, svd_solver='arpack')
        col = np.array(col)
        pca.fit(col)
        pca_col = pca.transform(col)
        pca_list.append(pca_col)

    pca_state = np.concatenate(tuple(pca_list), axis=1)
    print(pca_state)
    print(pca_state.shape)

    action = np.array([7.00e-02,7.00e-02, 7.00e-02, 1.00e+00, 2.55e+02])/2
    step_type, next_reward, discount, next_obs = env.step(action)
    print(step_type)
    print(next_reward)
    print(discount)
    for i in next_obs:
        print("The dimension of {0} is {1}.".format(i, obs[i].shape))
    

    print(env.subtask._action_space._component_action_spaces)
    print(env.subtask._action_space._composite_action_spec)


    # Launch the viewer application.
    # viewer.launch(env)
    env.close()

if __name__ == '__main__':
    app.run(main)