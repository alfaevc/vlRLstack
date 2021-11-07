from typing import Sequence

from absl import app
from dm_control import viewer

from rgb_stacking import environment


def main(argv: Sequence[str]) -> None:

    del argv

    # Load the rgb stacking environment.
    env = environment.rgb_stacking(observation_set=environment.ObservationSet.VISION_ONLY, object_triplet='rgb_test_triplet1')
    state = env.reset()
    print(state)

    # Launch the viewer application.
    # viewer.launch(env)
    env.close()

if __name__ == '__main__':
    app.run(main)