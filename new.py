import argparse
import os
from datetime import datetime
from absl import app, flags

from rljax.algorithm import SLAC
# from rljax.env import make_continuous_env
from rljax.trainer import SLACTrainer
import dmc2gym
from rgb_stacking import environment

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def test(argv):
    env = dmc2gym.make(domain_name="rgb_stacking", task_name='rgb_test_triplet3')
    env_test = dmc2gym.make(domain_name="rgb_stacking", task_name='rgb_test_triplet3')

    algo = SLAC(
        num_agent_steps=10**6,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0,
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join("logs", "rgb_stacking", f"{str(algo)}-seed{0}-{time}")

    trainer = SLACTrainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        save_params=True
    )
    trainer.train()
    env.close()
    env_test.close()


if __name__ == "__main__":
    app.run(test)
