from absl import flags
import acme
from acme import specs
from acme.agents.jax import sac
from absl import app
import helpers
import jax
from dm_control import viewer
import copy
import pyvirtualdisplay
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
import imageio 
import base64
import IPython
import numpy as np

from dm_control import viewer
from dm_robotics.moma import action_spaces

import environment as env

import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_steps', 10000,
                     'Number of env steps to run training for.')
flags.DEFINE_integer('eval_every', 1000, 'How often to run evaluation')
#flags.DEFINE_string('env_name', 'dm-control',
                    #'What environment to run')
flags.DEFINE_string('env_name', 'MountainCarContinuous-v0',
                    'What environment to run')
flags.DEFINE_integer('num_sgd_steps_per_step', 5,
                     'Number of SGD steps per learner step().')
flags.DEFINE_integer('seed', 0, 'Random seed.')

def render(env):
    return env.physics.render(camera_id=0)

def display_video(frames, filename='temp.mp4'):
  """Save and display video."""

  # Write video
  with imageio.get_writer(filename, fps=60) as video:
    for frame in frames:
      video.append_data(frame)

  # Read video and display the video
  video = open(filename, 'rb').read()
  b64_video = base64.b64encode(video)
  video_tag = ('<video  width="320" height="240" controls alt="test" '
               'src="data:video/mp4;base64,{0}">').format(b64_video.decode())

  return IPython.display.HTML(video_tag)




def main(_):
  # Create an environment, grab the spec, and use it to create networks.
  environment = env.rgb_stacking(observation_set=env.ObservationSet.VISION_ONLY, object_triplet='rgb_test_triplet1')
  #environment = helpers.make_environment(task=FLAGS.env_name)
  
  #environment.close()
  environment_spec = specs.make_environment_spec(environment)
  agent_networks = sac.make_networks(environment_spec)
  display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
  # Construct the agent.
  config = sac.SACConfig(
      target_entropy=sac.target_entropy_from_env_spec(environment_spec,0.2),
      num_sgd_steps_per_step=FLAGS.num_sgd_steps_per_step)
  agent = sac.SAC(
      environment_spec, agent_networks, config=config, seed=FLAGS.seed)

  # Create the environment loop used for training.
  train_loop = acme.EnvironmentLoop(environment, agent, label='train_loop')
  # Create the evaluation actor and loop.
  eval_actor = agent.builder.make_actor(
      random_key=jax.random.PRNGKey(FLAGS.seed),
      policy_network=sac.apply_policy_and_sample(
          agent_networks, eval_mode=True),
      variable_source=agent)
  #eval_env = helpers.make_environment(task=FLAGS.env_name)
  eval_env = env.rgb_stacking(observation_set=env.ObservationSet.VISION_ONLY, object_triplet='rgb_test_triplet1')
  #eval_env.close()
  eval_loop = acme.EnvironmentLoop(eval_env, eval_actor, label='eval_loop')
  #frames = [render(environment)]
  
  assert FLAGS.num_steps % FLAGS.eval_every == 0
  for _ in range(FLAGS.num_steps // FLAGS.eval_every):
    eval_loop.run(num_episodes=1)
    #frames.append(render(environment))
    train_loop.run(num_steps=FLAGS.eval_every)
    environment.close()
    eval_env.close()
  
  #display_video(np.array(frames))
  eval_loop.run(num_episodes=1)
  

if __name__ == '__main__':
  app.run(main)

 


  
