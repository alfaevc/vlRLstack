# vIRL Stack
Visual Reinforcement Learning using latent variable models trained for the task of stacking objects.

## Description 
This repo conains all the files used for the development of reinforcement learning algorithms trained on images for the task of stacking complex geometric object. Our scripts are modified versions of Deepmind's RGB Stacking environmnet, Deepmind's ACME framework, the RLKit Library (Pytorch Implementation) and RLJAX Library (JAX Implemenetation) algorithms.

## Motivation
This repo is the accumulated work done towards the COMS-6998 final project by Jobin, Alvin and Jerry. We show through our experiments that learning a latent variable model through representation learning assists in agents successfully learning long-horizon tasks such as stacking objects, using a robot arm, especially when training from images alone. This approach separates the representation learning task from the RL task allowing the agent to reach the optimal policy faster (i.e. higher sample efficiency) while also achieveing a better converged average reward when compared to learning directly from the observation space. We used an actor-critic method to further increase the sample efficiency as well as the robustness to hyperparameters as opposed to a policy-gradient method.
