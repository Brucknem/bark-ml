import sys
import logging
import time
import argparse
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

# BARK imports
#from bark_project.modules.runtime.commons.parameters import ParameterServer

# tf2rl imports
import tf2rl
from tf2rl.experiments.irl_trainer import IRLTrainer
from tf2rl.experiments.utils import restore_latest_n_traj

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf2rl.runners.tf2rl_runner import TF2RLRunner

class GAILRunner(TF2RLRunner):
  """GAIL runner implementation based on tf2rl library."""

  def __init__(self,
              environment=None,
              agent=None,
              params=None):
      
    TF2RLRunner.__init__(self,
                    environment=environment,
                    agent=agent,
                    params=params)
    

  def _train(self):
    """Traines the GAIL agent with the IRLTrainer which was
    instantiated by self.GetTrainer.
    """
    self._trainer()

  
  def GetTrainer(self):
    """Creates an IRLtrainer instance."""
    policy = self._agent._generator   # the agent's generator network, so in our case the DDPG agent
    irl = self._agent._discriminator  # the agent's discriminator network so in our case the GAIL network

    # creating args from the ParameterServer which can be given to the IRLtrainer:
    args = self._get_args_from_params()

    # getting the expert trajectories from the .pkl file:
    expert_trajs = restore_latest_n_traj(dirname=args.expert_path_dir,
                                        # n_path=args['n_path'],
                                        max_steps=args.max_steps)
    
    trainer = IRLTrainer(policy=policy,
                         env=self._environment,
                         args=args,
                         irl=irl,
                         expert_obs=expert_trajs["obses"],
                         expert_next_obs=expert_trajs["next_obses"],
                         expert_act=expert_trajs["acts"])

    return trainer


  def _get_args_from_params(self):
    """creates an args object from the ParameterServer object, that
    can be given to the IRLtrainer.
    Args:
      EXPERIMENT SETTINGS:
        - max_steps:              int, Maximum number steps to interact with env.
        - episode_max_steps:      int, Maximum steps in an episode
        - n_experiments:          int, Number of experiments
        - show_progress:          bool, Call `render` in training process
        - save_model_interval:    int, Interval to save model
        - save_summary_interval:  int, Interval to save summary
        - dir_suffix:             str, Suffix for directory that contains results
        - normalize_obs:          bool, Normalize observation
        - logdir:                 str, Output directory
        - logging_level           str, logging level, choices=['DEBUG', 'INFO', 'WARNING']
        - model_dir:              str, Directory to restore model
      REPLAY BUFFER:
        - expert_path_dir:        str, Directory that contains expert trajectories
        - use_propritized_rb:     bool, Flag to use prioritized experience replay
        - use_nstep_rb:           bool, Flag to use nstep experience replay
        - n_step:                 int, Number of steps to look over
      TEST SETTINGS:
        - evaluaate:              bool, evaluate trained agent.
        - test_interval:          int, Interval to evaluate trained model
        - show_test_progress:     bool, Call `render` in evaluation process
        - test_episodes:          int, Number of episodes to evaluate at once
        - save_test_path:         str, Save trajectories of evaluation
        - save_test_movie:        bool, Save rendering results
        - show_test_images:       bool, Show input images to neural networks when an episode finishes
      OTHER:
        - gpu:                    int, name of gpu device

    """
    args = {}
    # experiment settings
    args['max_steps'] = self._params['ML']['GAILRunner']['tf2rl']['max_steps']
    args['episode_max_steps'] = self._params['ML']['GAILRunner']['tf2rl']['episode_max_steps']
    args['n_experiments'] = self._params['ML']['GAILRunner']['tf2rl']['n_experiments']
    args['show_progress'] = self._params['ML']['GAILRunner']['tf2rl']['show_progress']
    args['save_model_interval'] = self._params['ML']['GAILRunner']['tf2rl']['save_model_interval']
    args['save_summary_interval'] = self._params['ML']['GAILRunner']['tf2rl']['save_summary_interval']
    args['dir_suffix'] = self._params['ML']['GAILRunner']['tf2rl']['dir_suffix']
    args['normalize_obs'] = self._params['ML']['GAILRunner']['tf2rl']['normalize_obs']
    args['logdir'] = self._params['ML']['GAILRunner']['tf2rl']['logdir']
    args['logging_level'] = self._params['ML']['GAILRunner']['tf2rl']['logging_level']
    args['model_dir'] = self._params['ML']['GAILRunner']['tf2rl']['model_dir']

    # replay buffer
    args['expert_path_dir'] = self._params['ML']['GAILRunner']['tf2rl']['expert_path_dir']
    args['use_prioritized_rb'] = self._params['ML']['GAILRunner']['tf2rl']['use_prioritized_rb']
    args['use_nstep_rb'] = self._params['ML']['GAILRunner']['tf2rl']['use_nstep_rb']
    args['n_step'] = self._params['ML']['GAILRunner']['tf2rl']['n_step']

    # test settings
    args['evaluate'] = self._params['ML']['GAILRunner']['tf2rl']['evaluate']
    args['test_interval'] = self._params['ML']['GAILRunner']['tf2rl']['test_interval']
    args['show_test_progress'] = self._params['ML']['GAILRunner']['tf2rl']['show_test_progress']
    args['test_episodes'] = self._params['ML']['GAILRunner']['tf2rl']['test_episodes']
    args['save_test_path'] = self._params['ML']['GAILRunner']['tf2rl']['save_test_path']
    args['save_test_movie'] = self._params['ML']['GAILRunner']['tf2rl']['save_test_movie']
    args['show_test_images'] = self._params['ML']['GAILRunner']['tf2rl']['show_test_images']

    # other:
    args['gpu'] = self._params["ML"]["Settings"]["GPUUse", "", 0]

    return argparse.Namespace(**args)