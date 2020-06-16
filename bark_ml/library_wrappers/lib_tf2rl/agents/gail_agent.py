# import sys
# import logging
# import time
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

# tfa
# from tf_agents.networks import actor_distribution_network
# from tf_agents.networks import normal_projection_network
from tf_agents.policies import greedy_policy

# from tf_agents.replay_buffers import tf_uniform_replay_buffer
# from tf_agents.utils.common import Checkpointer
from tf_agents.trajectories import time_step as ts

# tf2rl
import tf2rl
from tf2rl.algos.ddpg import DDPG
from tf2rl.algos.gail import GAIL

from bark_ml.library_wrappers.lib_tf2rl.agents.tf2rl_agent import BehaviorTF2RLAgent
from bark_ml.library_wrappers.lib_tf_agents.agents.tfa_agent import BehaviorTFAAgent
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML

class BehaviorGAILAgent(BehaviorTF2RLAgent, BehaviorContinuousML):
  """GAIL agent based on the tf2rl library."""

  def __init__(self,
                environment=None,
                params=None):
      
    BehaviorTF2RLAgent.__init__(self,
                                    environment=environment,
                                    params=params)
    BehaviorContinuousML.__init__(self, params)
    # self._replay_buffer = self.GetReplayBuffer()
    # self._dataset = self.GetDataset()
    # self._collect_policy = self.GetCollectionPolicy()
    # self._eval_policy = self.GetEvalPolicy()

    self._generator = self._get_generator()
    self._discriminator = self._get_discriminator()


  def _get_generator(self):
    policy = DDPG(
      state_shape=self._environment.observation_space.shape,
      action_dim=self._environment.action_space.high.size,
      max_action=self._environment.action_space.high[0],
      gpu=self._params.["ML"]["Settings"]["GPUUse", "", 0],
      actor_units=self._params.["ML"]["BehaviorGAILAgent"]["Generator"]["ActorFcLayerParams", "", [400, 300]],
      critic_units=self._params.["ML"]["BehaviorGAILAgent"]["Generator"]["CriticJointFcLayerParams", "", [400, 300]],
      n_warmup=self._params.["ML"]["BehaviorGAILAgent"]["WarmUp", "", 10000],
      batch_size=self._params.["ML"]["BehaviorGAILAgent"]["Generator"]["BatchSize", "", 100])
    return policy

  
  def _get_discriminator(self):
    """Instantiate discriminator network here."""
    irl = GAIL(
      state_shape=self._environment.observation_space.shape,
      action_dim=self._environment.action_space.high.size, 
      units=self._params.["ML"]["BehaviorGAILAgent"]["Discriminator"]["FcLayerParams", "", [400, 300]],
      enable_sn=self._params.["ML"]["BehaviorGAILAgent"]["EnableSN", "", False],
      batch_size=self._params.["ML"]["BehaviorGAILAgent"]["Discriminator"]["BatchSize", "", 32],
      gpu=self._params.["ML"]["Settings"]["GPUUse", "", 0])
    return irl

  # def GetEvalPolicy(self):
  #   return greedy_policy.GreedyPolicy(self._agent.policy)

  def Reset(self):
    pass

  # @property
  # def eval_policy(self):
  #   return self._eval_policy

  def Act(self, state):
    return self._generator.get_action(state)


  def Plan(self, observed_world, dt):
    """In SAC never called and additionally itself calls
    unimplemented functions there so ommited
    """
    pass


