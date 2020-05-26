import tensorflow as tf

# tfa
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.policies import greedy_policy

from tf_agents.agents.sac import sac_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils.common import Checkpointer
from tf_agents.trajectories import time_step as ts

from bark_ml.library_wrappers.lib_tf_agents.agents.tfa_agent import BehaviorTFAAgent
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML


class BehaviorGAILAgent(BehaviorTFAAgent, BehaviorContinuousML):
  """GAIL-Agent
     This agent is based on the tf-agents library.
  """
  def __init__(self,
               environment=None,
               params=None):
    BehaviorTFAAgent.__init__(self,
                              environment=environment,
                              params=params)
    BehaviorContinuousML.__init__(self, params)
    self._replay_buffer = self.GetReplayBuffer()
    self._expert_trajectories = self.GetExpertTrajectories()
    self._collect_policy = self.GetCollectionPolicy()
    self._eval_policy = self.GetEvalPolicy()

  # TODO Implement GetAgent() function
  def GetAgent(self, env, params):
    # generator network
    generator_net = None

    # discriminator network
    critic_net = None
    
    # agent
    tf_agent = None
    
    tf_agent.initialize()
    return tf_agent

  def GetReplayBuffer(self):
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=self._agent.collect_data_spec,
      batch_size=self._wrapped_env.batch_size,
      max_length=self._params["ML"]["BehaviorGAILAgent"]["ReplayBufferCapacity", "", 10000])

  # TODO Implement GetExpertTrajectories() function
  def GetExpertTrajectories(self):
    """
    This function returns the expert trajectories that are used for imitation learning.
    """
    return None

  def GetCollectionPolicy(self):
    return self._agent.collect_policy

  def GetEvalPolicy(self):
    return greedy_policy.GreedyPolicy(self._agent.policy)

  def Reset(self):
    pass

  @property
  def collect_policy(self):
    return self._collect_policy

  @property
  def eval_policy(self):
    return self._eval_policy

  def Act(self, state):
    action_step = self.eval_policy.action(
      ts.transition(state, reward=0.0, discount=1.0))
    return action_step.action.numpy()

  def Plan(self, observed_world, dt):
    if self._training == True:
      observed_state = self._environment._observer.Observe(observed_world)
      action = self.Act(observed_state)
      super().ActionToBehavior(action)
    return super().Plan(observed_world, dt)
