import sys
import logging
import time
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

# BARK imports
from bark.core.models.behavior import BehaviorModel

# tf2rl imports
import tf2rl

# BARK-ML imports


class BehaviorTF2RLAgent:
  """Base class for agents based on the tf2rl library."""

  def __init__(self,
                environment=None,
                params=None):
    """constructor

    Args:
        environment (Runtime, optional): A environment with a gym environment interface. Defaults to None.
        params (ParameterServer, optional): The parameter server holding the settings. Defaults to None.
    """
    self._params = params
    self._environment = environment
    self._training = False
    pass


  def Reset(self):
    """agent specific implemetation"""
    pass


  def Act(self, state):
    """agent specific implemetation"""
    pass


  def Plan(self, observed_world, dt):
    """agent specific implemetation"""
    pass


  def Save(self):
    """Placeholder
    """
    pass


  def Load(self):
    """Placeholder
    """
    pass




