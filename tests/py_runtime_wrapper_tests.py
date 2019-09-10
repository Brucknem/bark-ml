# Copyright (c) 2019 fortiss GmbH
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
from modules.runtime.scenario.scenario_generation.uniform_vehicle_distribution \
  import UniformVehicleDistribution
from bark.runtime import eval_runtime
from modules.runtime.commons.parameters import ParameterServer
from modules.runtime.viewer.matplotlib_viewer import MPViewer

from src.rl_runtime import RuntimeRL
from src.observers.nearest_state_observer import StateConcatenation
from src.wrappers.dynamic_model import DynamicModel
from src.wrappers.motion_primitives import MotionPrimitives
from src.evaluators.goal_reached import GoalReached
import numpy as np


class RuntimeRLWrapperTests(unittest.TestCase):
  @staticmethod
  def test_dynamic_behavior_model():
    params = ParameterServer(filename="data/highway_merging.json")
    scenario_generation = UniformVehicleDistribution(num_scenarios=3,
                                                     random_seed=0,
                                                     params=params)
    state_observer = StateConcatenation(params=params)
    action_wrapper = DynamicModel(params=params)
    evaluator = GoalReached(params=params)
    viewer = MPViewer(params=params,
                      x_range=[-30,30],
                      y_range=[-20,40],
                      follow_agent_id=True) #use_world_bounds=True) #

    runtimerl = RuntimeRL(action_wrapper=action_wrapper,
                          observer=state_observer,
                          evaluator=evaluator,
                          step_time=0.2,
                          viewer=viewer,
                          scenario_generator=scenario_generation)
    print(runtimerl._action_wrapper.action_space)

    # pass the runtime into cpp
    action = np.array([0,0])
    eval_runtime(runtimerl, action)

  @staticmethod
  def test_motion_primitives_concat_state():
    params = ParameterServer(filename="data/highway_merging.json")
    scenario_generation = UniformVehicleDistribution(num_scenarios=3,
                                                     random_seed=0,
                                                     params=params)
    state_observer = StateConcatenation(params=params)
    action_wrapper = MotionPrimitives(params=params)
    evaluator = GoalReached(params=params)
    viewer = MPViewer(params=params,
                      x_range=[-30,30],
                      y_range=[-20,40],
                      follow_agent_id=True)
    runtimerl = RuntimeRL(action_wrapper=action_wrapper,
                          observer=state_observer,
                          evaluator=evaluator,
                          step_time=0.05,
                          viewer=viewer,
                          scenario_generator=scenario_generation)

    eval_runtime(runtimerl, 0)
if __name__ == '__main__':
    unittest.main()