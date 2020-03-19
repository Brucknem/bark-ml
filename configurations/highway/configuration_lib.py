from absl import app
from absl import flags
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.environments import parallel_py_environment

from modules.runtime.scenario.scenario_generation.uniform_vehicle_distribution \
  import UniformVehicleDistribution
from modules.runtime.scenario.scenario_generation.deterministic \
  import DeterministicScenarioGeneration
from modules.runtime.scenario.scenario_generation.configurable_scenario_generation import \
  ConfigurableScenarioGeneration
from modules.runtime.scenario.scenario_generation.config_with_ease import \
  LaneCorridorConfig, ConfigWithEase

from modules.runtime.commons.parameters import ParameterServer
from modules.runtime.viewer.matplotlib_viewer import MPViewer
from modules.runtime.viewer.video_renderer import VideoRenderer
from modules.runtime.viewer.pygame_viewer import PygameViewer


from src.rl_runtime import RuntimeRL
from src.observers.nearest_state_observer import ClosestAgentsObserver
from src.wrappers.dynamic_model import DynamicModel
from src.wrappers.tfa_wrapper import TFAWrapper
from src.evaluators.goal_reached import GoalReached
from src.agents.sac_agent import SACAgent
from src.agents.ppo_agent import PPOAgent
from src.runners.sac_runner import SACRunner
from src.runners.ppo_runner import PPORunner
from configurations.base_configuration import BaseConfiguration

# configuration specific evaluator
from configurations.highway.custom_evaluator import CustomEvaluator
from configurations.highway.scenario_conf import LeftLaneCorridorConfig, \
  RightLaneCorridorConfig


class HighwayConfiguration(BaseConfiguration):
  """Hermetic and reproducible configuration class
  """
  def __init__(self,
               params):
    BaseConfiguration.__init__(
      self,
      params)

  def _build_configuration(self):
    """Builds a configuration using an SAC agent
    """
    self._scenario_generator = \
      ConfigWithEase(num_scenarios=1,
                     map_file_name=self._params["BaseDir"] + "/tests/data/city_highway_straight.xodr",
                     random_seed=0,
                     params=self._params,
                     lane_corridor_configs=[LeftLaneCorridorConfig(params=self._params),
                                            RightLaneCorridorConfig(params=self._params)])

    self._observer = ClosestAgentsObserver(self._params)
    self._behavior_model = DynamicModel(params=self._params)
    self._evaluator = CustomEvaluator(params=self._params)
    viewer = MPViewer(params=self._params,
                      use_world_bounds=True)
                      # follow_agent_id=True)
    self._viewer = viewer
    self._runtime = RuntimeRL(action_wrapper=self._behavior_model,
                              observer=self._observer,
                              evaluator=self._evaluator,
                              step_time=0.2,
                              viewer=self._viewer,
                              scenario_generator=self._scenario_generator)
    tfa_env = tf_py_environment.TFPyEnvironment(TFAWrapper(self._runtime))
    self._agent = SACAgent(tfa_env, params=self._params)
    self._runner = SACRunner(tfa_env,
                             self._agent,
                             params=self._params,
                             unwrapped_runtime=self._runtime)

    # agent = SACAgent(tfa_env, params=self._params)
    # observer = ClosestAgentsObserver(self._params)
    # overwrite scenario generation
    self._scenario_generator = \
      ConfigWithEase(num_scenarios=100,
                     map_file_name=self._params["BaseDir"] + "/tests/data/city_highway_straight.xodr",
                     random_seed=0,
                     params=self._params,
                     lane_corridor_configs=[LeftLaneCorridorConfig(params=self._params),
                                            RightLaneCorridorConfig(
                                              ml_agent=self._agent,
                                              observer=self._observer,
                                              params=self._params)])
    self._runtime = RuntimeRL(action_wrapper=self._behavior_model,
                              observer=self._observer,
                              evaluator=self._evaluator,
                              step_time=0.2,
                              viewer=self._viewer,
                              scenario_generator=self._scenario_generator)
    tfa_env = tf_py_environment.TFPyEnvironment(TFAWrapper(self._runtime))
    self._runner = SACRunner(tfa_env,
                             self._agent,
                             params=self._params,
                             unwrapped_runtime=self._runtime)