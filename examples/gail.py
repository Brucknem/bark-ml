# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# TensorFlow Agents (https://github.com/tensorflow/agents) example
import os
from pathlib import Path
base_path = os.path.join(Path.home(), '.bark-ml', '.gail')
checkpoints_path = os.path.join(base_path, 'checkpoints')
summaries_path = os.path.join(base_path, 'summaries')
Path(checkpoints_path).mkdir(exist_ok=True, parents=True)
Path(summaries_path).mkdir(exist_ok=True, parents=True)

import gym
from absl import app
from absl import flags

# BARK imports
from bark_project.modules.runtime.commons.parameters import ParameterServer
from modules.runtime.viewer.matplotlib_viewer import MPViewer
from modules.runtime.viewer.video_renderer import VideoRenderer

# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, \
  ContinuousMergingBlueprint, ContinuousIntersectionBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorGAILAgent
from bark_ml.library_wrappers.lib_tf_agents.runners import GAILRunner


FLAGS = flags.FLAGS
flags.DEFINE_enum("mode",
                  "train",
                  # "visualize",
                  ["train", "visualize", "evaluate"],
                  "Mode the configuration should be executed in.")


def run_configuration(argv):
  params = ParameterServer(filename="examples/example_params/tfa_params.json")
  # params = ParameterServer()
  params["ML"]["BehaviorTFAAgents"]["CheckpointPath"] = checkpoints_path
  params["ML"]["TFARunner"]["SummaryPath"] = summaries_path
  params["World"]["remove_agents_out_of_map"] = True

  # create environment
  bp = ContinuousMergingBlueprint(params,
                                  number_of_senarios=500,
                                  random_seed=0)
  env = SingleAgentRuntime(blueprint=bp,
                           render=False)

  # GAIL-agent
  gail_agent = BehaviorGAILAgent(environment=env,
                               params=params)
  env.ml_behavior = gail_agent
  runner = GAILRunner(params=params,
                     environment=env,
                     agent=gail_agent)

  if FLAGS.mode == "train":
    runner.Train()
  elif FLAGS.mode == "visualize":
    runner.Visualize(5)
  
  # store all used params of the training
  # params.Save(os.path.join(Path.home(), "examples/example_params/tfa_params.json"))


if __name__ == '__main__':
  app.run(run_configuration)
  print('********************************************** Finished **********************************************')