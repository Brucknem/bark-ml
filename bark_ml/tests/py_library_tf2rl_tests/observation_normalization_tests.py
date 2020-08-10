import unittest
import numpy as np
from gym.spaces.box import Box

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf2rl.normalization_utils import rescale, normalize
import os
from pathlib import Path

from absl import app
from absl import flags
import joblib

# BARK imports
from bark_project.bark.runtime.commons.parameters import ParameterServer
from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.viewer.video_renderer import VideoRenderer

# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, \
  ContinuousMergingBlueprint, ContinuousIntersectionBlueprint, GailMergingBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf2rl.tf2rl_wrapper import TF2RLWrapper
from bark_ml.library_wrappers.lib_tf2rl.agents.gail_agent import BehaviorGAILAgent
from bark_ml.library_wrappers.lib_tf2rl.runners.gail_runner import GAILRunner
from bark_ml.library_wrappers.lib_tf2rl.load_expert_trajectories import load_expert_trajectories
from bark_ml.library_wrappers.lib_tf2rl.load_save_utils import list_files_in_dir
import numpy as np

class ObservationNormalizationTests(unittest.TestCase):
    def test_normalization(self):
        params = ParameterServer(filename="examples/example_params/gail_params.json")

        bp = ContinuousMergingBlueprint(params, 
                                    number_of_senarios=100,
                                    random_seed=0)
        env = SingleAgentRuntime(blueprint=bp,
                            render=False)

        # wrapped environment for compatibility with tf2rl
        normalize_features = True
        wrapped_env = TF2RLWrapper(env, normalize_features=normalize_features)

        dirname = 'bark_ml/tests/py_library_tf2rl_tests/data/expert_trajectories/sac_both'

        joblib_files = list_files_in_dir(dirname, file_ending='.jblb')
        raw_trajectories = joblib.load(joblib_files[0])

        expert_trajectories, avg_trajectory_length, num_trajectories = load_expert_trajectories(
            dirname, 
            normalize_features=normalize_features, 
            env=env)

if __name__ == '__main__':
    unittest.main()
