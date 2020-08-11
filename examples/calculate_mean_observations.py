import os
from pathlib import Path

from absl import app
from absl import flags

# BARK imports
from bark_project.bark.runtime.commons.parameters import ParameterServer
from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.viewer.video_renderer import VideoRenderer

# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, \
  ContinuousMergingBlueprint, ContinuousIntersectionBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf2rl.tf2rl_wrapper import TF2RLWrapper
from bark_ml.library_wrappers.lib_tf2rl.agents.gail_agent import BehaviorGAILAgent
from bark_ml.library_wrappers.lib_tf2rl.runners.gail_runner import GAILRunner
from bark_ml.library_wrappers.lib_tf2rl.load_expert_trajectories import load_expert_trajectories
import numpy as np

def calculate_mean(trajectories):
    """Calculates the mean of the observations and actions of the given trajectories.
    """
    observations = np.mean(trajectories['obses'], axis=0)
    actions = np.mean(trajectories['acts'], axis=0)
    return observations, actions

def print_observation(name, observation):
    """Prints the given observation
    """
    print(name)
    keys = ['X:', 'Y:', 'T:', 'V:']
    for i, key in enumerate(keys):
        values = [observation[j] for j in range(i, len(observation), 4)]
        print(key, values)
    print('*' * 80)

def print_action(name, action):
    """Prints the given action
    """
    print(name)
    print('Acc: ', action[0])
    print('Ang: ', action[1])
    print('*' * 80)

def compare_means(normalized, merging, interaction_dataset):
    merging_mean_observation, merging_mean_action = calculate_mean(merging)
    interaction_dataset_mean_observation, interaction_dataset_mean_action = calculate_mean(interaction_dataset)

    print('*' * 80)
    print('Normalized: ', normalized)

    print('*' * 80)
    print('Observations')
    print('*' * 80)
    print_observation('Merging', merging_mean_observation)
    print_observation('Interaction Dataset', interaction_dataset_mean_observation)
    print_observation('Difference', interaction_dataset_mean_observation - merging_mean_observation)

    print('*' * 80)
    print('Actions')
    print('*' * 80)
    print_action('Merging', merging_mean_action)
    print_action('Interaction Dataset', interaction_dataset_mean_action)
    print_action('Difference', interaction_dataset_mean_action - merging_mean_action)

def run_configuration(argv):
    """Loads the expert trajectories for the merging blueprint and the interaction dataset. Compares their mean actions and observations.

    Args:
        argv ([type]): [description]
    """
    merging, _, _ = load_expert_trajectories(
        "../com_github_gail_4_bark_large_data_store/expert_trajectories/sac/merging_not_normalized",
    ) 

    interaction_dataset, _, _ = load_expert_trajectories(
        "../com_github_gail_4_bark_large_data_store/expert_trajectories/interaction_dataset/DR_DEU_Merging_MT_v01_shifted/timestep_200_ms",
    ) 
    compare_means(False, merging, interaction_dataset)
    
    params = ParameterServer(filename="examples/example_params/gail_params.json")
    bp = ContinuousMergingBlueprint(params,
                                    number_of_senarios=2500,
                                    random_seed=0)
    env = SingleAgentRuntime(blueprint=bp,
                            render=False)
                            
    merging, _, _ = load_expert_trajectories(
        "../com_github_gail_4_bark_large_data_store/expert_trajectories/sac/merging_not_normalized",
        normalize_features=True, env=env) 
    interaction_dataset, _, _ = load_expert_trajectories(
        "../com_github_gail_4_bark_large_data_store/expert_trajectories/interaction_dataset/DR_DEU_Merging_MT_v01_shifted/timestep_200_ms",
        normalize_features=True, env=env) 
    compare_means(True, merging, interaction_dataset)

if __name__ == '__main__':
  app.run(run_configuration)