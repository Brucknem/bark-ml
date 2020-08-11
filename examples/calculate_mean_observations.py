import numpy as np
from absl import app
from bark_ml.library_wrappers.lib_tf2rl.load_expert_trajectories import load_expert_trajectories

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

def run_configuration(argv):
    """Loads the expert trajectories for the merging blueprint and the interaction dataset. Compares their mean actions and observations.

    Args:
        argv ([type]): [description]
    """
    merging, _, _ = load_expert_trajectories(
        "../com_github_gail_4_bark_large_data_store/expert_trajectories/sac/merging_not_normalized",
    ) 
    merging_mean_observation, merging_mean_action = calculate_mean(merging)

    interaction_dataset, _, _ = load_expert_trajectories(
        "../com_github_gail_4_bark_large_data_store/expert_trajectories/interaction_dataset/DR_DEU_Merging_MT_v01_shifted/timestep_200_ms",
    ) 
    interaction_dataset_mean_observation, interaction_dataset_mean_action = calculate_mean(interaction_dataset)

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

if __name__ == '__main__':
  app.run(run_configuration)