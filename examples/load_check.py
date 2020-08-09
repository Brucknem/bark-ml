from bark_project.bark.runtime.commons.parameters import ParameterServer
from bark_ml.library_wrappers.lib_tf2rl.load_expert_trajectories import load_expert_trajectories
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, \
  ContinuousMergingBlueprint, ContinuousIntersectionBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime

params = ParameterServer(filename="examples/example_params/gail_params.json")

params["World"]["remove_agents_out_of_map"] = True

bp = ContinuousMergingBlueprint(params,
                                number_of_senarios=2500,
                                random_seed=0)
env = SingleAgentRuntime(blueprint=bp, render=False)

dirname="/Users/Marcel/Repositories/gail-4-bark/bark-ml/examples/expert_trajectories/sac_20000_observations"
expert_trajectories, avg_trajectory_length, num_trajectories = load_expert_trajectories(dirname,
      normalize_features=True,sac=False,
      env=env, # the unwrapped env has to be used, since that contains the unnormalized spaces.
      subset_size=-1
      )
print('Average trajectory length: {}'.format(avg_trajectory_length))
print('Number of trajectories: {}'.format(num_trajectories))