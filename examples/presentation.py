from bark_ml.library_wrappers.lib_tf2rl.load_expert_trajectories import load_expert_trajectories


expert_trajectories, avg_trajectory_length, num_trajectories = load_expert_trajectories("../../com_github_gail_4_bark_large_data_store/bark-ml/examples/expert_trajectories/interaction_dataset/DR_DEU_Merging_MT_v01_shifted") 

print(avg_trajectory_length)
print(num_trajectories)
