test_suite(
  name = "unit_tests",
  tests = [
    "//bark_ml/tests:py_environment_tests",
    "//bark_ml/tests:py_observer_tests",
    "//bark_ml/tests:py_evaluator_tests",
    "//bark_ml/tests:py_behavior_tests",
    "//bark_ml/tests:py_library_tfa_tests",
    "//utils/tests:test_generate_launch_configuration",
    "//bark_ml/tests/py_library_tf2rl_tests:load_expert_trajectories_tests",
    "//bark_ml/tests/py_library_tf2rl_tests:generate_expert_trajectories_tests",
    "//bark_ml/tests:py_gail_agent_tests",
    "//bark_ml/tests:py_gail_runner_tests",
    "//bark_ml/tests:py_gail_bark_training_tests"
  ]
)

test_suite(
  name = "examples_tests",
  tests = [
    "//examples:blueprint_config",
    "//examples:continuous_env",
    "//examples:discrete_env",
    "//examples:tfa"
  ]
)

test_suite(
  name = "diadem_tests",
  tests = [
    "//examples:diadem_dqn",
  ]
)