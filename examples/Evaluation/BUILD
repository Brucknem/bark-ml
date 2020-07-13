py_binary(
  name = "Benchmarking",
  srcs = ["Benchmarking.py"],
  data = ["@bark_project//bark/python_wrapper:core.so",
          "//bark_ml:generate_core","@benchmark_database//data/database1:database1",
          "//examples/example_params:example_params"],
  imports = ["../external/bark_project/bark/python_wrapper/"],
  deps = [
    "//bark_ml/environments:single_agent_runtime",
    "//bark_ml/environments:blueprints",
    "@bark_project//bark/runtime/commons:parameters",
    "@bark_project//bark/runtime/commons:xodr_parser",
    "@bark_project//bark/runtime/viewer:matplotlib_viewer",
    "@bark_project//bark/benchmark:benchmark_runner",
    "@bark_project//bark/benchmark:benchmark_analyzer",
    "@bark_project//bark/runtime/scenario/scenario_generation:scenario_generation",
    "@bark_project//bark/runtime:runtime",
    "@benchmark_database//load:benchmark_database",
    "@benchmark_database//serialization:database_serializer"
    ],
  visibility = ["//visibility:public"]
)
