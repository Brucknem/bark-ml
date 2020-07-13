workspace(name = "bark_ml")

load("//utils:dependencies.bzl", "bark_ml_dependencies")
bark_ml_dependencies()

load("@bark_project//tools:deps.bzl", "bark_dependencies")
bark_dependencies()

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()

# -------- Benchmark Database -----------------------
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository") 
git_repository(
  name = "benchmark_database",
  commit="78a248e3a1b272c4c8df0708306cfc05cbf8aab5",
  remote = "https://github.com/bark-simulator/benchmark-database"
)

load("@benchmark_database//util:deps.bzl", "benchmark_database_dependencies")
load("@benchmark_database//load:load.bzl", "benchmark_database_release")
benchmark_database_dependencies()
benchmark_database_release()
# --------------------------------------------------
