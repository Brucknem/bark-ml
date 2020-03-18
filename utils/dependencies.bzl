load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

def _maybe(repo_rule, name, **kwargs):
    if name not in native.existing_rules():
        repo_rule(name = name, **kwargs)

def barkml_deps():
  _maybe(
    native.local_repository,
    name = "bark_project",
    path="/Users/hart/2019/bark",
  )
  
  # _maybe(
  #   git_repository,
  #   name = "bark_project",
  #   commit="e13042163625fdb7c5332da195f8d88d9bd70845",
  #   remote = "https://github.com/bark-simulator/bark",
  # )

  _maybe(
    git_repository,
    name = "com_github_gflags_gflags",
    branch = "master",
    remote = "https://github.com/gflags/gflags"
  )

  _maybe(
    git_repository,
    name = "com_github_google_glog",
    branch = "master",
    remote = "https://github.com/google/glog"
  )

  _maybe(
    git_repository,
    name = "gtest",
    branch = "master",
    remote = "https://github.com/google/googletest"
  )

  _maybe(
    http_archive,
    name = "pybind11",
    strip_prefix = "pybind11-2.3.0",
    urls = ["https://github.com/pybind/pybind11/archive/v2.3.0.zip"],
    build_file = "@bark_project//tools/pybind11:pybind.BUILD"
  )

  _maybe(
    git_repository,
    name = "com_github_nelhage_rules_boost",
    branch = "master",
    remote = "https://github.com/nelhage/rules_boost"
  )

  _maybe(
    http_archive,
    name = "com_github_eigen_eigen",
    build_file = "@bark_project//tools/eigen:eigen.BUILD",
    sha256 = "dd254beb0bafc695d0f62ae1a222ff85b52dbaa3a16f76e781dce22d0d20a4a6",
    strip_prefix = "eigen-eigen-5a0156e40feb",
    urls = [
        "http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2",
    ]
  )

  _maybe(
    native.new_local_repository,
    name = "python_linux",
    path = "./python/venv/",
    build_file_content = """
cc_library(
    name = "python-lib",
    srcs = glob(["lib/libpython3.*", "libs/python3.lib", "libs/python36.lib"]),
    hdrs = glob(["include/**/*.h", "include/*.h"]),
    includes = ["include/python3.6m", "include", "include/python3.7m", "include/python3.5m"], 
    visibility = ["//visibility:public"],
)
    """
  )

  _maybe(
    new_git_repository,
    name = "com_github_interaction_dataset_interaction_dataset",
    build_file = "@bark_project//tools/interaction-dataset:interaction-dataset.BUILD",
    commit = "8e53eecfa9cdcb2203517af2f8ed154ad40c2956",
    remote = "https://github.com/interaction-dataset/interaction-dataset.git",
    shallow_since = "1568028656 +0200",
  )