genrule(
    name = "unzip_lfs",
    outs = ["bark-ml"],
    cmd = "bash initialize.sh",
)

filegroup(
    name = "com_github_gail_4_bark_large_data_store",
    srcs = glob(["**/**", ]),
    visibility = ["//visibility:public"],
)