# Description:
# Distributed runtime modules for machine learning, which allows coordination between multiple
# processes for distributed operations.

load(
    "@local_tsl//tsl/platform:rules_cc.bzl",
    "cc_library",
)
load("//xla/tsl:tsl.bzl", "internal_visibility")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//xla/tsl:internal",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "call_options",
    srcs = ["call_options.cc"],
    hdrs = ["call_options.h"],
    deps = [
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_tsl//tsl/platform:types",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "call_options.h",
    ],
)
