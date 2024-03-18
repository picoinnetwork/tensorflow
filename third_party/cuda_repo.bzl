# Copyright 2024 The TensorFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for defining Google ML CUDA dependencies."""

load("@cuda_redist_json//:build_defs.bzl", "get_cuda_distributives", "get_cudnn_distributives")
load("//third_party:repo.bzl", "tf_mirror_urls")

_OS_ARCH_DICT = {
    "amd64": "x86_64-unknown-linux-gnu",
    "aarch64": "aarch64-unknown-linux-gnu",
}
_REDIST_ARCH_DICT = {
    "linux-x86_64": "x86_64-unknown-linux-gnu",
    "linux-sbsa": "aarch64-unknown-linux-gnu",
}

_CUDA_DIST_PATH_PREFIX = "https://developer.download.nvidia.com/compute/cuda/redist/"
_CUDNN_DIST_PATH_PREFIX = "https://developer.download.nvidia.com/compute/cudnn/redist/"

def _get_env_var(ctx, name):
    if name in ctx.os.environ:
        return ctx.os.environ[name]
    else:
        return None

def _get_archive_name(url, archive_suffix = ".tar.xz"):
    last_slash_index = url.rfind("/")
    return url[last_slash_index + 1:-len(archive_suffix)]

def _cuda_http_archive_impl(repository_ctx):
    cuda_or_cudnn_version = None
    dist_version = ""
    saved_version = ""
    cuda_version = _get_env_var(repository_ctx, "TF_CUDA_VERSION")
    cudnn_version = _get_env_var(repository_ctx, "TF_CUDNN_VERSION")
    if repository_ctx.attr.is_cudnn_dist:
        cuda_or_cudnn_version = cudnn_version
    else:
        cuda_or_cudnn_version = cuda_version
    if cuda_or_cudnn_version:
        # Download archive only when GPU config is used.
        dist_version = repository_ctx.attr.dist_version
        arch_key = _OS_ARCH_DICT[repository_ctx.os.arch]
        if arch_key not in repository_ctx.attr.relative_url_dict.keys():
            arch_key = "cuda{version}_{arch}".format(
                version = cuda_version.split(".")[0],
                arch = arch_key,
            )
        if arch_key in repository_ctx.attr.relative_url_dict.keys():
            (relative_url, sha256) = repository_ctx.attr.relative_url_dict[arch_key]
            url = (repository_ctx.attr.cudnn_dist_path_prefix if repository_ctx.attr.is_cudnn_dist else repository_ctx.attr.cuda_dist_path_prefix) + relative_url

            archive_name = _get_archive_name(url)
            file_name = archive_name + repository_ctx.attr.extension

            repository_ctx.download(
                url = tf_mirror_urls(url),
                output = file_name,
                sha256 = sha256,
            )
            repository_ctx.extract(
                archive = file_name,
                stripPrefix = repository_ctx.attr.override_strip_prefix if repository_ctx.attr.override_strip_prefix else archive_name,
            )
            repository_ctx.delete(file_name)

            if repository_ctx.attr.build_template:
                saved_version = dist_version.split(".")[0] if dist_version else ""
                build_template = repository_ctx.attr.build_template

                # Workaround for CUDA 11 distributive versions.
                if cuda_version and cuda_version.startswith("11"):
                    if saved_version == "11":
                        if repository_ctx.name == "cuda_cudart":
                            saved_version = "11.0"
                        if repository_ctx.name == "cuda_cupti":
                            saved_version = cuda_version

                repository_ctx.template(
                    "BUILD",
                    build_template,
                    {"%{version}": saved_version},
                )
            else:
                repository_ctx.file(
                    "BUILD",
                    repository_ctx.read(repository_ctx.attr.build_file),
                )
        else:
            # If no matching arch is found, use the dummy build file.
            repository_ctx.file(
                "BUILD",
                repository_ctx.read(repository_ctx.attr.dummy_build_file),
            )
    else:
        # If no CUDA or CUDNN version is found, use the dummy build file if present.
        repository_ctx.file(
            "BUILD",
            repository_ctx.read(repository_ctx.attr.dummy_build_file or repository_ctx.attr.build_file),
        )
    repository_ctx.file("version.txt", saved_version)

_cuda_http_archive = repository_rule(
    implementation = _cuda_http_archive_impl,
    attrs = {
        "dist_version": attr.string(mandatory = True),
        "relative_url_dict": attr.string_list_dict(mandatory = True),
        "build_template": attr.label(),
        "dummy_build_file": attr.label(),
        "build_file": attr.label(),
        "is_cudnn_dist": attr.bool(),
        "override_strip_prefix": attr.string(),
        "cudnn_dist_path_prefix": attr.string(default = _CUDNN_DIST_PATH_PREFIX),
        "cuda_dist_path_prefix": attr.string(default = _CUDA_DIST_PATH_PREFIX),
        "extension": attr.string(default = ".tar.xz"),
    },
    environ = ["TF_CUDA_VERSION", "TF_CUDNN_VERSION"],
)

def cuda_http_archive(name, dist_version, relative_url_dict, **kwargs):
    _cuda_http_archive(
        name = name,
        dist_version = dist_version,
        relative_url_dict = relative_url_dict,
        **kwargs
    )

def _cuda_wheel_impl(repository_ctx):
    cuda_version = _get_env_var(repository_ctx, "TF_CUDA_VERSION")
    if cuda_version:
        # Download archive only when GPU config is used.
        arch = _OS_ARCH_DICT[repository_ctx.os.arch]
        dict_key = "{cuda_version}-{arch}".format(
            cuda_version = cuda_version,
            arch = arch,
        )
        supported_versions = repository_ctx.attr.url_dict.keys()
        if dict_key not in supported_versions:
            fail(
                ("The supported NCCL versions are {supported_versions}." +
                 " Please provide a supported CUDA version in TF_CUDA_VERSION" +
                 " environment variable or add NCCL distributive for" +
                 " CUDA version={version}, OS={arch}.")
                    .format(
                    supported_versions = supported_versions,
                    version = cuda_version,
                    arch = arch,
                ),
            )
        sha256 = repository_ctx.attr.sha256_dict[dict_key]
        url = repository_ctx.attr.url_dict[dict_key]

        archive_name = _get_archive_name(url, archive_suffix = ".whl")
        file_name = archive_name + repository_ctx.attr.extension

        repository_ctx.download(
            url = tf_mirror_urls(url),
            output = file_name,
            sha256 = sha256,
        )
        repository_ctx.extract(
            archive = file_name,
            stripPrefix = repository_ctx.attr.strip_prefix,
        )
        repository_ctx.delete(file_name)

        repository_ctx.file(
            "BUILD",
            repository_ctx.read(repository_ctx.attr.build_file),
        )
    else:
        # If no CUDA version is found, use the dummy build file if present.
        repository_ctx.file(
            "BUILD",
            repository_ctx.read(repository_ctx.attr.dummy_build_file or repository_ctx.attr.build_file),
        )

_cuda_wheel = repository_rule(
    implementation = _cuda_wheel_impl,
    attrs = {
        "sha256_dict": attr.string_dict(mandatory = True),
        "url_dict": attr.string_dict(mandatory = True),
        "build_file": attr.label(),
        "dummy_build_file": attr.label(),
        "strip_prefix": attr.string(),
        "extension": attr.string(default = ".zip"),
    },
    environ = ["TF_CUDA_VERSION"],
)

def cuda_wheel(name, sha256_dict, url_dict, **kwargs):
    _cuda_wheel(
        name = name,
        sha256_dict = sha256_dict,
        url_dict = url_dict,
        **kwargs
    )

def _get_relative_url_dict(dist_info):
    relative_url_dict = {}
    for arch in _REDIST_ARCH_DICT.keys():
        # CUDNN JSON might contain paths for each CUDA version.
        if "relative_path" not in dist_info[arch]:
            for cuda_version, data in dist_info[arch].items():
                relative_url_dict["{cuda_version}_{arch}" \
                    .format(
                    cuda_version = cuda_version,
                    arch = _REDIST_ARCH_DICT[arch],
                )] = [data["relative_path"], data["sha256"]]
        else:
            relative_url_dict[_REDIST_ARCH_DICT[arch]] = [
                dist_info[arch]["relative_path"],
                dist_info[arch]["sha256"],
            ]
    return relative_url_dict

def _get_cuda_archive(
        repo_name,
        dist_dict,
        dist_name,
        build_file = None,
        build_template = None,
        dummy_build_file = None,
        is_cudnn_dist = False):
    if dist_name in dist_dict.keys():
        return cuda_http_archive(
            name = repo_name,
            dist_version = dist_dict[dist_name]["version"],
            build_file = Label(build_file) if build_file else None,
            build_template = Label(build_template) if build_template else None,
            dummy_build_file = Label(dummy_build_file) if dummy_build_file else None,
            relative_url_dict = _get_relative_url_dict(dist_dict[dist_name]),
            is_cudnn_dist = is_cudnn_dist,
        )
    else:
        return cuda_http_archive(
            name = repo_name,
            dist_version = "",
            build_file = Label(build_file) if build_file else None,
            build_template = Label(build_template) if build_template else None,
            dummy_build_file = Label(dummy_build_file) if dummy_build_file else None,
            relative_url_dict = {"": []},
            is_cudnn_dist = is_cudnn_dist,
        )

def cuda_distributives(cuda_nccl_wheel_dict):
    nccl_artifacts_dict = {"sha256_dict": {}, "url_dict": {}}
    for cuda_version, nccl_wheel_info in cuda_nccl_wheel_dict.items():
        for arch in _OS_ARCH_DICT.values():
            if arch in nccl_wheel_info.keys():
                cuda_version_to_arch_key = "%s-%s" % (cuda_version, arch)
                nccl_artifacts_dict["sha256_dict"][cuda_version_to_arch_key] = nccl_wheel_info[arch]["sha256"]
                nccl_artifacts_dict["url_dict"][cuda_version_to_arch_key] = nccl_wheel_info[arch]["url"]

    cuda_wheel(
        name = "cuda_nccl",
        sha256_dict = nccl_artifacts_dict["sha256_dict"],
        url_dict = nccl_artifacts_dict["url_dict"],
        build_file = Label("//third_party/gpus/cuda:cuda_nccl.BUILD"),
        dummy_build_file = Label("//third_party/gpus/cuda:cuda_nccl_dummy.BUILD"),
        strip_prefix = "nvidia/nccl",
    )

    cuda_distributives = get_cuda_distributives()
    cudnn_distributives = get_cudnn_distributives()

    _get_cuda_archive(
        repo_name = "cuda_cccl",
        dist_dict = cuda_distributives,
        dist_name = "cuda_cccl",
        build_file = "//third_party/gpus/cuda:cuda_cccl.BUILD",
    )
    _get_cuda_archive(
        repo_name = "cuda_cublas",
        dist_dict = cuda_distributives,
        dist_name = "libcublas",
        build_template = "//third_party/gpus/cuda:cuda_cublas.BUILD.tpl",
        dummy_build_file = "//third_party/gpus/cuda:cuda_cublas_dummy.BUILD",
    )
    _get_cuda_archive(
        repo_name = "cuda_cudart",
        dist_dict = cuda_distributives,
        dist_name = "cuda_cudart",
        build_template = "//third_party/gpus/cuda:cuda_cudart.BUILD.tpl",
        dummy_build_file = "//third_party/gpus/cuda:cuda_cudart_dummy.BUILD",
    )
    _get_cuda_archive(
        repo_name = "cuda_cudnn",
        dist_dict = cudnn_distributives,
        dist_name = "cudnn",
        build_template = "//third_party/gpus/cuda:cuda_cudnn.BUILD.tpl",
        dummy_build_file = "//third_party/gpus/cuda:cuda_cudnn_dummy.BUILD",
        is_cudnn_dist = True,
    )
    _get_cuda_archive(
        repo_name = "cuda_cufft",
        dist_dict = cuda_distributives,
        dist_name = "libcufft",
        build_template = "//third_party/gpus/cuda:cuda_cufft.BUILD.tpl",
        dummy_build_file = "//third_party/gpus/cuda:cuda_cufft_dummy.BUILD",
    )
    _get_cuda_archive(
        repo_name = "cuda_cupti",
        dist_dict = cuda_distributives,
        dist_name = "cuda_cupti",
        build_template = "//third_party/gpus/cuda:cuda_cupti.BUILD.tpl",
        dummy_build_file = "//third_party/gpus/cuda:cuda_cupti_dummy.BUILD",
    )
    _get_cuda_archive(
        repo_name = "cuda_curand",
        dist_dict = cuda_distributives,
        dist_name = "libcurand",
        build_template = "//third_party/gpus/cuda:cuda_curand.BUILD.tpl",
        dummy_build_file = "//third_party/gpus/cuda:cuda_curand_dummy.BUILD",
    )
    _get_cuda_archive(
        repo_name = "cuda_cusolver",
        dist_dict = cuda_distributives,
        dist_name = "libcusolver",
        build_template = "//third_party/gpus/cuda:cuda_cusolver.BUILD.tpl",
        dummy_build_file = "//third_party/gpus/cuda:cuda_cusolver_dummy.BUILD",
    )
    _get_cuda_archive(
        repo_name = "cuda_cusparse",
        dist_dict = cuda_distributives,
        dist_name = "libcusparse",
        build_template = "//third_party/gpus/cuda:cuda_cusparse.BUILD.tpl",
        dummy_build_file = "//third_party/gpus/cuda:cuda_cusparse_dummy.BUILD",
    )
    _get_cuda_archive(
        repo_name = "cuda_nvcc",
        dist_dict = cuda_distributives,
        dist_name = "cuda_nvcc",
        build_file = "//third_party/gpus/cuda:cuda_nvcc.BUILD",
    )
    _get_cuda_archive(
        repo_name = "cuda_nvjitlink",
        dist_dict = cuda_distributives,
        dist_name = "libnvjitlink",
        build_template = "//third_party/gpus/cuda:cuda_nvjitlink.BUILD.tpl",
        dummy_build_file = "//third_party/gpus/cuda:cuda_nvjitlink_dummy.BUILD",
    )
    _get_cuda_archive(
        repo_name = "cuda_nvml",
        dist_dict = cuda_distributives,
        dist_name = "cuda_nvml_dev",
        build_file = "//third_party/gpus/cuda:cuda_nvml.BUILD",
    )
    _get_cuda_archive(
        repo_name = "cuda_nvprune",
        dist_dict = cuda_distributives,
        dist_name = "cuda_nvprune",
        build_file = "//third_party/gpus/cuda:cuda_nvprune.BUILD",
    )
    _get_cuda_archive(
        repo_name = "cuda_nvtx",
        dist_dict = cuda_distributives,
        dist_name = "cuda_nvtx",
        build_file = "//third_party/gpus/cuda:cuda_nvtx.BUILD",
    )
