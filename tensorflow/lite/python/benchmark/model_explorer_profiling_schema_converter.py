# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts the profiling proto generated from benchmarking to a node overlay schema for the model explorer."""

import collections
from collections.abc import Sequence
import json
import re
from typing import Any

from absl import app
from absl import flags
from absl import logging

from tensorflow.lite.profiling.proto import profiling_info_pb2

_PROFILING_PROTO_PATHS = flags.DEFINE_list(
    "profiling_proto_paths",
    "",
    "Comma separated list of paths to the profiling protos. Only needed if the"
    " model has multiple signatures.",
)

_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", "", "Path to the output node overlay schema."
)

_MODEL_EXPLORER_JSON_TYPE = flags.DEFINE_enum(
    "model_explorer_json_type",
    "per_op_latency",
    ["per_op_latency", "op_type"],
    "Type of model explorer json to generate.",
)


def get_op_profile_json(
    op_profile: profiling_info_pb2.OpProfileData,
) -> dict[str, Any]:
  """Generates the Model Explorer json for the op profile.

  Args:
    op_profile: profiling_info_pb2.OpProfileData

  Returns:
    Model explorer json for the op profile.

  Raises:
    ValueError: If the op profile name is not in the expected format.
    ValueError: If the model explorer json type is not supported.
  """
  op_profile_key_re = re.findall(r":(\d+)$", op_profile.name)
  if not op_profile_key_re:
    raise ValueError("Op profile name is not in the expected format.")
  op_profile_key = op_profile_key_re[0]

  if _MODEL_EXPLORER_JSON_TYPE.value == "per_op_latency":
    return {
        op_profile_key: {
            "value": op_profile.avg_inference_microseconds / 1000.0
        }
    }
  elif _MODEL_EXPLORER_JSON_TYPE.value == "op_type":
    return {op_profile_key: {"value": op_profile.node_type}}
  else:
    raise ValueError(
        "Unsupported model explorer json type: %s"
        % _MODEL_EXPLORER_JSON_TYPE.value
    )


def main(argv: Sequence[str]) -> None:
  del argv  # Unused.

  if not _PROFILING_PROTO_PATHS.value:
    raise ValueError("At least one profiling proto path should be provided.")

  output_json = collections.defaultdict(dict)
  for proto_path in _PROFILING_PROTO_PATHS.value:
    with open(proto_path, "rb") as f:
      benchmark_profiling_proto = (
          profiling_info_pb2.BenchmarkProfilingData.FromString(f.read())
      )
      for (
          subgraph_profile
      ) in benchmark_profiling_proto.runtime_profile.subgraph_profiles:
        subgraph_profile_json = {}
        for op_profile in subgraph_profile.per_op_profiles:
          subgraph_profile_json.update(get_op_profile_json(op_profile))
        output_json[subgraph_profile.subgraph_name][
            "results"
        ] = subgraph_profile_json

        if _MODEL_EXPLORER_JSON_TYPE.value == "per_op_latency":
          output_json[subgraph_profile.subgraph_name]["gradient"] = [
              {"stop": 0, "bgColor": "green"},
              {"stop": 0.33, "bgColor": "yellow"},
              {"stop": 0.67, "bgColor": "orange"},
              {"stop": 1, "bgColor": "red"},
          ]

  if _OUTPUT_PATH.value:
    with open(_OUTPUT_PATH.value, "w") as f:
      json.dump(output_json, f)

  else:
    logging.info(json.dumps(output_json, indent=2))


if __name__ == "__main__":
  app.run(main)
