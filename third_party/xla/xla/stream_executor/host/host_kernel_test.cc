/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/stream_executor/host/host_kernel.h"

#include <cstdint>
#include <vector>

#include "xla/service/hlo_runner.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/host/host_kernel_c_api.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor::host {

static SE_HOST_KernelError* AddI32(const SE_HOST_KernelCallFrame* call_frame) {
  SE_HOST_KernelArg& lhs = call_frame->args[0];
  SE_HOST_KernelArg& rhs = call_frame->args[1];
  SE_HOST_KernelArg& out = call_frame->args[2];

  int32_t* lhs_ptr = reinterpret_cast<int32_t*>(lhs.data);
  int32_t* rhs_ptr = reinterpret_cast<int32_t*>(rhs.data);
  int32_t* out_ptr = reinterpret_cast<int32_t*>(out.data);

  const auto zstep = call_frame->thread_dims->x * call_frame->thread_dims->y;
  const auto ystep = call_frame->thread_dims->x;

  uint64_t i = call_frame->thread->x + call_frame->thread->y * ystep +
               call_frame->thread->z * zstep;
  *(out_ptr + i) = *(lhs_ptr + i) + *(rhs_ptr + i);

  return nullptr;
}

TEST(HostKernelTest, Addition1D) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("Host"));
  xla::HloRunner runner(platform);

  HostKernel kernel(/*arity=*/3, AddI32,
                    runner.backend().eigen_intra_op_thread_pool());

  std::vector<int32_t> lhs = {1, 2, 3, 4};
  std::vector<int32_t> rhs = {5, 6, 7, 8};
  std::vector<int32_t> out = {0, 0, 0, 0};

  DeviceMemoryBase lhs_mem(lhs.data(), lhs.size() * sizeof(int32_t));
  DeviceMemoryBase rhs_mem(rhs.data(), rhs.size() * sizeof(int32_t));
  DeviceMemoryBase out_mem(out.data(), out.size() * sizeof(int32_t));
  std::vector<DeviceMemoryBase> args = {lhs_mem, rhs_mem, out_mem};

  TF_ASSERT_OK(kernel.Launch(ThreadDim(4), args));

  std::vector<int32_t> expected = {6, 8, 10, 12};
  EXPECT_EQ(out, expected);
}

TEST(HostKernelTest, Addition3D) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("Host"));
  xla::HloRunner runner(platform);

  HostKernel kernel(/*arity=*/3, AddI32,
                    runner.backend().eigen_intra_op_thread_pool());

  // Lets pretend there is a 3-dimensional 2x2x3 data
  std::vector<int32_t> lhs = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int32_t> rhs = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
  std::vector<int32_t> out = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  DeviceMemoryBase lhs_mem(lhs.data(), lhs.size() * sizeof(int32_t));
  DeviceMemoryBase rhs_mem(rhs.data(), rhs.size() * sizeof(int32_t));
  DeviceMemoryBase out_mem(out.data(), out.size() * sizeof(int32_t));
  std::vector<DeviceMemoryBase> args = {lhs_mem, rhs_mem, out_mem};

  TF_ASSERT_OK(kernel.Launch(ThreadDim(2, 2, 3), args));

  std::vector<int32_t> expected = {11, 13, 15, 17, 19, 21,
                                   23, 25, 27, 29, 31, 33};
  EXPECT_EQ(out, expected);
}

}  // namespace stream_executor::host
