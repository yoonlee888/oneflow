/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef ONEFLOW_XRT_TENSORRT_COMMON_H_
#define ONEFLOW_XRT_TENSORRT_COMMON_H_

#include "NvInferVersion.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

#ifdef NV_TENSORRT_MAJOR
#if NV_TENSORRT_MAJOR > 7
#define TRT_OPTIONAL_NOEXCEPT noexcept
#else
#define TRT_OPTIONAL_NOEXCEPT
#endif
#else
#define TRT_OPTIONAL_NOEXCEPT
#endif

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_TENSORRT_COMMON_H_
