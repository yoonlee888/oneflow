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

#include "oneflow/api/cpp/tests/api_test.h"
#include <cstddef>
#include <random>
#ifdef __linux__

#include <unistd.h>        // readlink
#include <linux/limits.h>  // PATH_MAX

#endif

namespace oneflow_api {

namespace {

std::mt19937 rng(std::random_device{}());

}

Shape RandomShape() {
  std::uniform_int_distribution<> dist_ndim(1, 4), dist_dims(16, 64);
  std::vector<std::int64_t> dims(dist_ndim(rng), 0);
  for (auto& x : dims) { x = dist_dims(rng); }
  return Shape(dims);
}

template<typename T>
std::vector<T> RandomData(size_t size) {
  std::uniform_int_distribution<> dist(-100, 100);
  std::vector<T> data(size);
  for (auto& x : data) { x = static_cast<T>(dist(rng)); }
  return data;
}
#define REGISTER_RANDOM_DATA(cpp_dtype) template std::vector<cpp_dtype> RandomData(size_t size);

REGISTER_RANDOM_DATA(float)
REGISTER_RANDOM_DATA(double)
REGISTER_RANDOM_DATA(int8_t)
REGISTER_RANDOM_DATA(int32_t)
REGISTER_RANDOM_DATA(int64_t)

std::string GetExeDir() {
#ifdef __linux__
  char result[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
  std::string exe_path(result, (count > 0) ? count : 0);
  size_t pos = exe_path.rfind('/');
  if (pos != std::string::npos) { return exe_path.substr(0, pos); }
#endif
  return ".";
}

}  // namespace oneflow_api
