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
#include "oneflow/core/common/global.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

void GlobalProcessCtx::GetCurrentMachineIdAndDeviceId(int64_t* machine_id, int64_t* device_id) {
  *machine_id = Rank();
  int64_t node_id = ThisNodeId();
  int64_t rank_index_offset = 0;
  for (int64_t i = 0; i < node_id; ++i) {
    rank_index_offset +=
        Global<ProcessCtx>::Get()->num_process_distribution_in_cluster().num_process(i);
  }
  *device_id = *machine_id - rank_index_offset;
}

int64_t GlobalProcessCtx::Rank() {
  CHECK_NOTNULL(Global<ProcessCtx>::Get());
  return Global<ProcessCtx>::Get()->rank();
}

int64_t GlobalProcessCtx::NodeSize() {
  CHECK_NOTNULL(Global<ProcessCtx>::Get());
  return Global<ProcessCtx>::Get()->node_size();
}

int64_t GlobalProcessCtx::ThisNodeId() {
  CHECK_NOTNULL(Global<ProcessCtx>::Get());
  return NodeId4Rank(Rank());
}

int64_t GlobalProcessCtx::NodeId4Rank(int64_t rank) {
  CHECK_NOTNULL(Global<ProcessCtx>::Get());
  int64_t rank_index_offset = 0;
  for (int64_t node_id = 0;
       node_id
       < Global<ProcessCtx>::Get()->num_process_distribution_in_cluster().num_process_size();
       ++node_id) {
    rank_index_offset +=
        Global<ProcessCtx>::Get()->num_process_distribution_in_cluster().num_process(node_id);
    if (rank < rank_index_offset) { return node_id; }
  }
  UNIMPLEMENTED();
}

int64_t GlobalProcessCtx::NumOfProcessOnNode() {
  if (Global<NumProcessDistribution>::Get() != nullptr) {
    return int64_t(Global<NumProcessDistribution>::Get()->num_process(0));
  }
  CHECK_NOTNULL(Global<ProcessCtx>::Get());
  int64_t node_id = ThisNodeId();
  return Global<ProcessCtx>::Get()->num_process_distribution_in_cluster().num_process(node_id);
}

const NumProcessDistribution& GlobalProcessCtx::NumProcessDistributionInCluster() {
  if (Global<NumProcessDistribution>::Get() != nullptr) {
    return *Global<NumProcessDistribution>::Get();
  }
  CHECK_NOTNULL(Global<ProcessCtx>::Get());
  return Global<ProcessCtx>::Get()->num_process_distribution_in_cluster();
}

bool GlobalProcessCtx::IsThisProcessMaster() {
  CHECK_NOTNULL(Global<ProcessCtx>::Get());
  return Global<ProcessCtx>::Get()->rank() == 0;
}

size_t GlobalProcessCtx::WorldSize() {
  CHECK_NOTNULL(Global<ProcessCtx>::Get());
  return Global<ProcessCtx>::Get()->ctrl_addr().size();
}

std::string GlobalProcessCtx::LogDirEntry() {
  CHECK_NOTNULL(Global<ProcessCtx>::Get());
  const auto& process_ctx = *Global<ProcessCtx>::Get();
  const auto& addr = process_ctx.ctrl_addr(process_ctx.rank());
  CHECK(addr.has_host());
  return addr.host() + "-" + std::to_string(addr.port()) + "-" + std::to_string(process_ctx.rank());
}

}  // namespace oneflow
