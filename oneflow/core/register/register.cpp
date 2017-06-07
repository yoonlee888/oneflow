#include "oneflow/core/register/register.h"

namespace oneflow {

Regst::Regst() {
  piece_id_ = std::numeric_limits<uint64_t>::max();
  model_version_id_ = std::numeric_limits<uint64_t>::max();
  regst_id_ = std::numeric_limits<uint64_t>::max();
}

void Regst::ForEachLbn(std::function<void(const std::string&)> func) {
  for (const auto& pair : lbn2blob_) {
    func(pair.first);
  }
}

Blob* Regst::GetBlobPtrFromLbn(const std::string& lbn) {
  auto it = lbn2blob_.find(lbn);
  if (it == lbn2blob_.end()) {
    return nullptr;
  } else {
    return it->second.get();
  }
}

} // namespace oneflow
