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
#include <cstring>
#include "oneflow/core/framework/consistency_check.h"
#include "oneflow/core/intrusive/flat_msg.h"
#include "oneflow/core/job/rank_group.h"
#include "oneflow/core/framework/transport_util.h"
#include "oneflow/core/job/rank_group_scope.h"
#include "oneflow/core/framework/synced_symbol_map.h"
#include "oneflow/core/framework/sync_symbol_nd_sbp.h"
#include "oneflow/core/framework/sync_symbol_parallel_desc.h"

namespace oneflow {

namespace {

class FlatMetaInfoConsistency;

class CheckMetaInfoConsistencyAsyncTransportCtx : public AsyncTransportCtx {
 public:
  CheckMetaInfoConsistencyAsyncTransportCtx(const TransportToken& transport_token,
                                            const Symbol<ParallelDesc>& placement,
                                            const Optional<Symbol<cfg::NdSbp>>& nd_sbp,
                                            const Optional<Symbol<cfg::NdSbp>>& grad_nd_sbp)
      : AsyncTransportCtx(transport_token),
        placement_(placement),
        nd_sbp_(nd_sbp),
        grad_nd_sbp_(grad_nd_sbp) {}

  ~CheckMetaInfoConsistencyAsyncTransportCtx() override;

  Maybe<void> PrepareSendBufferAndCallback(int64_t rank, void** buffer, std::size_t* size,
                                           std::function<void()>* Callback) override;

  Maybe<void> PrepareRecvBufferAndCallback(int64_t rank, void** buffer, std::size_t* size,
                                           std::function<void()>* Callback) override;

  Maybe<void> Check() const;

 private:
  Symbol<ParallelDesc> placement_;
  Optional<Symbol<cfg::NdSbp>> nd_sbp_;
  Optional<Symbol<cfg::NdSbp>> grad_nd_sbp_;
  std::shared_ptr<FlatMetaInfoConsistency> flat_meta_info_consistency_;
};

// clang-format off
FLAT_MSG_BEGIN(FlatMetaInfoConsistency);
 public:
  static Maybe<FlatMetaInfoConsistency> New() {
    const auto& consistency = std::make_shared<FlatMetaInfoConsistency>();
    consistency->clear();
    return consistency;
  }
  static Maybe<FlatMetaInfoConsistency> New(const Symbol<ParallelDesc>& placement,
    const Optional<Symbol<cfg::NdSbp>>& nd_sbp, const Optional<Symbol<cfg::NdSbp>>& grad_nd_sbp) {
    const auto& consistency = std::make_shared<FlatMetaInfoConsistency>();
    consistency->clear();
    JUST(consistency->Init(placement, nd_sbp, grad_nd_sbp));
    return consistency;
  }

  Maybe<void> Check(const Symbol<ParallelDesc>& placement,
    const Optional<Symbol<cfg::NdSbp>>& nd_sbp, const Optional<Symbol<cfg::NdSbp>>& grad_nd_sbp) {
    
    const auto& this_placement =
        JUST(SyncedSymbolMap<ParallelDesc>::Symbol4SyncedSymbolId(
            this->placement_symbol_id()));
    CHECK_OR_RETURN(this_placement == placement) << "Each rank must have same input placement";
    CHECK_EQ_OR_RETURN(nd_sbp.has_value(), this->has_nd_sbp_symbol_id());
    if (this->has_nd_sbp_symbol_id()) {
      const auto& that_nd_sbp =
          JUST(SyncedSymbolMap<cfg::NdSbp>::Symbol4SyncedSymbolId(
              this->nd_sbp_symbol_id()));
      const auto& this_nd_sbp = JUST(nd_sbp);
      CHECK_OR_RETURN(this_nd_sbp == that_nd_sbp)  << "Each rank must have same input sbp";
    }
    CHECK_EQ_OR_RETURN(grad_nd_sbp.has_value(), this->has_grad_nd_sbp_symbol_id());
    if (this->has_grad_nd_sbp_symbol_id()) {
       const auto& that_grad_nd_sbp =
          JUST(SyncedSymbolMap<cfg::NdSbp>::Symbol4SyncedSymbolId(
              this->grad_nd_sbp_symbol_id()));
      const auto& this_grad_nd_sbp = JUST(grad_nd_sbp);
      CHECK_OR_RETURN(this_grad_nd_sbp == that_grad_nd_sbp) << "Each rank must have same input grad sbp";
    }
    return Maybe<void>::Ok();
  }
 private:
  Maybe<void> Init(const Symbol<ParallelDesc>& placement, const Optional<Symbol<cfg::NdSbp>>& nd_sbp,
    const Optional<Symbol<cfg::NdSbp>>& grad_nd_sbp) {
    this->set_placement_symbol_id(
        JUST(SyncedSymbolMap<ParallelDesc>::FindOrSync(placement, &SyncSymbolParallelDesc)));
    if (nd_sbp.has_value()) {
      this->set_nd_sbp_symbol_id(
          JUST(SyncedSymbolMap<cfg::NdSbp>::FindOrSync(JUST(nd_sbp), &SyncSymbolNdSbp)));
    }
    if (grad_nd_sbp.has_value()) {
      this->set_grad_nd_sbp_symbol_id(
          JUST(SyncedSymbolMap<cfg::NdSbp>::FindOrSync(JUST(grad_nd_sbp), &SyncSymbolNdSbp)));
    }
    return Maybe<void>::Ok();
  }
  FLAT_MSG_DEFINE_OPTIONAL(uint64_t, placement_symbol_id);
  FLAT_MSG_DEFINE_OPTIONAL(uint64_t, nd_sbp_symbol_id);
  FLAT_MSG_DEFINE_OPTIONAL(uint64_t, grad_nd_sbp_symbol_id);
FLAT_MSG_END(FlatMetaInfoConsistency);
// clang-format off

CheckMetaInfoConsistencyAsyncTransportCtx::~CheckMetaInfoConsistencyAsyncTransportCtx() {}

Maybe<void> CheckMetaInfoConsistencyAsyncTransportCtx::PrepareSendBufferAndCallback(
    int64_t rank, void** buffer, std::size_t* size, std::function<void()>* Callback) {
  const auto& meta_info_consistency =
      JUST(FlatMetaInfoConsistency::New(placement_, nd_sbp_, grad_nd_sbp_));
  *buffer = meta_info_consistency.get();
  *size = sizeof(FlatMetaInfoConsistency);
  *Callback = [meta_info_consistency] {};
  return Maybe<void>::Ok();
}

Maybe<void> CheckMetaInfoConsistencyAsyncTransportCtx::PrepareRecvBufferAndCallback(
    int64_t rank, void** buffer, std::size_t* size, std::function<void()>* Callback) {
  const auto& flat_meta_info_consistency = JUST(FlatMetaInfoConsistency::New());
  *buffer = flat_meta_info_consistency.get();
  *size = sizeof(FlatMetaInfoConsistency);
  *Callback = [flat_meta_info_consistency]() {};
  flat_meta_info_consistency_ = flat_meta_info_consistency;
  return Maybe<void>::Ok();
}

Maybe<void> CheckMetaInfoConsistencyAsyncTransportCtx::Check() const {
  if (!flat_meta_info_consistency_) { return Maybe<void>::Ok(); }
  JUST(flat_meta_info_consistency_->Check(
      placement_, nd_sbp_, grad_nd_sbp_));
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> DataConsistencyCheck(const void* buffer_ptr, size_t buffer_size,
                                 Symbol<ParallelDesc> placement) {
  const auto& rank_group = JUST(RankGroup::New(placement));

  std::vector<char> recv_buffer(buffer_size);
  char* recv_ptr = recv_buffer.data();

  TransportToken transport_token = JUST(TransportToken::NewTransportToken(kTransportTokenTypeData));
  NaiveAsyncTransportCtx ctx(
      transport_token,
      [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        *buffer = const_cast<void*>(buffer_ptr);
        *size = buffer_size;
        *Cb = [] {};
        return Maybe<void>::Ok();
      },
      [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        *buffer = recv_ptr;
        *size = buffer_size;
        *Cb = [] {};
        return Maybe<void>::Ok();
      });
  JUST(TransportUtil::SendToNextRankInRing(rank_group, transport_token, &ctx));
  JUST(TransportUtil::ReceiveFromPrevRankInRing(rank_group, transport_token, &ctx));
  JUST(TransportUtil::WaitUntilDoneOrTimeout(ctx, TransportUtil::TimeoutSeconds()));
  CHECK_OR_RETURN(std::memcmp(buffer_ptr, reinterpret_cast<const void*>(recv_ptr), buffer_size)
                  == 0)
      << "Each rank must have same input sequence or numpy array";
  return Maybe<void>::Ok();
}

Maybe<void> MetaInfoConsistencyCheck(const Symbol<ParallelDesc>& placement,
    const Optional<Symbol<cfg::NdSbp>>& nd_sbp, const Optional<Symbol<cfg::NdSbp>>& grad_nd_sbp) {
  const auto& rank_group = JUST(RankGroupScope::CurrentRankGroup());
  const auto& transport_token =
      JUST(TransportToken::NewTransportToken(kTransportTokenTypeCheckRankGroupConsistency));
  const auto& ctx = std::make_shared<CheckMetaInfoConsistencyAsyncTransportCtx>(
    transport_token, placement, nd_sbp, grad_nd_sbp);
  JUST(TransportUtil::SendToNextRankInRing(rank_group, transport_token, ctx.get()));
  JUST(TransportUtil::ReceiveFromPrevRankInRing(rank_group, transport_token, ctx.get()));
  JUST(TransportUtil::WaitUntilDoneOrTimeout(*ctx, TransportUtil::TimeoutSeconds()));
  JUST(ctx->Check());
  return Maybe<void>::Ok();
}

Maybe<void> MetaInfoConsistencyCheck(const Symbol<ParallelDesc>& placement,
    const Optional<Symbol<cfg::NdSbp>>& nd_sbp) {
  JUST(MetaInfoConsistencyCheck(placement, nd_sbp, Optional<Symbol<cfg::NdSbp>>()));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
