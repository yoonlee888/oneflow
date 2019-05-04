#include "oneflow/core/operator/reduce_sum_like_op.h"
#include "oneflow/core/operator/reduce_sbp_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

namespace {}

void ReduceSumLikeOp::InitFromOpConf() {
  CHECK(op_conf().has_reduce_sum_like_conf());
  EnrollInputBn("x");
  EnrollInputBn("like", false)->set_use_header_only(true);
  EnrollOutputBn("y");
  EnrollOutputBn("temp_storage", false);
}

const PbMessage& ReduceSumLikeOp::GetCustomizedConf() const {
  return op_conf().reduce_sum_like_conf();
}

void ReduceSumLikeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*) const {
  const ReduceSumLikeOpConf& conf = op_conf().reduce_sum_like_conf();
  BlobDesc* x_blob = GetBlobDesc4BnInOp("x");
  BlobDesc* like_blob = GetBlobDesc4BnInOp("like");
  if (conf.axis().empty()) { CHECK_EQ(x_blob->shape(), like_blob->shape()); }
  *GetBlobDesc4BnInOp("temp_storage") = *x_blob;
  GetBlobDesc4BnInOp("y")->CopyMetaFrom(*like_blob);
}

void ReduceSumLikeOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  *HasBatchDim4BnInOp("y") = *HasBatchDim4BnInOp("like");
  *HasBatchDim4BnInOp("temp_storage") = *HasBatchDim4BnInOp("like");
}

void ReduceSumLikeOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  int32_t num_axes = LogicalBlobDesc4Ibn("x").shape().NumAxes();
  auto IsReducedAxis =
      ReduceSbpUtil::MakePredicatorIsReducedAxis(op_conf().reduce_sum_like_conf().axis(), num_axes);
  FOR_RANGE(int64_t, i, 0, num_axes) {
    if (IsReducedAxis(i)) {
      SbpSignatureBuilder()
          .Split("x", i)
          .Broadcast("like")
          .PartialSum(output_bns())
          .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    } else {
      SbpSignatureBuilder()
          .Split(input_bns(), i)
          .Split(output_bns(), i)
          .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    }
  }
}

REGISTER_OP(OperatorConf::kReduceSumLikeConf, ReduceSumLikeOp);

}  // namespace oneflow
