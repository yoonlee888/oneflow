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
#include <cstdint>
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct FusedTrilScaleSoftmaxMaskScaleState : public AutoGradCaptureState {
  bool x_requires_grad = true;
  int64_t diagonal = 0;
  float tril_fill_value = 0;
  float tril_scale_value = 1;
  float mask_scale_value = 1;
};

class FusedTrilScaleSoftmaxMaskScale : public OpExprGradFunction<FusedTrilScaleSoftmaxMaskScaleState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(FusedTrilScaleSoftmaxMaskScaleState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);
    ctx->x_requires_grad = inputs.at(0)->requires_grad();
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->diagonal = JUST(composed_attrs.GetAttr<int64_t>("diagonal"));
    ctx->tril_fill_value = JUST(composed_attrs.GetAttr<float>("tril_fill_value"));
    ctx->tril_scale_value = JUST(composed_attrs.GetAttr<float>("tril_scale_value"));
    ctx->mask_scale_value = JUST(composed_attrs.GetAttr<float>("mask_scale_value"));

    ctx->SaveTensorForBackward(outputs.at(1));
    ctx->SaveTensorForBackward(inputs.at(1));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const FusedTrilScaleSoftmaxMaskScaleState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    if (!ctx->x_requires_grad) { return Maybe<void>::Ok(); }
    CHECK_EQ_OR_RETURN(out_grads.size(), 2);
    in_grads->resize(2);
    const auto& softmax_y = ctx->SavedTensors().at(0);
    const auto& mask = ctx->SavedTensors().at(1);
    const int64_t& diagonal = ctx->diagonal;
    const float& tril_scale_value = ctx->tril_scale_value;
    const float& mask_scale_value = ctx->mask_scale_value;
    const std::shared_ptr<oneflow::one::Tensor>& fused_tril_scale_softmax_mask_scale =
        JUST(functional::FusedTrilScaleSoftmaxMaskScaleGrad(softmax_y, out_grads.at(1), mask, diagonal, 
                                                            tril_scale_value, mask_scale_value));
    in_grads->at(0) = fused_tril_scale_softmax_mask_scale;
    
    return Maybe<void>::Ok();
  }

 private:
  AttrMap base_attrs_;
};

REGISTER_OP_EXPR_GRAD_FUNCTION("fused_tril_scale_softmax_mask_scale", FusedTrilScaleSoftmaxMaskScale);

}  // namespace one
}  // namespace oneflow