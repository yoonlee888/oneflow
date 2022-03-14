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
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

struct SinCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  int32_t index_x;
};

class Sin : public OpExprGradFunction<SinCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(SinCaptureState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrMap& attrs) const override {
    ctx->index_x = ctx->SaveTensorForBackward(inputs.at(0));
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const SinCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    LOG(WARNING) << "=====================";
    in_grads->resize(1);
    const auto& x = JUST(functional::Cos(ctx->SavedTensors().at(ctx->index_x)));
    in_grads->at(0) = JUST(functional::Mul(out_grads.at(0), x));
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("sin", Sin);

}  // namespace one
}  // namespace oneflow
