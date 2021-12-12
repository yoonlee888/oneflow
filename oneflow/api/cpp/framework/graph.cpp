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

#include "oneflow/api/common/ofblob.h"
#include "oneflow/api/common/scope.h"
#include "oneflow/api/cpp/framework/device.h"
#include "oneflow/api/cpp/framework/graph.h"
#include "oneflow/api/cpp/framework/shape.h"
#include "oneflow/api/cpp/framework/tensor.h"
#include "oneflow/api/common/job_build_and_infer_ctx.h"
#include <cstdio>
#include <fstream>
#include <istream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "oneflow/api/python/job_build/job_build_and_infer.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/multi_client_session_context.h"
#include "oneflow/core/framework/nn_graph.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/tensor_util.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job_conf.cfg.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/session.h"
#include "oneflow/core/operator/interface_blob_conf.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow_api {

namespace of = oneflow;

enum class XrtKind : int { kNone = 0, kTensorRT = 1, kOpenVINO = 2 };

namespace {

class CompileScope {
 public:
  CompileScope(const of::JobConfigProto& job_config, const of::Device& device, XrtKind kind) {
    const std::shared_ptr<of::Scope> scope = CHECK_JUST(of::MakeScope(job_config, device));
    CHECK_JUST(of::ThreadLocalScopeStackPush(scope));

    of::cfg::JobConfigProto job_config_cfg(job_config);
#ifdef WITH_OPENVINO
    if (kind == XrtKind::kOpenVINO) {
      *(job_config_cfg.mutable_xrt_config()->mutable_use_openvino()) = true;
    }
#endif
#ifdef WITH_TENSORRT
    if (kind == XrtKind::kTensorRT) {
      *(job_config_cfg.mutable_xrt_config()->mutable_use_tensorrt()) = true;
    }
#endif
    CHECK_JUST(of::JobBuildAndInferCtx_Open(job_config.job_name()));
    CHECK_JUST(of::CurJobBuildAndInferCtx_SetJobConf(job_config_cfg));
  }

  ~CompileScope() {
    CHECK_JUST(of::JobBuildAndInferCtx_Close());
    CHECK_JUST(of::ThreadLocalScopeStackPop());
  }

 private:
  of::LazyMode::Guard lazy_mode_enabled_guard{true};
};

std::shared_ptr<of::one::TensorTuple> ConvertToTensorTuple(
    const std::vector<std::shared_ptr<of::one::Tensor>>& tensors) {
  auto tensor_tuple = std::make_shared<of::one::TensorTuple>();
  for (const auto& tensor : tensors) { tensor_tuple->emplace_back(tensor); }
  return tensor_tuple;
}

std::string GetDeviceTag(const Device& device) {
  if (device.type() == "cuda") {
    return "gpu";
  } else {
    return "cpu";
  }
}

template<class T1, class T2>
const std::pair<std::vector<T1>, std::vector<T2>> Unzip(const of::HashMap<T1, T2>& hash_map) {
  std::vector<T1> vec1;
  std::vector<T2> vec2;
  for (const auto& entry : hash_map) {
    vec1.emplace_back(entry.first);
    vec2.emplace_back(entry.second);
  }
  return std::make_pair(vec1, vec2);
}

}  // namespace

class Graph::GraphImpl final {
 public:
  explicit GraphImpl(const std::string& model_path, const Device& device);
  explicit GraphImpl(const std::string& model_path);
  std::vector<Tensor> Forward(const std::vector<Tensor>& inputs);
  void set_batch_size(int batch_size) { batch_size_ = batch_size; }
  void enable_openvino() { xrt_kind_ = XrtKind::kTensorRT; }
  void enable_tensorrt() { xrt_kind_ = XrtKind::kOpenVINO; }

 private:
  oneflow::Maybe<void> Compile(const std::vector<Tensor>& inputs);
  oneflow::Maybe<std::vector<Tensor>> Run(const std::vector<Tensor>& inputs) const;
  oneflow::Maybe<void> AddOp(oneflow::OperatorConf op_conf);
  oneflow::Maybe<void> BuildGraph(const std::vector<Tensor>& inputs);
  oneflow::Maybe<void> LoadCheckpoint();
  oneflow::Maybe<void> RegisterTensors();

  std::shared_ptr<oneflow::NNGraph> graph_ = nullptr;
  const std::string model_path_;
  bool is_compiled_ = false;
  int batch_size_ = 0;
  XrtKind xrt_kind_ = XrtKind::kNone;
  Device device_;
  oneflow::Job job_;

  oneflow::HashMap<std::string, std::shared_ptr<oneflow::one::Tensor>> input_name_to_tensor_;
  oneflow::HashMap<std::string, std::shared_ptr<oneflow::one::Tensor>> output_name_to_tensor_;
  oneflow::HashMap<std::string, std::shared_ptr<oneflow::one::Tensor>> variable_op_name_to_tensor_;
  std::shared_ptr<oneflow::one::TensorTuple> output_tensor_tuple_;
  std::shared_ptr<oneflow::one::TensorTuple> parameter_tensor_tuple_;
};

std::vector<Tensor> Graph::Forward(const std::vector<Tensor>& inputs) {
  return graph_->Forward(inputs);
}

Graph::Graph(const std::string& model_path, const Device& device)
    : graph_(std::make_shared<GraphImpl>(model_path, device)) {}

Graph::Graph(const std::string& model_path) : graph_(std::make_shared<GraphImpl>(model_path)) {}

Graph::Graph(const std::shared_ptr<GraphImpl>& graph) : graph_(graph) {}

void Graph::set_batch_size(int batch_size) { graph_->set_batch_size(batch_size); }

void Graph::enable_openvino() { graph_->enable_openvino(); }

void Graph::enable_tensorrt() { graph_->enable_tensorrt(); }

Graph::GraphImpl::GraphImpl(const std::string& model_path, const Device& device)
    : model_path_(model_path), device_(device) {
  // TODO(zzk0): model_path is a directory, need to concatenate filename
  // we need a mlir model name.
  {
    std::ifstream input(model_path + "/model.pb");
    CHECK(input.is_open());
    CHECK(job_.ParseFromIstream(&input));
  }
  graph_ = std::make_shared<of::NNGraph>(job_.job_conf().job_name());
  of::Global<of::MultiClientSessionContext>::Get()->AddCGraph(graph_).GetOrThrow();
}

Graph::GraphImpl::GraphImpl(const std::string& model_path) : GraphImpl(model_path, Device("cpu")) {}

std::vector<Tensor> Graph::GraphImpl::Forward(const std::vector<Tensor>& inputs) {
  if (!is_compiled_) {
    Compile(inputs).GetOrThrow();
    is_compiled_ = true;
  }
  return Run(inputs).GetOrThrow();
}

of::Maybe<void> Graph::GraphImpl::Compile(const std::vector<Tensor>& inputs) {
  JUST(BuildGraph(inputs));
  JUST(LoadCheckpoint());
  JUST(RegisterTensors());
  JUST(graph_->CompileAndInitRuntime());
  return of::Maybe<void>::Ok();
}

of::Maybe<std::vector<Tensor>> Graph::GraphImpl::Run(const std::vector<Tensor>& inputs) const {
  const auto input_tensor_tuple = std::make_shared<of::one::TensorTuple>();
  for (const auto& tensor : inputs) { input_tensor_tuple->emplace_back(tensor.tensor_); }

  JUST(of::RunLazyNNGraph(*input_tensor_tuple, *output_tensor_tuple_, *parameter_tensor_tuple_,
                          graph_));
  JUST(of::SoftSyncNNGraphBuffers(*output_tensor_tuple_, graph_));

  std::vector<Tensor> outputs;
  for (const auto& tensor : *output_tensor_tuple_) { outputs.emplace_back(Tensor(tensor)); }
  return outputs;
}

of::Maybe<void> Graph::GraphImpl::AddOp(of::OperatorConf op_conf) {
  {
    const std::shared_ptr<of::Scope> scope = JUST(of::GetCurrentScope());
    op_conf.set_scope_symbol_id(scope->symbol_id().value_or(0));
  }
  op_conf.set_device_tag(GetDeviceTag(device_));
  if (batch_size_ > 0 && op_conf.has_input_conf()) {
    op_conf.mutable_input_conf()->mutable_blob_conf()->mutable_shape()->mutable_dim()->Set(
        0, batch_size_);
    std::cout << "Print input conf" << std::endl;
    std::cout << op_conf.ShortDebugString() << std::endl;
  }
  auto* ctx = JUST(of::GetCurInferCtx());
  JUST(ctx->AddAndInferConsistentOp(op_conf));
  return of::Maybe<void>::Ok();
}

of::Maybe<void> Graph::GraphImpl::BuildGraph(const std::vector<Tensor>& inputs) {
  CompileScope build_graph_scope(job_.job_conf(), *device_.device_->shared_from_symbol(),
                                 xrt_kind_);
  {
    // TODO(zzk0): remove this; used for input tensor order
    int input_tensor_order = 0;
    const of::OpGraph op_graph(job_);
    JUST(op_graph.ForEachOpNode([&](const of::OpNode& node) -> of::Maybe<void> {
      const of::OperatorConf& op_conf = node.op().op_conf();
      JUST(AddOp(op_conf));
      if (op_conf.has_input_conf()) {
        // TODO(zzk0): input tensor order
        input_name_to_tensor_[op_conf.name()] = inputs.at(input_tensor_order++).tensor_;
      } else if (op_conf.has_variable_conf()) {
        const of::LazyMode::Guard lazy_mode_disabled_guard{false};
        const of::VariableOpConf variable_conf = op_conf.variable_conf();
        variable_op_name_to_tensor_[op_conf.name()] = JUST(of::one::functional::Empty(
            of::Shape(variable_conf.shape()),
            JUST(of::DType::Get(static_cast<of::DataType>(variable_conf.data_type()))),
            *device_.device_));
      }
      return of::Maybe<void>::Ok();
    }));
  }
  JUST(of::CurJobBuildAndInferCtx_Complete());
  JUST(of::CurJobBuildAndInferCtx_Rebuild());
  {
    const std::shared_ptr<of::Job> complete_job = JUST(of::GetCurrentJob());
    const of::OpGraph complete_graph(*complete_job);
    JUST(complete_graph.ForEachOpNode([&](const of::OpNode& node) -> of::Maybe<void> {
      const of::LazyMode::Guard lazy_mode_disabled_guard{false};
      const of::OperatorConf& op_conf = node.op().op_conf();
      if (op_conf.has_output_conf()) {
        const of::InterfaceBlobConf blob_conf = op_conf.output_conf().blob_conf();
        output_name_to_tensor_[op_conf.name()] = JUST(of::one::functional::Empty(
            of::Shape(blob_conf.shape()),
            JUST(of::DType::Get(static_cast<of::DataType>(blob_conf.data_type()))),
            *device_.device_));
      }
      return of::Maybe<void>::Ok();
    }));
  }
  return of::Maybe<void>::Ok();
}

of::Maybe<void> Graph::GraphImpl::LoadCheckpoint() {
  for (const auto& variable_op_name_and_tensor : variable_op_name_to_tensor_) {
    const auto& variable_op_name = variable_op_name_and_tensor.first;
    const auto& variable_tensor = variable_op_name_and_tensor.second;
    const std::string variable_filename = model_path_ + "/" + variable_op_name + "/out";
    const std::string buffer = [&variable_filename]() {
      std::ifstream variable_file(variable_filename, std::ios::binary);
      CHECK(variable_file.is_open());
      std::stringstream ss;
      ss << variable_file.rdbuf();
      return ss.str();
    }();
    const auto& callback =
        std::make_shared<std::function<void(uint64_t)>>([&](uint64_t of_blob_ptr) {
          CHECK_JUST(of::BlobBufferCopyUtil<void>::From(
              of_blob_ptr, buffer.data(),
              variable_tensor->shape()->elem_cnt()
                  * of::GetSizeOfDataType(variable_tensor->dtype()->data_type())));
        });
    JUST(of::one::SyncAccessTensorWithTimeOut(variable_tensor, callback, "mut"));
  }

  return of::Maybe<void>::Ok();
}

of::Maybe<void> Graph::GraphImpl::RegisterTensors() {
  {
    const auto pair = Unzip(input_name_to_tensor_);
    const std::vector<std::string>& input_op_names = pair.first;
    const std::vector<std::shared_ptr<of::one::Tensor>>& input_tensors = pair.second;
    JUST(graph_->RegisterInputOpNamesAndTensors(input_op_names, input_tensors));
  }
  {
    const auto pair = Unzip(output_name_to_tensor_);
    const std::vector<std::string>& output_op_names = pair.first;
    const std::vector<std::shared_ptr<of::one::Tensor>>& output_tensors = pair.second;
    JUST(graph_->RegisterOutputOpNamesAndTensors(output_op_names, output_tensors));
    output_tensor_tuple_ = ConvertToTensorTuple(output_tensors);
  }
  {
    const auto pair = Unzip(variable_op_name_to_tensor_);
    const std::vector<std::string>& variable_op_names = pair.first;
    const std::vector<std::shared_ptr<of::one::Tensor>>& variable_tensors = pair.second;
    JUST(graph_->RegisterVariableOpNamesAndTensors(variable_op_names, variable_tensors));
    parameter_tensor_tuple_ = ConvertToTensorTuple(variable_tensors);
  }
  return of::Maybe<void>::Ok();
}

Graph Load(const std::string& model_path, const Device& device) {
  Graph graph(model_path, device);
  return graph;
}

Graph Load(const std::string& model_path) {
  const Device device = Device("cpu");
  return Load(model_path, device);
}

}  // namespace oneflow_api
