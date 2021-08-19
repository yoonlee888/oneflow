"""
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
"""
import os
import unittest

import numpy as np

import oneflow
import oneflow as flow
import oneflow.framework.graph_build_util as graph_build_util
import oneflow.unittest


class SubModule(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = flow.nn.Conv2d(1, 1, 5)
        self.relu = flow.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x


class CustomModule(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = SubModule()
        self.fc1 = flow.nn.Linear(36, 4)
        self.register_buffer("dummy_buff", flow.Tensor(1, 4))

    def forward(self, x):
        x = self.layer(x)
        x = oneflow.F.flatten(x, 1)
        x = self.fc1(x) + self.dummy_buff
        return x


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestGraphWithSysConf(flow.unittest.TestCase):
    def test_graph_config(test_case):
        flow.boxing.enable_fusion(True)

        flow.boxing.nccl.set_fusion_threshold_mbytes(800)
        flow.boxing.nccl.set_fusion_max_ops_num(10)
        flow.boxing.nccl.allow_fuse_all_reduce(True)
        flow.boxing.nccl.allow_fuse_reduce_scatter(True)
        flow.boxing.nccl.allow_fuse_all_gather(True)
        flow.boxing.nccl.allow_fuse_reduce(True)
        flow.boxing.nccl.allow_fuse_broadcast(True)
        flow.boxing.nccl.allow_fuse_mixed_ops(True)
        flow.boxing.nccl.enable_use_buffer_to_fuse_all_reduce(True)
        flow.boxing.nccl.set_stream_num(3)
        flow.boxing.nccl.enable_all_to_all(True)
        flow.boxing.nccl.enable_use_compute_stream(True)

        flow.backends.cudnn.set_reserved_mem_mbytes(1000)

        flow.utils.load_library("")

        class CustomGraphSysConf(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = CustomModule()
                self.config.enable_amp(True)
                loss_scale = flow.nn.graph.amp.DynamicLossScalePolicy(3000, 1000, 3.0)
                self.config.amp_add_loss_scale_policy(loss_scale)
                self.config.enable_fuse_add_to_output(True)

            def build(self, x):
                x = self.m(x)
                return x

        g = CustomGraphSysConf()

        print("backends conf: \n", g._backends_conf_proto)
        print("graph conf: \n", g._graph_conf_proto)