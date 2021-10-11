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

import unittest

import numpy as np

import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest
import oneflow.optim as optim


class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y


class TestTensorDataset(flow.unittest.TestCase):
    def test_tensor_dataset(test_case):
        device = flow.device("cuda")

        num_inputs = 2
        num_examples = 1000000
        true_w = [2, -3.4]
        true_b = 4.2
        net = LinearNet(num_inputs)
        flow.nn.init.normal_(net.linear.weight, mean=0, std=0.01)
        flow.nn.init.constant_(net.linear.bias, val=0)
        net.to(device)
        loss = nn.MSELoss()
        loss.to(device)
        optimizer = optim.SGD(net.parameters(), lr=0.03)

        features = flow.tensor(
            np.random.normal(0, 1, (num_examples, num_inputs)), dtype=flow.float
        )
        labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
        labels += flow.tensor(
            np.random.normal(0, 0.01, size=labels.size()), dtype=flow.float
        )

        batch_size = 128
        dataset = flow.utils.data.TensorDataset(features, labels)
        data_iter = flow.utils.data.DataLoader(
            dataset, batch_size, shuffle=False, num_workers=0
        )
        num_epochs = 1
        print_interval = num_examples // batch_size // 10

        X = flow.tensor(np.random.randn(batch_size, 2), dtype=flow.float32)
        y = flow.ones((batch_size), dtype=flow.int64)
        for epoch in range(1, num_epochs + 1):
            for i, (X, y) in enumerate(data_iter, 1):
            # for i in range(num_examples // batch_size):
                X = X.to(device, dtype=flow.float32)
                y = y.to(device, dtype=flow.float32)
                output = net(X)
                l = loss(output, y).sum()
                optimizer.zero_grad()
                l.backward()
                optimizer.step()

                if i % print_interval == 0:
                    print("epoch: %d  step: %5d  loss: %.3f " % (epoch, i, l.numpy()))


if __name__ == "__main__":
    unittest.main()
    # 1 epoch training log(branch: master commit 195c08f36e194aba67ed0de6bc57bcd09418d760)
    # speed test 
    # batch_size 128   >>   140.3 144.3 163.5

    # speed test without dataloader
    # batch_size 128   >>   11.6  11.2  11.1
