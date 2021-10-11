import os
import unittest

import oneflow.unittest
import oneflow as flow
import oneflow.utils.vision as vision
import oneflow.nn as nn
import oneflow.optim as optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(flow._C.relu(self.conv1(x)))
        x = self.pool(flow._C.relu(self.conv2(x)))
        x = flow.flatten(x, 1)  # flatten all dimensions except batch
        x = flow._C.relu(self.fc1(x))
        x = flow._C.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test(test_case, batch_size):
    if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
        device = flow.device("cpu")
    else:
        device = flow.device("cuda")
    net = Net()
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    # transform = vision.transforms.Compose(
    #     [
    #         vision.transforms.ToTensor(),
    #     ]
    # )

    transform = vision.transforms.Compose(
        [
            vision.transforms.ToTensor(),
            vision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_epoch = 1
    data_dir = os.path.join(
        os.getenv("ONEFLOW_TEST_CACHE_DIR", "./data-test-flow"), "cifar10"
    )

    trainset = vision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    trainloader = flow.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print_interval = 50000 // batch_size // 10 # train dataset contains 50000 images
    final_loss = 0
    for epoch in range(1, train_epoch + 1):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 1):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(dtype=flow.float32, device=device)
            labels = labels.to(dtype=flow.int64, device=device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.cpu().detach().numpy()
            if i % print_interval == 0:  # print every print_interval mini-batches
                final_loss = running_loss / print_interval
                print("epoch: %d  step: %5d  loss: %.3f " % (epoch, i, final_loss))
                running_loss = 0.0

    print("final loss : ", final_loss)


@flow.unittest.skip_unless_1n1d()
class TestCifarDataset(flow.unittest.TestCase):
    def test_cifar_dataset(test_case):
        test(test_case, 256)


if __name__ == "__main__":
    unittest.main()
    # 1 epoch training log
    # speed test branch: master commit 195c08f36e194aba67ed0de6bc57bcd09418d760
    # batch_size 4   >>   115.1 118.7 113.0
    # batch_size 256 >>   52.3 52.9 52.9

    # speed test without normalize
    # batch_size 4   >>   85.6  84.3 86.2
    # batch_size 256 >>   23.3  23.0 22.9


