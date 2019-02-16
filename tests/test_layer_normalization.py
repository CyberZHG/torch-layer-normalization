import os
import tempfile
import random
from unittest import TestCase
import torch
import torch.nn as nn
from torch.utils import data
from torch_layer_normalization import LayerNormalization


class SimpleDataset(data.Dataset):

    def __init__(self, shape):
        self.shape = shape

    def __len__(self):
        return 100

    def __getitem__(self, _):
        x = torch.randn(self.shape)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        y = 0.5 * (x - mean) / std - 0.3
        return x, y


class TestLayerNormalization(TestCase):

    def test_first_step(self):
        net = LayerNormalization((2, 3))
        inputs = torch.Tensor([[[0.2, 0.1, 0.3], [0.5, 0.1, 0.1]]])
        outputs = net(inputs)
        expected = torch.Tensor([[[0.0, -1.22474487, 1.22474487], [1.41421356, -0.707106781, -0.707106781]]])
        self.assertTrue(expected.allclose(outputs), (expected, outputs))

    def test_first_step_without_param(self):
        net = LayerNormalization(3, gamma=False, beta=False)
        inputs = torch.Tensor([[[0.2, 0.1, 0.3], [0.5, 0.1, 0.1]]])
        outputs = net(inputs)
        expected = torch.Tensor([[[0.0, -1.22474487, 1.22474487], [1.41421356, -0.707106781, -0.707106781]]])
        self.assertTrue(expected.allclose(outputs), (expected, outputs))

    def test_all_zeros(self):
        shape = torch.randint(1, 100, (3,), dtype=torch.int32).tolist()
        normal_shape = shape[-1]
        net = LayerNormalization(normal_shape)
        inputs = torch.zeros(shape)
        outputs = net(inputs)
        self.assertTrue(inputs.allclose(outputs), (inputs, outputs))

    def test_fit(self):
        dim = torch.randint(2, 4, (1,), dtype=torch.int32).tolist()[0]
        shape = torch.randint(2, 5, (dim,), dtype=torch.int32).tolist()
        normal_shape = shape[-1]
        net = LayerNormalization(normal_shape)
        optimizer = torch.optim.Adam(net.parameters())
        criterion = nn.MSELoss()
        dataset = SimpleDataset(shape)
        for epoch in range(20):
            running_loss = 0.0
            for i in range(len(dataset)):
                x, y = dataset[i]
                optimizer.zero_grad()
                y_hat = net(x)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print('Epoch: %2d  Loss: %3.4f' % (epoch + 1, running_loss / len(dataset)))
        for i, (x, y) in enumerate(dataset):
            if i > 10:
                break
            y_hat = net(x)
            self.assertTrue(y.allclose(y_hat, rtol=0.0, atol=1e-3), (i, y, y_hat))

    def test_save_load(self):
        net = LayerNormalization(3)
        model_path = os.path.join(tempfile.gettempdir(), 'test_layer_normalization_%f.pth' % random.random())
        torch.save(net, model_path)
        net = torch.load(model_path)
        print(net)
        self.assertEqual(torch.Size([3]), net.normal_shape)
