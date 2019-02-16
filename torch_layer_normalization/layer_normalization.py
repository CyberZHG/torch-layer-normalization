import torch
import torch.nn as nn


__all__ = ['LayerNormalization']


class LayerNormalization(nn.Module):

    def __init__(self,
                 normal_shape,
                 scale=True,
                 center=True,
                 epsilon=1e-10):
        super(LayerNormalization, self).__init__()
        if isinstance(normal_shape, int):
            normal_shape = (normal_shape,)
        self.normal_shape = torch.Size(normal_shape)
        self.center, self.scale, self.epsilon = center, scale, epsilon
        gamma, beta = None, None
        if scale:
            gamma = nn.Parameter(torch.Tensor(*normal_shape))
        if center:
            beta = nn.Parameter(torch.Tensor(*normal_shape))
        self.register_parameter('gamma', gamma)
        self.register_parameter('beta', beta)
        self.reset_parameters()

    def reset_parameters(self):
        if self.scale:
            self.gamma.data.fill_(1)
        if self.center:
            self.beta.data.zero_()

    def forward(self, inputs):
        mean = torch.mean(inputs, dim=-1, keepdim=True)
        var = torch.mean((inputs - mean) ** 2, dim=-1, keepdim=True)
        std = torch.sqrt(var + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs

    def extra_repr(self):
        return '{normal_shape}, center={center}, scale={scale}, eps={epsilon}'.format(**self.__dict__)
