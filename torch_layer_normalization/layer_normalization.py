import torch
import torch.nn as nn


__all__ = ['LayerNormalization']


class LayerNormalization(nn.Module):

    def __init__(self,
                 normal_shape,
                 scale=True,
                 center=True,
                 epsilon=1e-10):
        """Layer normalization layer

        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

        :param normal_shape: The shape of the input tensor or the last dimension of the input tensor.
        :param scale: Add a scale parameter if it is True.
        :param center: Add an offset parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        """
        super(LayerNormalization, self).__init__()
        if isinstance(normal_shape, int):
            normal_shape = (normal_shape,)
        else:
            normal_shape = (normal_shape[-1],)
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

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.mean((x - mean) ** 2, dim=-1, keepdim=True)
        std = torch.sqrt(var + self.epsilon)
        y = (x - mean) / std
        if self.scale:
            y *= self.gamma
        if self.center:
            y += self.beta
        return y

    def extra_repr(self):
        return '{normal_shape}, center={center}, scale={scale}, eps={epsilon}'.format(**self.__dict__)
