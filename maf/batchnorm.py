import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Transform


class BatchNorm(nn.Module):
    def __init__(self, input_dim, momentum=0.1, ϵ=1e-8):
        super(BatchNorm, self).__init__()

        self.γ = nn.Parameter(torch.zeros(input_dim))
        self.β = nn.Parameter(torch.zeros(input_dim))
        self.momentum = momentum
        self.ϵ = ϵ

        self.register_buffer('μ', torch.zeros(input_dim))
        self.register_buffer('σ2', torch.zeros(input_dim))

    def forward(self, u):
        ## forward computation is only called when drawing samples in test stage
        ## thus we use the statistics estimated from the entire training dataset
        μ, σ2 = self.μ, self.σ2
        x = (u - self.β) * torch.exp(-self.γ) * torch.sqrt(σ2 + self.ϵ) + μ
        return x

    def inverse(self, x):
        ## In density estimation, we should normalize x in reverse computation
        n = x.size(0)
        if self.training:
            μ = x.mean(0)
            ## unbiased estimation
            σ2 = (x - μ).pow(2).mean(0) * n / (n-1)
            self.μ.mul_(1 - self.momentum).add_(μ.data * self.momentum)
            self.σ2.mul_(1 - self.momentum).add_(σ2.data * self.momentum)
        else:
            μ, σ2 = self.μ, self.σ2

        u = torch.exp(self.γ) * (x - μ) / torch.sqrt(σ2 + self.ϵ) + self.β
        return u

    def log_abs_det_jacobian(self, x):
        ## log(|det(∂x/∂u)|)
        n = x.size(0)
        σ2 = (x - x.mean(0)).pow(2).mean(0) * n / (n-1)
        return (-self.γ + 0.5 * torch.log(σ2 + self.ϵ)).repeat(n, 1)

class BatchNormTransform(Transform):
    def __init__(self, input_dim):
        super(BatchNormTransform, self).__init__()
        self.module = nn.Module()
        self.module.bn = BatchNorm(input_dim)

    def _call(self, u):
        return self.module.bn.forward(u)

    def _inverse(self, x):
        return self.module.bn.inverse(x)

    def log_abs_det_jacobian(self, u, x):
        return self.module.bn.log_abs_det_jacobian(x)
