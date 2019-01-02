import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np

## input layer mask
def create_mask_i(num_i, num_o, D):
    assert num_i == D, "number of input should equal to D"
    i = torch.arange(num_i) # [0, D-1]
    o = torch.arange(num_o) % (D - 1) # [0, D-2]
    return o.unsqueeze(-1) >= i.unsqueeze(0)

## hidden layer mask
def create_mask_h(num_i, num_o, D):
    i = torch.arange(num_i) % (D - 1) # [0, D-2]
    o = torch.arange(num_o) % (D - 1) # [0, D-2]
    return o.unsqueeze(-1) >= i.unsqueeze(0)

## output layer mask
def create_mask_o(num_i, num_o, D):
    i = torch.arange(num_i) % (D - 1) # [0, D-2]
    o = torch.arange(num_o) % D #[0, D-1]
    return o.unsqueeze(-1) > i.unsqueeze(0)

def create_masks(D, hs, m):
    """
    masks = create_masks(5, [10, 10], 1)
    masks[2] @ masks[1] @ masks[0]
    """
    masks = [create_mask_i(D, hs[0], D)]
    for i in range(1, len(hs)):
        masks.append(create_mask_h(hs[i-1], hs[i], D))
    masks.append(create_mask_o(hs[-1], m*D, D))
    return [mask.float() for mask in masks]


class MaskedLinear(nn.Linear):
    def __init__(self, num_i, num_o, mask, bias=True):
        super(MaskedLinear, self).__init__(num_i, num_o, bias)
        self.register_buffer('mask', mask)

    # def reset_parameters(self):
    #     nn.init.orthogonal_(self.weight)
    #     if self.bias is not None:
    #         self.bias.data.fill_(0.0)

    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)

class AutoRegressiveNN(nn.Module):
    """
    D = 5
    arn = AutoRegressiveNN(D, [10, 10], 1)
    x = torch.rand(3, D)
    o = arn(x)
    """
    def __init__(self, D, hs, m):
        super(AutoRegressiveNN, self).__init__()
        self.m = m
        masks = create_masks(D, hs, m)
        layers = []
        ## input layer
        layers.append(MaskedLinear(D, hs[0], masks[0]))
        layers.append(nn.ReLU())
        ## hidden layers
        for i in range(1, len(hs)):
            layers.append(MaskedLinear(hs[i-1], hs[i], masks[i]))
            layers.append(nn.ReLU())
        ## output layer
        layers.append(MaskedLinear(hs[-1], m*D, masks[-1]))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        o = self.layers(x) if self.m == 1 else self.layers(x).chunk(self.m, dim=-1)
        return o


################################## transforms
clamp_ = lambda x, min, max: x + (x.clamp(min, max) - x).detach()

class MAF(nn.Module):
    """
    Masked Autoregressive Flow. MAF is used in density estimation, its forward
    computation (sampling) is expensive (requires D steps), while its inverse
    computation is cheap.

    maf = MAF(5, [15, 15])
    u = torch.rand(10, 5)
    x = maf.forward(u)
    (maf.inverse(x) - u).abs().max()
    """
    def __init__(self, D, hs, clamp_min=-5.0, clamp_max=3.0):
        super(MAF, self).__init__()
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.arn = AutoRegressiveNN(D, hs, 2)
        self.cache = {}

    def forward(self, u):
        ## u -> x
        x = torch.zeros_like(u)
        for i in range(u.shape[-1]):
            μ, α = self.arn(x)
            # clamp α
            α[:, i] = clamp_(α[:, i], self.clamp_min, self.clamp_max)
            x[:, i] = u[:, i] * torch.exp(α[:, i]) + μ[:, i]
        return x

    def inverse(self, x):
        ## x -> u
        μ, α = self.arn(x)
        # clamp α
        α = clamp_(α, self.clamp_min, self.clamp_max)
        u = (x - μ) * torch.exp(-α)
        self.cache[x] = α
        return u

    def logdetJ(self, u, x):
        if x in self.cache:
            α = self.cache.pop(x)
        else:
            _, α = self.arn(x)
            # clamp α
            α = clamp_(α, self.clamp_min, self.clamp_max)
        return torch.sum(α, dim=-1, keepdim=True)

class Reverse(nn.Module):
    """
    reverse = Reverse(5)
    u = torch.rand(10, 5)
    x = reverse.forward(u)
    reverse.inverse(x)
    """
    def __init__(self, D):
        super(Reverse, self).__init__()
        self.perm = np.arange(D-1, -1, -1)
        self.inv_perm = self.perm

    def forward(self, u):
        return u[:, self.inv_perm]

    def inverse(self, x):
        return x[:, self.perm]

    def logdetJ(self, u, x):
        return torch.zeros_like(x[:, 0:1])

class Permute(nn.Module):
    """
    permute = Permute(5)
    u = torch.rand(10, 5)
    x = permute.forward(u)
    permute.inverse(x)
    """
    def __init__(self, D):
        super(Permute, self).__init__()
        self.perm = np.random.permutation(D)
        self.inv_perm = np.argsort(self.perm)

    def forward(self, u):
        return u[:, self.inv_perm]

    def inverse(self, x):
        return x[:, self.perm]

    def logdetJ(self, u, x):
        return torch.zeros_like(x[:, 0:1])

class BatchNorm(nn.Module):
    """
    D = 5
    bn = BatchNorm(D)
    bn.eval()
    u = torch.randn(10, D)
    x = bn.forward(u)
    torch.abs(bn.inverse(x) - u).max()
    bn.train()
    """
    def __init__(self, D, ρ=0.9, ϵ=1e-5):
        super(BatchNorm, self).__init__()

        self.w = nn.Parameter(torch.zeros(D)) # logγ
        self.b = nn.Parameter(torch.zeros(D)) # β
        self.ρ = ρ
        self.ϵ = ϵ

        self.register_buffer('m', torch.zeros(D)) # running mean
        self.register_buffer('v', torch.ones(D)) # running variance

    def forward(self, u):
        ## The forward is only used when drawing samples from the transformed distribution,
        ## thus we use the statistics estimated from the entire training dataset.
        return (u - self.b) * torch.exp(-self.w) * torch.sqrt(self.v + self.ϵ) + self.m

    def inverse(self, x):
        ## In the training stage we use the statistics estimated from current mini-batch;
        ## in the testing stage we use the statistics estimated from the entire training dataset.
        if self.training:
            m, v = x.mean(0), x.std(0)
            self.m.mul_(self.ρ).add_(m * (1 - self.ρ))
            self.v.mul_(self.ρ).add_(v * (1 - self.ρ))
            #self.m.sub_(self.ρ * (self.m - m))
            #self.v.sub_(self.ρ * (self.v - v))
        else:
            m, v = self.m, self.v

        return torch.exp(self.w) * (x - m) / torch.sqrt(v + self.ϵ) + self.b

    def logdetJ(self, u, x):
        ## The Jacobian's determinant is computed using the current mini-batch
        v = x.std(0)
        return (-self.w + 0.5 * torch.log(v + self.ϵ)).sum(dim=-1, keepdim=True)

class IAF(nn.Module):
    """
    Inverse Autoregressive Flow. IAF is mainly used in variational inference,
    its forward computation (sampling) is cheap, but the inverse computation
    (log-likelihood) is expensive (requires D steps).

    iaf = IAF(5, [15, 15])
    u = torch.rand(10, 5)
    x = iaf.forward(u)
    (iaf.inverse(x) - u).abs().max()
    """
    def __init__(self, D, hs):
        super(IAF, self).__init__()
        self.maf = MAF(D, hs)

    def forward(self, u):
        ## u -> x
        return self.maf.inverse(u)

    def inverse(self, x):
        ## x -> u
        return self.maf.forward(x)

    def logdetJ(self, u, x):
        return -self.maf.logdetJ(x, u)

################################## transforms

class NormFlowDE(nn.Module):
    """
    Normalizing Flow for Density Estimation.

    D, hs = 2, [10, 10]
    transforms = [MAF(D, hs), Reverse(D)]
    μ, Σ = torch.ones(D), torch.eye(D)

    m = NormFlowDE(μ, Σ, transforms)
    x = torch.rand(10, D)
    m.sample(3)
    m.log_prob(x)
    """
    def __init__(self, μ, Σ, transforms):
        super(NormFlowDE, self).__init__()
        self.transforms = nn.ModuleList(transforms)
        self.base_dist = dist.MultivariateNormal(μ, Σ)

    def sample(self, n=1):
        u = self.base_dist.sample((n,))

        self.transforms.eval() # eval mode
        with torch.no_grad():
            for transform in self.transforms:
                u = transform.forward(u)
        self.transforms.train() # train mode

        return u

    def log_prob(self, y):
        x = y
        J = 0
        for transform in reversed(self.transforms):
            u = transform.inverse(x)
            J = J + transform.logdetJ(u, x)
            x = u
        log_prob = self.base_dist.log_prob(x).unsqueeze(-1)
        return log_prob - J

    def forward(self, x):
        return self.log_prob(x)
