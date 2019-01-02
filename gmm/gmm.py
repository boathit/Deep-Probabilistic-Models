import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributions as dist
from torch.distributions import Uniform, Normal
import matplotlib.pyplot as plt
%matplotlib inline

class GMMClustering(nn.Module):
    def __init__(self, D, K, hidden_size, a=0, b=1, dropout=0.1):
        """
        a, b are the prior knowledge of μ range such that μ ∈ [a, b].
        """
        super(GMMClustering, self).__init__()
        self.D = D
        self.K = K
        ## todo: maybe use the prior knowledge to initialize M
        ## such that μs are uniformly distributed in the data space
        #self.M = nn.Parameter(torch.randn(K, D)) # μs
        self.M = nn.Parameter(Uniform(a, b).sample((K, D))) #μs
        self.logS = nn.Parameter(torch.randn(K, D)) # logσs
        ## π is the parameter of the Categorical distribution
        self.x2logπ = nn.Sequential(MLP(D, hidden_size, K, dropout),
                                    nn.LogSoftmax(dim=1))
        self.d_uniform = Uniform(0, 1)

    def forward(self, x, τ=1.0):
        """
        Input:
          x (batch, D)
        Output:
          z (batch, K)
          l (scalar): negative log-likehood
        """
        z = self.encoder(x, τ)
        μ, σ = self.decoder(z)
        l = NLLGauss(μ, σ, x)
        return z, l

    def encoder(self, x, τ):
        """
        Input:
          x (batch, D)
        Output:
          z (batch, K): Gumbel-softmax samples.
        """
        logπ = self.x2logπ(x)
        u = self.d_uniform.sample(logπ.size())
        g = -torch.log(-torch.log(u))
        z = F.softmax((logπ + g)/τ, dim=1)
        return z

    def decoder(self, z):
        """
        Input:
          z (batch, K)
        Output:
          μ (batch, D)
          σ (batch, D)
        """
        μ = torch.mm(z, self.M)
        σ = torch.exp(torch.mm(z, self.logS))
        return μ, σ

def NLLGauss(μ, σ, x):
    """
    μ (batch, D)
    σ (batch, D)
    x (batch, D)
    """
    return -torch.mean(Normal(μ, σ).log_prob(x))


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 dropout, use_selu=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.nonlinear_f = F.selu if use_selu else F.leaky_relu
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h1 = self.dropout(self.nonlinear_f(self.fc1(x)))
        return self.fc2(h1)


μ1, Σ1 = torch.tensor([0.0, 0.0]), torch.diag(torch.tensor([1.0, 2.0]))
d1 = dist.MultivariateNormal(μ1, Σ1)
μ2, Σ2 = torch.tensor([5.0, 5.0]), torch.diag(torch.tensor([1.0, 1.0]))
d2 = dist.MultivariateNormal(μ2, Σ2)

X1 = d1.sample((500,))
X2 = d2.sample((500,))
x = torch.cat([X1, X2])

D, K = 2, 2
gmmc = GMMClustering(D, K, 64)
optimizer = torch.optim.Adam(gmmc.parameters())

for i in range(8000):
    z, l = gmmc(x)
    if i % 1000 == 0:
        print("loss: {}".format(l.item()))
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

z, l = gmmc(x)

val, idx = torch.max(z, dim=1)

gmmc.M

###############################################################################


class MDN(nn.Module):
    """
    D is the dimension of the data.
    K is the number of components of GMM.
    """
    def __init__(self, D, K, hidden_size):
        super(MDN, self).__init__()
        self.D = D
        self.K = K
        self.fc = nn.Linear(D, hidden_size)
        self.f_log_π = nn.Sequential(
            nn.Linear(hidden_size, K),
            nn.LogSoftmax(dim=1)
        )
        self.f_μ = nn.Linear(hidden_size, D*K)
        self.f_log_σ = nn.Linear(hidden_size, D*K)

    def forward(self, x):
        """
        Input:
          x (batch, D)
        Output:
          log_π (batch, K)
          μ (batch, K, D)
          σ (batch, K, D)
        """
        h = F.relu(self.fc(x))
        log_π = self.f_log_π(h)
        μ = self.f_μ(h)
        σ = torch.exp(torch.clamp(self.f_log_σ(h), min=-10.0, max=5.0))
        return log_π, μ.view(-1, self.K, self.D), σ.view(-1, self.K, self.D)


def NLLGMM(log_π, μ, σ, x):
    """
    Negative log-likelihood of GMM.
    K is the number of components of GMM and D is the dimension of data.
    Input:
      log_π (batch, K)
      μ (batch, K, D)
      σ (batch, K, D)
      x (batch, D)
    """
    D = x.size(1)
    C = 0.5 * D * (np.log(2) + np.log(np.pi))
    ## (batch, K, D)
    z = (x.unsqueeze(1) - μ) / σ
    ## (batch, K)
    exponents = log_π - 0.5*z.pow(2).sum(dim=2) - σ.log().sum(dim=2) - C
    return -torch.mean(torch.logsumexp(exponents, dim=1))

###############################################################################
