import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro.distributions as dist
from pyro.distributions import InverseAutoregressiveFlowStable
from pyro.nn import AutoRegressiveNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MAFsDensityEstimator(nn.Module):
    def __init__(self, D, iafs):
        super(MAFsDensityEstimator, self).__init__()
        self.iafs = iafs
        self.iafs_modules = nn.ModuleList([iaf.module for iaf in self.iafs])
        ## inversing IAF to get MAF
        mafs = [iaf.inv for iaf in self.iafs]
        μ, σ = torch.zeros(D, device=device), torch.eye(D, device=device)
        self.d = dist.TransformedDistribution(
            dist.MultivariateNormal(μ, σ),
            mafs
        )
    def forward(self, x):
        return self.d.log_prob(x)

    def sample(self, n):
        return self.d.sample((n,))

## draw samples from target distribution
def drawP(n):
    x2_dist = dist.Normal(0, 4)
    x2 = x2_dist.sample((n,))
    x1_dist = dist.Normal(.25 * x2.pow(2), torch.ones_like(x2))
    x1 = x1_dist.sample()
    x = torch.stack([x1, x2], dim=1)
    return x

x = drawP(512).to(device)

K, D = 3, 2
mafs = [InverseAutoregressiveFlowStable(AutoRegressiveNN(2, [4])) for _ in range(K)]
m = MAFsDensityEstimator(D, mafs).to(device)
optimizer = torch.optim.Adam(m.parameters(), lr=0.001)
lossF = lambda x: -torch.mean(m(x))

epochs, iterations = 5, 5000
for epoch in range(epochs):
    epochLoss = 0.0
    for i in range(iterations):
        loss = lossF(x)
        epochLoss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: {}  Loss: {}".format(epoch, epochLoss/iterations))


import matplotlib.pyplot as plt
%matplotlib inline

## learned distribution
z = m.sample(512)
plt.plot(z.cpu().numpy()[:, 0], z.cpu().numpy()[:, 1], ".r")
plt.savefig("iaf-1.png")

## target distribution
plt.plot(x.cpu().numpy()[:, 0], x.cpu().numpy()[:, 1], ".r")
