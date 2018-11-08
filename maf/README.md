# Masked Autoregressive Flow

## The model

We will use the `AutoRegressiveNN` and `InverseAutoregressiveFlowStable` implemented in pyro.

```{.python .input  n=1}
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro.distributions as dist
from pyro.distributions import InverseAutoregressiveFlowStable
from pyro.nn import AutoRegressiveNN
from batchnorm import BatchNormTransform

import matplotlib.pyplot as plt
%matplotlib inline

device = torch.device("cpu")
```

```{.python .input  n=3}
class MAFsDensityEstimator(nn.Module):
    def __init__(self, D, K):
        super(MAFsDensityEstimator, self).__init__()
        modules = nn.ModuleList()
        ## inversing IAF to get MAF
        mafs = []
        for _ in range(K):
            iaf = InverseAutoregressiveFlowStable(AutoRegressiveNN(D, [2*D]))
            mafs.append(iaf.inv)
            modules.append(iaf.module)
            ## add batchnorm layer
            bnt = BatchNormTransform(D)
            mafs.append(bnt)
            modules.append(bnt.module)

        self.modules = modules
        μ, σ = torch.zeros(D, device=device), torch.eye(D, device=device)
        self.d = dist.TransformedDistribution(
            dist.MultivariateNormal(μ, σ),
            mafs
        )
    def forward(self, x):
        return self.d.log_prob(x)

    def sample(self, n):
        return self.d.sample((n,))
```

Let us define a target distribution to be learned.

```{.python .input  n=4}
## draw samples from target distribution
def drawP(n):
    x2_dist = dist.Normal(0, 4)
    x2 = x2_dist.sample((n,))
    x1_dist = dist.Normal(.25 * x2.pow(2), torch.ones_like(x2))
    x1 = x1_dist.sample()
    x = torch.stack([x1, x2], dim=1)
    return x
```

Drawing the training data and defining the model.

```{.python .input  n=5}
x = drawP(512).to(device)

K, D = 5, 2
m = MAFsDensityEstimator(D, K).to(device)
optimizer = torch.optim.Adam(m.parameters(), lr=0.001)
lossF = lambda x: -torch.mean(m(x))
```

Training.

```{.python .input  n=9}
epochs, iterations = 2, 5000
for epoch in range(epochs):
    epochLoss = 0.0
    for i in range(iterations):
        loss = lossF(x)
        epochLoss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: {}  Loss: {}".format(epoch, epochLoss/iterations))
```

## Visualization

### The learned distribution

```{.python .input  n=11}
z = m.sample(512)
plt.plot(z.cpu().numpy()[:, 0], z.cpu().numpy()[:, 1], ".r")
#plt.savefig("../images/iaf-1.png")
```

![](../images/iaf-1.png)

### The target distribution

```{.python .input}
plt.plot(x.cpu().numpy()[:, 0], x.cpu().numpy()[:, 1], ".r")
#plt.savefig("images/iaf-2.png")
```

![](../images/iaf-2.png)
