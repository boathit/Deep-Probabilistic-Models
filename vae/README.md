# Variational AutoEncoder

```{.python .input  n=1}
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
%matplotlib inline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

```{.python .input  n=2}
def getDataLoaders(trainData, testData, batch_size):
    trainLoader = torch.utils.data.DataLoader(dataset=trainData,
                                              batch_size=batch_size,
                                              shuffle=True)
    testLoader = torch.utils.data.DataLoader(dataset=testData,
                                             batch_size=batch_size,
                                             shuffle=False)
    return trainLoader, testLoader

def getmnistDataLoaders(path, batch_size):
    trainData = datasets.MNIST(root=path,
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
    testData = datasets.MNIST(root=path,
                              train=False,
                              transform=transforms.ToTensor())

    return getDataLoaders(trainData, testData, batch_size)


batch_size = 256

trainLoader, testLoader = getmnistDataLoaders("../data", batch_size)
```

```{.python .input  n=14}
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(28*28, 400)
        self.fμ = nn.Linear(400, 20)
        self.flogσ = nn.Linear(400, 20)

    def forward(self, x):
        h = F.relu(self.fc(x))
        return self.fμ(h), self.flogσ(h)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.f = nn.Sequential(nn.Linear(20, 400),
                               nn.ReLU(),
                               nn.Linear(400, 784),
                               nn.LogSigmoid())
    def forward(self, z):
        return self.f(z)

def reparameterize(μ, logσ):
    σ = torch.exp(logσ)
    ϵ = torch.randn_like(σ)
    return μ + σ * ϵ

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        μ, logσ = self.encoder(x)
        z = reparameterize(μ, logσ)
        logx̂ = self.decoder(z)
        ## loss
        term1 = F.binary_cross_entropy_with_logits(logx̂, x, size_average=False)
        term2 = -0.5 * torch.sum(1 + 2*logσ - torch.pow(μ, 2) - torch.exp(2*logσ))
        return term1 + term2
```

```{.python .input  n=15}
m = VAE().to(device)
optimizer = torch.optim.Adam(m.parameters())
```

```{.python .input}
epochs = 7
for epoch in range(epochs):
    epochLoss = 0.0
    for x, _ in trainLoader:
        x = x.reshape(-1, 784).to(device)
        loss = m(x)
        epochLoss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: {}  Loss: {}".format(epoch, epochLoss/len(trainLoader.dataset)))
```

## Visualization

```{.python .input  n=17}
def drawsamples(decoder, n):
    with torch.no_grad():
        z = torch.randn(n, 20).to(device)
        x = torch.exp(decoder(z))
        return x.reshape(n, 28, 28)
```

```{.python .input  n=18}
x = drawsamples(m.decoder, 5)
img = torch.cat(torch.unbind(x, dim=0)).cpu().numpy()
```

```{.python .input}
plt.figure(figsize=(4, 20))
plt.imshow(img)
#plt.savefig("../images/vae-1.png")
```

![](../images/vae-1.png)

```{.python .input}

```

```{.python .input}

```
