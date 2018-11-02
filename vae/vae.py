import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

torch.manual_seed(1)

batch_size = 128
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True,
    **kwargs
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False,
                   transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True,
    **kwargs
)

train_loader.dataset[0][0].size()

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # encoder
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        # decoder
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3) # return logits of x̂

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def lossF(xhat_logits, xs, mu, logvar):
    BCE = F.binary_cross_entropy_with_logits(xhat_logits, xs.view(-1, 784),
                                             size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(epoch):
    model.train()
    train_loss = 0
    for i, (xs, _) in enumerate(train_loader):
        xs = xs.to(device)
        xhat_logits, mu, logvar = model(xs)
        loss = lossF(xhat_logits, xs, mu, logvar)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, i * len(xs), len(train_loader.dataset),
                100*i / len(train_loader),
                loss.item() / len(xs)
            ))
    print("Epoch: {} Average loss: {:.4f}".format(epoch, train_loss/len(train_loader.dataset)))

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (xs, _) in enumerate(test_loader):
            xs = xs.to(device)
            xhat_logits, mu, logvar = model(xs)
            test_loss += lossF(xhat_logits, xs, mu, logvar).item()
    test_loss /= len(test_loader.dataset)
    print("===> Test set loss: {:.4f}".format(test_loss))

#xs = torch.randn(2, 28, 28)
#xhat_logits, mu, logvar = model(xs)
#lossF(xhat_logits, xs, mu, logvar)
#train(0)
#test(0)

for epoch in range(1, 6):
    train(epoch)
    test(epoch)
    with torch.no_grad():
        sample = torch.randn(64, 20).to(device)
        sample = F.sigmoid(model.decode(sample)).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'results/sample_{}.png'.format(epoch))
