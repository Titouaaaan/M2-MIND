import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


# load the data
mnist_train = datasets.MNIST(
    root="/tme1/data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
mnist_test = datasets.MNIST(
    root="/tme1/data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

BATCH_SIZE = 128
train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False)

# visualize some data
sample_img = train_loader.dataset[0][0] # these guys are tensors
input_size = sample_img.numel()
# print("Input size:", input_size) # should be 784, cuz it returns the total amount of elem in tensor

device = ('cuda' if torch.cuda.is_available() else 'cpu')

class VAE(nn.Module):
    def __init__(self, input_size=784, hidden_size=512, latent_dim=20):
        super(VAE, self).__init__()

        # encoder
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, latent_dim) 
        self.fc_logsigma = nn.Linear(hidden_size, latent_dim)

        # decoder
        self.fc2 = nn.Linear(latent_dim, hidden_size)
        self.fc3 = nn.Linear(hidden_size, input_size)

    def encode(self, x): # linéaire → ReLU → linéaire
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logsigma = self.fc_logsigma(h)  # logsigma = log(sigma^2)
        return mu, logsigma

    def reparameterize(self, mu, logsigma):
        sigma = torch.exp(0.5 * logsigma)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def decode(self, z):
        h = torch.relu(self.fc2(z))
        x_hat = torch.sigmoid(self.fc3(h))
        return x_hat

    def forward(self, x):
        mu, logsigma = self.encode(x)
        z = self.reparameterize(mu, logsigma)
        x_hat = self.decode(z)
        return x_hat, mu, logsigma
    
def vae_loss(x, x_hat, mu, logsigma):
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
    return recon_loss + kl

# training loop
epochs = 10
lr = 1e-3

model = VAE(input_size=784).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

reconstructed_imgs = []

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for x, _ in train_loader:
        x = x.to(device).view(x.size(0), -1)  # flatten

        optimizer.zero_grad()

        x_hat, mu, logsigma = model(x)
        loss = vae_loss(x, x_hat, mu, logsigma)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    with torch.no_grad():
        model.eval()
        sample_img = train_loader.dataset[0][0]
        reconstructed = model(sample_img.to(device).view(1, -1))[0]
        reconstructed_imgs.append(reconstructed.detach().cpu().view(28, 28)) # then just imshow this
      

    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss / len(train_loader.dataset):.4f}")

sample_img = train_loader.dataset[0][0]
plt.imshow(sample_img.squeeze())
plt.show()

rows = 3
cols = 3
fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
for ax, im in zip(axes.flatten(), reconstructed_imgs):
    ax.imshow(im, cmap="gray")
    ax.axis("off")

plt.tight_layout()
plt.show()

