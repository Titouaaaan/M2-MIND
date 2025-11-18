import logging
logging.basicConfig(level=logging.INFO)

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import click


from sklearn.datasets import fetch_openml

from datamaestro import prepare_dataset
ds = prepare_dataset("com.lecun.mnist")

train_img, train_labels = ds.train.images.data(), ds.train.labels.data()
test_img, test_labels = ds.test.images.data(), ds.test.labels.data()

# Changer le DATA_PATH
# dotn think we will be using this locally
DATA_PATH = "/tmp/mnist"

# Ratio du jeu de train à utiliser
TRAIN_RATIO = 0.05
TEST_RATIO = 0.2
def store_grad(var):
    """Stores the gradient during backward

    For a tensor x, call `store_grad(x)`
    before `loss.backward`. The gradient will be available
    as `x.grad`

    """
    def hook(grad):
        var.grad = grad
    var.register_hook(hook)
    return var

# keep only part of the dataset
train_img = train_img[: int(len(train_img) * TRAIN_RATIO)]
train_labels = train_labels[: int(len(train_labels) * TRAIN_RATIO)]
test_img = test_img[: int(len(test_img) * TEST_RATIO)]
test_labels = test_labels[: int(len(test_labels) * TEST_RATIO)]

print(f"Train dataset size: {len(train_img)}")
assert len(train_img) == len(train_labels)
print(f"Test dataset size: {len(test_img)}")
assert len(test_img) == len(test_labels)
# dataset properly split!

#make our small network
# Vous utiliserez un réseau composé de 3 couches linéaires avec 100 sorties, suivis d'une
# couche linéaire pour la classication (10 classes les chiffres de 0 à 9). Vous utiliserez un
# coût cross-entropique, des batchs de taille 300, et 1000 itérations (epochs).

dim_output = 10
batch_size = 300
num_epochs = 100

class MNISTDataset(Dataset):
        def __init__(self, imgs, labels):
            self.imgs = imgs
            self.labels = labels
        def __len__(self):
            return len(self.imgs)
        def __getitem__(self, idx):
            return self.imgs[idx], self.labels[idx]

train_i = torch.tensor(train_img, dtype=torch.float32).unsqueeze(1)
test_i  = torch.tensor(test_img,  dtype=torch.float32).unsqueeze(1)
train_l = torch.tensor(train_labels, dtype=torch.long)
test_l  = torch.tensor(test_labels,  dtype=torch.long)

train_ds = MNISTDataset(train_i, train_l)
test_ds = MNISTDataset(test_i, test_l)

train_loader = DataLoader(train_ds, batch_size, True)
test_loader = DataLoader(test_ds, batch_size, True)

input_dim = train_img[0].shape[0] * train_img[0].shape[1]  # should be 28x28
print(f"Input dim: {input_dim}")

class Model(nn.Module):
    def __init__(self, input_dim, dim_output):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, dim_output)
    
    def forward(self, x):
        # inside here we also need to add a retain grad 
        x = x.view(-1, input_dim)  # Flatten the input
        self.x1 = x

        x = F.relu(self.fc1(x))
        self.x2 = x
        x = F.relu(self.fc2(x))

        self.x3 = x
        x = F.relu(self.fc3(x))

        self.x4 = x
        x = self.fc4(x)

        return x

def run(num_epochs, model, train_loader, test_loader):
    writer = SummaryWriter('runs/')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for x,y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x,y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = loss_function(output, y)
            loss.backward()
            
            """ if epoch % 20 == 0:
                writer.add_histogram("Grads/Input_fc1", model.x1.grad, epoch)
                writer.add_histogram("Grads/Input_fc2", model.x2.grad, epoch)
                writer.add_histogram("Grads/Input_fc3", model.x3.grad, epoch)
                writer.add_histogram("Grads/Input_fc4", model.x4.grad, epoch) """

            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 20 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                test_loss = 0
                for x,y in test_loader:
                    x,y = x.to(device), y.to(device)
                    output = model(x)
                    loss = loss_function(output, y)
                    test_loss += loss.item()
                    prediction = torch.argmax(output, dim=1)
                    correct += (prediction == y).sum().item() # over a batch
                    total += y.size(0)
                accuracy = correct/total
                print(f"At epoch {epoch+1}, test accuracy is {accuracy*100}%")
                writer.add_scalar('Accuracy/test', accuracy, epoch) # save epoch test accuracy
                writer.add_scalar('Loss/test', test_loss / len(test_loader), epoch) #save epoch test loss
            
                # lets also save the weights of the model (every 20 epochs)
                # save it as a histogram
                for name, param in model.named_parameters(): # loop over the dict of weights
                    writer.add_histogram(f"Weights/{name}", param, epoch)
        
        avg_loss = epoch_loss / len(train_loader) #save epoch train loss
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/train', avg_loss, epoch)
    
    writer.close()

    torch.save(model.state_dict(), 'models/basic_model.pth')

model = Model(input_dim=input_dim, dim_output=dim_output)
run(num_epochs, model, train_loader, test_loader)