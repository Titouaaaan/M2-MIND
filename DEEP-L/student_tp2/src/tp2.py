from pathlib import Path
import os
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
from utils import *
# Téléchargement des données

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # only if we have a gpu
# need to add .to(device) to all the tensors to the gpu. This needs to be on all tensors or it crashes :(

from datamaestro import prepare_dataset
ds = prepare_dataset("com.lecun.mnist");
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()

print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)

# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Pour visualiser
# Les images doivent etre en format Channel (3) x Hauteur x Largeur
images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1,3,1,1).double()/255.
# Permet de fabriquer une grille d'images
images = make_grid(images)
# Affichage avec tensorboard
writer.add_image(f'samples', images, 0)


savepath = Path("model.pch")

# NOTE: ALL THE CLASSES AND THE FUNCTIONS TO ANSWER THE QUESTIONS 
# ARE IN THE utils.py FILE, cleaner that way

# ====================================================
# Question 1:

# test on random data for SGD and Adam, gets about the same results
# test_optimizer('SGD')
# test_optimizer('Adam')

# now we can test on the actual mnist data!
# preprocess the data:
x_train = torch.tensor(train_images, dtype=torch.double)
print(x_train.shape)
# this looks like this torch.tensor([60000, 28, 28])
# we want to flatten the last two dimensions into one
x_train = x_train.reshape(-1, 28*28) / 255.0
print(x_train.shape)
# cool beans now we have torch.Size([60000, 784])
# do the same for x_test
x_test = torch.tensor(test_images, dtype=torch.double).reshape(-1, 28*28) / 255.0

y_train, y_test = torch.tensor(train_labels, dtype=torch.long), torch.tensor(test_labels, dtype=torch.long)
print(y_train.shape) # should look like this torch.tensor([60000])
# but we also need to one hot encode this, so it fits the output size of 10 (both train and test)
y_train = torch.zeros(y_train.size(0), 10, dtype=torch.double)
y_test = torch.zeros(y_test.size(0), 10, dtype=torch.double)
print(y_train.shape)

in_features = 28*28
out_features = 10 #bc we use 

w = torch.nn.Parameter(torch.randn(in_features,out_features, dtype=torch.double))
b = torch.nn.Parameter(torch.randn(out_features, dtype=torch.double))

# UNCOMMENT THIS TO RUN IT
# optim_GD(x_train, y_train, w, b, eps=0.05, opt='Adam', niter=400) 

# curve looks nice! over 400 iterations we can see that it converges very well. 
# name of the run if visualization is needed: runs\Oct07_09-42-38_DESKTOP-GCQA3UV
# very smooth loss curve, converges to 0
# Now lets test the accuracy of our model
# theres probably cleaner ways to do this but it works fine for now

""" loaded_model = torch.load('simpleMNIST.pth')
w = loaded_model['w']
b = loaded_model['b'] """

# we will evaluate this later, like question 3

# ================================================================
# Question 2:
# Test same as previously but with our more complex NN
# also here we wanna test on the mnist numbers, no classification
# hence we use CE instead of MSE, so the y labels should not be one hot encoded

""" y_train_classif = torch.tensor(train_labels, dtype=torch.long)
x_train_classif = x_train.float()
model = ComplexNN(in_features=x_train.shape[1], out_features=10)
loss_module = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.05)
optim.zero_grad()
writer = SummaryWriter()
for n in range(400):
    ypred = model(x_train_classif)
    loss = loss_module(ypred, y_train_classif)
    writer.add_scalar('Loss/train', loss.item(), n)
    print(f"Itérations {n}: loss {loss.item()}")
    loss.backward()

    optim.step()
    optim.zero_grad()
writer.close() """

# concerning the Sequential, i directly made the class 
# (didn't understand we had to do it manually at first)
# not going to implement sequential, but in a nutshell it would be
# model = nn.Sequential(
#                       nn.Linear(10, 100),
#                       nn.TanH(),
#                       nn.Linear(100,1)
#                           )
# ====================================================

# Question 3
# So here we can reuse the work i did above, but just organise 
# it better and put it into a class    
# we created a dataloader, that gives batches of data in a shuffled way (if we want)
batch_size=32
data = MNISTDataLoader(batch_size=batch_size, shuffle=True, train=True)
""" for x,y in data:
    print(x.shape, y.shape) """
# this returns batches of this shape: torch.Size([32, 784]) torch.Size([32])
# ====================================================================

# Question 4
# see class AutoEncoder(torch.nn.Module) in utils.py
# ====================================================================

# Question 5
# test de l'autoencoder avec une couche lineaire
autoencoder = AutoEncoder(in_features=28*28, out_features=128)
# here we are going to work with batches of size 32
# so in input of the autoencoder will be (28*28, 128), we are reducing from 784 to 128
loss_module = torch.nn.MSELoss()
optim = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
optim.zero_grad()
writer = SummaryWriter()
for n in range(100):
    epoch_loss = 0
    for x, _ in data:
        ypred = autoencoder(x)
        loss = loss_module(ypred, x)
        epoch_loss += loss.item()
        loss.backward()

        optim.step()
        optim.zero_grad()
        #print(epoch_loss)
    loss = epoch_loss/len(data)
    writer.add_scalar('Average Loss/train', loss)
    print(f"Itérations {n}: loss {loss}")

writer.close() 
