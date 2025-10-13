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
print(f'using {device} for device')

from datamaestro import prepare_dataset
ds = prepare_dataset("com.lecun.mnist");
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()

print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)

curr_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
# writer = SummaryWriter("runs/run_"+curr_date)

# Pour visualiser
# Les images doivent etre en format Channel (3) x Hauteur x Largeur
images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1,3,1,1).double()/255.
# Permet de fabriquer une grille d'images
images = make_grid(images)


savepath = Path("model.pch")

# NOTE: ALL THE CLASSES AND THE FUNCTIONS TO ANSWER THE QUESTIONS 
# ARE IN THE utils.py FILE, cleaner that way

# ====================================================
# Question 1:

# test on random data for SGD and Adam, gets about the same results
test_optimizer('SGD')
test_optimizer('Adam')

# now we can test on the actual mnist data!
# preprocess the data:
x_train = torch.tensor(train_images, dtype=torch.double).to(device)
print(x_train.shape)
# this looks like this torch.tensor([60000, 28, 28])
# we want to flatten the last two dimensions into one
x_train = x_train.reshape(-1, 28*28) / 255.0
print(x_train.shape)
# cool beans now we have torch.Size([60000, 784])
# do the same for x_test
x_test = torch.tensor(test_images, dtype=torch.double).reshape(-1, 28*28) / 255.0
x_test = x_test.to(device)

y_train, y_test = torch.tensor(train_labels, dtype=torch.long).to(device), torch.tensor(test_labels, dtype=torch.long).to(device)
print(y_train.shape) # should look like this torch.tensor([60000])
# but we also need to one hot encode this, so it fits the output size of 10 (both train and test)
y_train = torch.zeros(y_train.size(0), 10, dtype=torch.double).to(device)
y_test = torch.zeros(y_test.size(0), 10, dtype=torch.double).to(device)
print(y_train.shape)

in_features = 28*28
out_features = 10 #bc we use 

w = torch.nn.Parameter(torch.randn(in_features,out_features, dtype=torch.double, device=device))
b = torch.nn.Parameter(torch.randn(out_features, dtype=torch.double, device=device))

# UNCOMMENT THIS TO RUN IT (if i commented it)
optim_GD(x_train, y_train, w, b, eps=0.05, opt='Adam', niter=400) 

# curve looks nice! over 400 iterations we can see that it converges very well. 
# name of the run if visualization is needed: runs\Oct07_09-42-38_DESKTOP-GCQA3UV
# very smooth loss curve, converges to 0
# Now lets test the accuracy of our model
# theres probably cleaner ways to do this but it works fine for now

loaded_model = torch.load('simpleMNIST.pth')
w = loaded_model['w']
b = loaded_model['b'] 

# we will evaluate this later, like question 3

# ================================================================
# Question 2:
# Test same as previously but with our more complex NN
# also here we wanna test on the mnist numbers, no classification
# hence we use CE instead of MSE, so the y labels should not be one hot encoded

y_train_classif = torch.tensor(train_labels, dtype=torch.long).to(device)
x_train_classif = x_train.float().to(device)
model = ComplexNN(in_features=x_train.shape[1], out_features=10).to(device)
loss_module = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.05)
optim.zero_grad()
writer = SummaryWriter("runs/run_question2_"+curr_date)
for n in range(400):
    ypred = model(x_train_classif)
    loss = loss_module(ypred, y_train_classif)
    writer.add_scalar('Loss/train', loss.item(), n)
    print(f"Itérations {n}: loss {loss.item()}")
    loss.backward()

    optim.step()
    optim.zero_grad()
writer.close() 

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

# LITTLE REMARK: for the gpu calculations, all the previous code and following code was adapted to run
# on the gpu
# earlier we determined the device (cuda if available)
# we can see that the optimizing of the models is much much faster on gpu than cpu
# this code was ran locally on a rtx 3070, for reference in case we do time comparisons after

# Question 5
# test de l'autoencoder avec une couche lineaire
autoencoder = AutoEncoder(in_features=28*28, out_features=128).to(device)
# here we are going to work with batches of size 32
# so in input of the autoencoder will be (28*28, 128), we are reducing from 784 to 128
loss_module = torch.nn.MSELoss()
optim = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
optim.zero_grad()

#setting up the checkpointing
log_dir = f"runs/autoencoder_log_{curr_date}"
ckpt_path = f"checkpoints/question_checkpoint_{curr_date}.pt"
writer = SummaryWriter(log_dir=log_dir)

# here 20 episodes is enough since each episode actually contains many optim steps (so no need for like 200/400 iterations)
# in our tests here we usually converge to 0.001 loss at around 7-8 iterations,
# without a big decrease after that (see logged values on tensorboard)
# we also add some checkpointing and log more info to the tensorboard

for n in range(20):
    epoch_loss = 0
    for x, _ in data:
        x = x.to(device) # pass to device since our dataloader doenst do it (would be cleaner if it did ngl)
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

    torch.save(autoencoder.state_dict(), ckpt_path) #checkpoint

    if n % 5 == 0:  # dont log every epoch!
        # take next batch for embedding visualization
        x_batch, y_batch = next(iter(data))
        x_batch = x_batch.to(device)
        # what we do here is grab the latent space and it's label and save it
        # in the tensorboard, int eh projector tab (top left), we can observe in a 3D space the
        # latent space of our numbers. The default algo to bring it down to 3 dimensions is PCA but you can also pick other ones
        # it works pretty well to visualise, but not perfect for some numbers (looks like some 3s and 8s get mixed up, 5s too)
        with torch.no_grad():
            encodings = autoencoder.encoder(x_batch)
        writer.add_embedding(encodings, metadata=y_batch, label_img=x_batch.view(-1, 1, 28, 28), global_step=n)

        img = x[0].view(1, 28, 28)  # first image in batch
        recon = ypred[0].view(1, 28, 28)
        writer.add_image('Original', img, global_step=n)
        writer.add_image('Reconstructed', recon, global_step=n)

writer.close() 

# ok so all of this is visible on the tensorboard
# Notes: 
#       1) each 'run' is in a seperate folder, ik its not the best but ill change it
#       2) this runs on gpu so very fast, need to check time differences with cpu
#       3) this was tested with the basic question 4 architecture, for sure would be a bit better 
#          with one extra layer, but above that i doubt we will see a big difference since this 
#          simple architecture already works well well. Just looking at the reconstructed images
#          let's us see that just 5 iterations already gives very good reconstructions (thanks batch learning!)
# ==================================================

# Question 6 - Highway network
# Implementation of the paper that introduces a NN with gates
class HighwayNetwork(torch.nn.Module):
    def __init__(self, input_size, num_layers=3, activation=torch.nn.ReLU()):
        super(HighwayNetwork, self).__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.nonlinear = torch.nn.ModuleList([torch.nn.Linear(input_size, input_size) for _ in range(num_layers)])
        self.the_gate = torch.nn.ModuleList([torch.nn.Linear(input_size, input_size) for _ in range(num_layers)])
        self.output_layer = torch.nn.Linear(input_size, 10)  # couche finale de classification

    def forward(self, x): # rappel: y = H(x) * T(x) + x * (1 - T(x))
        for i in range(self.num_layers):
            H = self.activation(self.nonlinear[i](x))
            T = torch.sigmoid(self.the_gate[i](x))
            x = H * T + x * (1 - T)
        return self.output_layer(x)

model = HighwayNetwork(input_size=784, num_layers=3, activation=nn.ReLU()).to(device)
ce_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
writer = SummaryWriter("runs/highway_"+curr_date)
n_epochs = 15

for epoch in range(n_epochs):
    somme_loss = 0.0
    for xb, yb in data:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        y_pred = model(xb)
        loss = ce_loss(y_pred, yb)
        loss.backward()
        optimizer.step()
        somme_loss += loss.item()

    avg_loss = somme_loss / len(data)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    print(f"Époque {epoch+1}/{n_epochs} - Loss moyenne : {avg_loss:.4f}")

writer.close()