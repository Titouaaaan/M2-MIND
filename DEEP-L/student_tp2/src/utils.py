# This file contains classes implemented from the previous tp
from datamaestro import prepare_dataset
import torch
from torch.autograd import Function
from torch.autograd import gradcheck
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# for tp 2 we dont actually need this anymore, since the torch optimizer deals with it
class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """Début d'implementation de la fonction MSE"""
    @staticmethod
    def forward(ctx, yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y)

        return ((yhat - y) ** 2).mean()

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors

        N = yhat.numel()  # nombre total d'éléments dans yhat pour la normalisation
        gradyhat = (2.0 / N) * (yhat - y) * grad_output
        grady = (-2.0 / N) * (yhat - y) * grad_output
        return gradyhat, grady
    
class Linear(Function): 
    @staticmethod
    def forward(ctx, X, W, b):
        ctx.save_for_backward(X, W, b)

        fwd = X @ W + b
        return fwd
    
    @staticmethod
    def backward(ctx, grad_output):
        X, W, b = ctx.saved_tensors

        grad_X = grad_output @ W.T
        grad_W =  X.T @ grad_output 
        grad_b = grad_output.sum(0, keepdims=True)
        return grad_X, grad_W, grad_b

class ComplexNN(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexNN, self).__init__()
        # layers
        self.first_layer = torch.nn.Linear(in_features,100)
        self.activation1 = torch.nn.Tanh()
        self.second_layer = torch.nn.Linear(100, out_features)
    def forward(self, x):
        return self.second_layer(self.activation1(self.first_layer(x)))

class MNISTDataLoader():
    def __init__(self, batch_size=32, shuffle=True, train=True):
        ds = prepare_dataset("com.lecun.mnist");
        if train:
            images, labels = ds.train.images.data(), ds.train.labels.data()
        else:
            images, labels = ds.test.images.data(), ds.test.labels.data()

        self.x = torch.tensor(images, dtype=torch.long).reshape(-1, 28*28) / 255.0
        self.y = torch.tensor(labels, dtype=torch.long)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.x)) # creates a list will all the indices, likely 0 to 60000 in our case
    
    def __getitem__(self, index):
        # return a specific data point with its label
        return self.x[index], self.y[index]

    def __len__(self): # we get the length according to the batch size and we round it (to not get a float)
        return int(np.ceil(len(self.x) / self.batch_size))
    
    def __iter__(self): # used before starting the iteration on the data
        # we take the list of the indices and shuffle them: ([0,1,2,3]) -> ([2,1,0,3]) forexample
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0
        return self
    
    def __next__(self): # the iteration
        if self.current_idx >= len(self.x): #check to see that we're around out of bounds of the DS
            raise StopIteration

        # get the start and ending index of the batch
        start = self.current_idx 
        end = min(start + self.batch_size, len(self.x))
        batch_idx = self.indices[start:end]
        self.current_idx = end
        #fetch the data
        x_batch = self.x[batch_idx]
        y_batch = self.y[batch_idx]
        return x_batch, y_batch

class AutoEncoder(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(AutoEncoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.encoder = torch.nn.Linear(self.in_features, self.out_features) # encoder linear layer
        self.activation = torch.nn.ReLU() 
        # here we dont specify the weights for the decoder since we will just reuse the weights of enc in forward()
        self.decoder_b = torch.nn.Parameter(torch.zeros(self.in_features)) # just the b bias for the decoder

    def forward(self, x):
        enc_fwd = self.activation(self.encoder(x)) # the encoder forward pass
        # here the decoder forward is the sigmoid activation function times (enc_output @ enc weights + bias dec)
        # here i think the reason why we can just use the weights is bc in the encoder forward pass,
        # we do x @ W.T, so technically using the 'tied' transposed weights means we just so z @ W for the decoder
        dec_fwd = torch.sigmoid((enc_fwd @ self.encoder.weight) + self.decoder_b)
        return dec_fwd # this is the f(x) reconstructed data 

# this is probably going to be outdated very quickly but its from the first tp, tp1_descente.py
""" def gradient_descent(x, y, w, b, eps):
    writer = SummaryWriter()

    for n_iter in range(100): # on pourrait meme aller vers 150 iterations pcq ca pourrait peut etre converger un peu plus
        # attention a bien avoir deux contextes differents pour track les valeurs
        context_linear = Context() 
        context_mse = Context()
        # Forward pass
        y_pred = Linear.forward(context_linear, x, w, b)
        loss = MSE.forward(context_mse, y_pred, y).mean()  # moyenne sur le batch

        # Enregistrement du loss pour TensorBoard
        writer.add_scalar('Loss/train', loss.item(), n_iter)
        print(f"Itérations {n_iter}: loss {loss.item()}")

        # Backward pass
        grad_loss = torch.ones_like(loss)
        grad_y_pred, _ = MSE.backward(context_mse, grad_loss)
        grad_x, grad_w, grad_b = Linear.backward(context_linear, grad_y_pred)

        # Mise à jour des paramètres
        with torch.no_grad():
            w -= eps * grad_w
            b -= eps * grad_b
    writer.close()
    return None """

def optim_GD(x, y, w, b, eps, opt, niter):
    # no need for context anymore
    if opt == 'SGD':
        optim = torch.optim.SGD(params=[w,b], lr=eps)
    elif opt == 'Adam':
        optim = torch.optim.Adam(params=[w,b], lr=eps)
    else:
        return 'Invalid optimizer!!'
    optim.zero_grad()
    writer = SummaryWriter()
    for n in range(niter):
        ypred = Linear.apply(x, w, b)
        loss = MSE.apply(ypred, y)
        writer.add_scalar('Loss/train', loss.item(), n)
        print(f"Itérations {n}: loss {loss.item()}")
        loss.backward()

        optim.step()
        optim.zero_grad()
    writer.close()
    # now we gotta save the model
    torch.save({'w': w, 'b': b}, 'simpleMNIST.pth')
    return None

def evaluate_model(w, b, x_test, y_test):
    pass

def test_optimizer(optim='SGD'):
    # Les données supervisées
    features = 13
    data_size = 50
    output_size = 3
    x = torch.randn(data_size, features)
    y = torch.randn(data_size, output_size)

    # Les paramètres du modèle à optimiser
    w = torch.nn.Parameter(torch.randn(output_size,features))
    b = torch.nn.Parameter(torch.randn(output_size))

    epsilon = 0.05
    optim_GD(x,y,w,b,epsilon,optim, niter=100)

