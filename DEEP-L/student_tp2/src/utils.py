# This file contains classes implemented from the previous tp
from datamaestro import prepare_dataset
import torch
from torch.autograd import Function
from torch.autograd import gradcheck
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
    def __init__(self, batch_size = 1, shuffle = False):
        ds = prepare_dataset("com.lecun.mnist");

        self.train_images, self.train_labels = ds.train.images.data(), ds.train.labels.data()
        self.test_images, self.test_labels =  ds.test.images.data(), ds.test.labels.data()

    def preprocess(self):
        # preprocess xtrain and test
        x_train = torch.tensor(self.train_images, dtype=torch.double).reshape(-1, 28*28) / 255.0
        x_test = torch.tensor(self.test_images, dtype=torch.double).reshape(-1, 28*28) / 255.0
        
        # get labels as torch tensors
        y_train= torch.tensor(self.train_labels, dtype=torch.long)
        y_test = torch.tensor(self.test_labels, dtype=torch.long)

        #one hot encode the labels 
        y_train = torch.zeros(y_train.size(0), 10, dtype=torch.double)
        y_test = torch.zeros(y_test.size(0), 10, dtype=torch.double)

        return 


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

