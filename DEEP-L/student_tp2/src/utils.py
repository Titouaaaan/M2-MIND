# This file contains classes implemented from the previous tp

import torch
from torch.autograd import Function
from torch.autograd import gradcheck
from torch.utils.tensorboard import SummaryWriter

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

        fwd = X @ W.T + b
        return fwd
    
    @staticmethod
    def backward(ctx, grad_output):
        X, W, b = ctx.saved_tensors

        grad_X = grad_output @ W
        grad_W =  grad_output.T @ X 
        grad_b = grad_output.sum(0, keepdims=True)
        return grad_X, grad_W, grad_b

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

def optim_GD(x, y, w, b, eps, opt):
    # no need for context anymore
    if opt == 'SGD':
        optim = torch.optim.SGD(params=[w,b], lr=eps)
    elif opt == 'Adam':
        optim = torch.optim.Adam(params=[w,b], lr=eps)
    else:
        return 'Invalid optimizer!!'
    optim.zero_grad()
    writer = SummaryWriter()
    for n in range(100):
        ypred = Linear.apply(x, w, b)
        loss = MSE.apply(ypred, y)
        writer.add_scalar('Loss/train', loss.item(), n)
        print(f"Itérations {n}: loss {loss.item()}")
        loss.backward()

        optim.step()
        optim.zero_grad()
    writer.close()
    return None

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
    optim_GD(x,y,w,b,epsilon,optim)

