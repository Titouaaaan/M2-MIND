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
        N = yhat.numel()  # nombre total d'éléments dans yhat
        gradyhat = 2.0 / N * (yhat - y) * grad_output
        grady = -2.0 / N * (yhat - y) * grad_output
        return gradyhat, grady
    
class Linear(Function): 
    @staticmethod
    def forward(ctx, X, W, b):
        ctx.save_for_backward(X, W, b)
        return X @ W + b
    
    @staticmethod
    def backward(ctx, grad_output):
        X, W, b = ctx.saved_tensors
        grad_X = grad_output @ W.T
        grad_W =  X.T @ grad_output 
        grad_b = grad_output.sum(0)
        return grad_X, grad_W, grad_b

# this is probably going to be outdated very quickly but its from the first tp, tp1_descente.py
def gradient_descent(x, y, w, b, eps):
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
    return None