
import torch
from torch.autograd import Function
from torch.autograd import gradcheck


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

## Utile dans ce TP que pour le script tp1_gradcheck
mse = MSE.apply
linear = Linear.apply

# Les données supervisées
x = torch.randn(50, 13)
y = torch.randn(50, 3)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3)
b = torch.randn(3)
context_linear = Context()
context_mse = Context()


