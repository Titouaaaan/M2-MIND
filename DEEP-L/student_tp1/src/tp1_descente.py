import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context


# Les données supervisées
x = torch.randn(50, 13)
y = torch.randn(50, 3)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

epsilon = 0.05

writer = SummaryWriter()
context = Context()

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
        w -= epsilon * grad_w
        b -= epsilon * grad_b

writer.close()