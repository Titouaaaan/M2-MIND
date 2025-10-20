from utils import RNN, device,SampleMetroDataset
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 50
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32

# latent space size hyperparameter (choose based on data length ig)
LATENT_SIZE = 15

PATH = "../data/" # make sure you are in /src to run the code


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length = LENGTH, stations_max = ds_train.stations_max)
data_train = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE,shuffle=False)


#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence
# hyperparam
n_iter = 20
learning_rate = 0.001

# model
rnn = RNN(input_size=DIM_INPUT, latent_size=LATENT_SIZE, output_size=CLASSES, decode_activation=torch.nn.Softmax(dim=1))
rnn = rnn.to(device)

# loss and optimizer
loss_module = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=rnn.parameters(), lr=learning_rate)

# tensorboard logging metrics
curr_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter("runs/run_exo2_"+curr_date)
#recall => writer.add_scalar('Loss/train', loss.item(), n) where loss is the output of the loss module

for epoch in range(n_iter):
    rnn.train()
    train_loss = 0

    for x, y in data_train:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        h = torch.zeros((x.size(0), LATENT_SIZE), device=device)
        h = rnn(x, h)
        pred = rnn.decode(h[-1])

        loss = loss_module(pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(data_train)

    # eval
    rnn.eval() # change the mode for the model
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad(): # ensure the gradients are not calculated/saved
        for x, y in data_test:
            x, y = x.to(device), y.to(device)
            h = torch.zeros((x.size(0), LATENT_SIZE), device=device)
            h = rnn(x, h)
            pred = rnn.decode(h[-1])

            loss = loss_module(pred, y)
            test_loss += loss.item()

            # check if we predicted properly, by checking the accracy of label prediction
            predicted = pred.argmax(dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0) # for bacth

    test_loss /= len(data_test)
    accuracy = correct / total * 100

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)

    print(f"Epoch {epoch} | Train loss = {train_loss:.4f} | Test loss = {test_loss:.4f} | Acc = {accuracy:.2f}%")

'''
Petites remarques apres avoir fait des tests:
Avec une sequence du longeur de 20, et un latent size (celui la on le fait pas varier pour le moment) de 15 
on se retrouve une accuracy vers la fin du train d'environ 30% donc pas incroyable

Par contre avec une longeur de 30 on arrive deja a monter dans les 45% d'accuracy (avec un pic a 53% carrement)
Et si on monte a 50 l'accuracy augmente pas vraiment, donc a priori il faudrait plutot augmenter le latent space size 
pour ameliorer les perfs

Ex: length=30,
Epoch 0 | Train loss = 2.3012 | Test loss = 2.2892 | Acc = 19.90%
Epoch 1 | Train loss = 2.2437 | Test loss = 2.2201 | Acc = 20.23%
Epoch 2 | Train loss = 2.1926 | Test loss = 2.1551 | Acc = 30.63%
Epoch 3 | Train loss = 2.1251 | Test loss = 2.1041 | Acc = 37.38%
Epoch 4 | Train loss = 2.0901 | Test loss = 2.0831 | Acc = 39.24%
Epoch 5 | Train loss = 2.0700 | Test loss = 2.0619 | Acc = 41.00%
Epoch 6 | Train loss = 2.0605 | Test loss = 2.0368 | Acc = 44.98%
Epoch 7 | Train loss = 2.0319 | Test loss = 2.0443 | Acc = 40.53%
Epoch 8 | Train loss = 2.0306 | Test loss = 2.0526 | Acc = 41.73%
Epoch 9 | Train loss = 2.0107 | Test loss = 2.0046 | Acc = 47.74%
Epoch 10 | Train loss = 2.0158 | Test loss = 2.0116 | Acc = 45.55%
Epoch 11 | Train loss = 1.9949 | Test loss = 2.0175 | Acc = 45.48%
Epoch 12 | Train loss = 2.1387 | Test loss = 2.0426 | Acc = 42.03%
Epoch 13 | Train loss = 2.0058 | Test loss = 2.0101 | Acc = 46.21%
Epoch 14 | Train loss = 1.9920 | Test loss = 2.0981 | Acc = 36.11%
Epoch 15 | Train loss = 1.9876 | Test loss = 1.9724 | Acc = 53.79%
Epoch 16 | Train loss = 1.9585 | Test loss = 2.0174 | Acc = 44.82%
Epoch 17 | Train loss = 1.9371 | Test loss = 2.0152 | Acc = 45.25%
Epoch 18 | Train loss = 1.9734 | Test loss = 2.0006 | Acc = 47.87%
Epoch 19 | Train loss = 1.9987 | Test loss = 2.0466 | Acc = 43.65%

et pour length=50,
Epoch 0 | Train loss = 2.3017 | Test loss = 2.2931 | Acc = 20.75%
Epoch 1 | Train loss = 2.2480 | Test loss = 2.2208 | Acc = 20.00%
Epoch 2 | Train loss = 2.2126 | Test loss = 2.2068 | Acc = 20.93%
Epoch 3 | Train loss = 2.1896 | Test loss = 2.1619 | Acc = 35.03%
Epoch 4 | Train loss = 2.1529 | Test loss = 2.1268 | Acc = 39.13%
Epoch 5 | Train loss = 2.1209 | Test loss = 2.1164 | Acc = 39.19%
Epoch 6 | Train loss = 2.1258 | Test loss = 2.1492 | Acc = 28.94%
Epoch 7 | Train loss = 2.1296 | Test loss = 2.1177 | Acc = 37.08%
Epoch 8 | Train loss = 2.1270 | Test loss = 2.1093 | Acc = 34.35%
Epoch 9 | Train loss = 2.1142 | Test loss = 2.1161 | Acc = 33.42%
Epoch 10 | Train loss = 2.1039 | Test loss = 2.0896 | Acc = 36.21%
Epoch 11 | Train loss = 2.0901 | Test loss = 2.0851 | Acc = 40.00%
Epoch 12 | Train loss = 2.0641 | Test loss = 2.0627 | Acc = 41.30%
Epoch 13 | Train loss = 2.0593 | Test loss = 2.0999 | Acc = 37.39%
Epoch 14 | Train loss = 2.0587 | Test loss = 2.0515 | Acc = 42.42%
Epoch 15 | Train loss = 2.0491 | Test loss = 2.0320 | Acc = 44.97%
Epoch 16 | Train loss = 2.0483 | Test loss = 2.0327 | Acc = 44.41%
Epoch 17 | Train loss = 2.0346 | Test loss = 2.0315 | Acc = 46.21%
Epoch 18 | Train loss = 2.0341 | Test loss = 2.0261 | Acc = 45.78%
Epoch 19 | Train loss = 2.0253 | Test loss = 2.0157 | Acc = 45.78%
'''

