from utils import RNN, device,  ForecastMetroDataset

from torch.utils.data import  DataLoader
import torch

from torch.utils.tensorboard import SummaryWriter
import datetime

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32

PATH = "../data/"

LATENT_SIZE = 15

matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch", "rb"))
ds_train = ForecastMetroDataset(
    matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = ForecastMetroDataset(
    matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

#  TODO:  Question 3 : Prédiction de séries temporelles
'''
donc mtn on veut faire de la prediction (donc on est plus dans le cas de la classification).
On veut predire la prochaine valeur d'une sequence, qui contient 2 valeurs continues pour les flux
donc cross entropie plus utile, donc on switch sur la MSE qui va permettre de calculer l'erreur
entre les predictions et les vraies valeurs

Quel coût est dans ce cas plus adapté que la cross-entropie ? MSE loss
Que doit-on changer au modèle précédent ? changer l'output du modele, on sort 2 valeurs continues au lieu d'un label discret (classe)
c.a.d output layer = Linear(latent_size, 2) et sans la softmax
'''

n_iter = 20
learning_rate = 0.001

# model
rnn = RNN(input_size=DIM_INPUT, latent_size=LATENT_SIZE, output_size=DIM_INPUT, decode_activation=None)
rnn = rnn.to(device)

# loss and optimizer
loss_module = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=rnn.parameters(), lr=learning_rate)

# tensorboard logging metrics
curr_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter("runs/run_exo2_"+curr_date)
#recall => writer.add_scalar('Loss/train', loss.item(), n) where loss is the output of the loss module

def predict_t1():
    ''' Predicts the next time step '''
    for epoch in range(n_iter):
        rnn.train()
        train_loss = 0

        for x, y in data_train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            batch, seq_len, n_stations, features = x.shape 

            for station_i in range(n_stations): # we iterate over each stattion
                x_i = x[:, :, station_i, :]  # batch x seq_seqlen x features 

                h = torch.zeros((batch, LATENT_SIZE), device=device) # set to 0 like previously
                h_out = rnn(x_i, h)
                pred = rnn.decode(h_out[-1])  # batch x features

                y_target = y[:, -1, station_i, :]  # take the next step of the station
                # also we train on each batch
                loss = loss_module(pred, y_target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                # print(loss.item())

        train_loss /= len(data_train) * n_stations # take avg of loss over the stations

        rnn.eval() # again make sure we change to eval to not change the weights
        test_loss = 0

        with torch.no_grad():
            for x, y in data_test:
                x, y = x.to(device), y.to(device)
                batch, seq_len, n_stations, features = x.shape

                for station_i in range(n_stations):
                    x_i = x[:, :, station_i, :]
                    h = torch.zeros((batch, LATENT_SIZE), device=device)
                    h_out = rnn(x_i, h)
                    pred = rnn.decode(h_out[-1])
                    y_target = y[:, -1, station_i, :]
                    loss = loss_module(pred, y_target)
                    test_loss += loss.item()

        test_loss /= len(data_test) * n_stations
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        print(f"Epoch {epoch} | Train loss = {train_loss:.4f} | Test loss = {test_loss:.4f}")

def predict_tH(H: int):
    ''' Predicts the next h time steps of a sequence'''
    for epoch in range(n_iter):
        rnn.train()
        train_loss = 0

        for x, y in data_train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            batch, seq_len, n_stations, features = x.shape

            for station_i in range(n_stations):
                x_i = x[:, :, station_i, :]  # batch x seq_len x features
                h = torch.zeros((batch, LATENT_SIZE), device=device)
                
                # we take the last in/out step of the stations
                x_next = x_i[:, -1, :]  # batch x features
                preds = [] #keep track of the preds

                for t in range(H): # for each step in the future that we wanna predict
                    h = rnn.one_step(x_next, h) # pred
                    pred_t = rnn.decode(h) # decode it
                    preds.append(pred_t) #save it
                    x_next = pred_t  # and use that prediction as the previous in/out value for the next prediction

                preds = torch.stack(preds, dim=1)    # batch x h x features
                y_target = y[:, -H:, station_i, :]   # same size as preds
                loss = loss_module(preds, y_target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        train_loss /= len(data_train) * n_stations

        rnn.eval()
        test_loss = 0

        with torch.no_grad():
            for x, y in data_test:
                x, y = x.to(device), y.to(device)
                batch, seq_len, n_stations, features = x.shape

                for station_i in range(n_stations):
                    x_i = x[:, :, station_i, :]
                    h = torch.zeros((batch, LATENT_SIZE), device=device)
                    x_next = x_i[:, -1, :]
                    preds = []

                    for t in range(H):
                        h = rnn.one_step(x_next, h)
                        pred_t = rnn.decode(h)
                        preds.append(pred_t)
                        x_next = pred_t

                    preds = torch.stack(preds, dim=1)
                    y_target = y[:, -H:, station_i, :]
                    loss = loss_module(preds, y_target)
                    test_loss += loss.item()

        test_loss /= len(data_test) * n_stations
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        print(f"Epoch {epoch} | Train loss = {train_loss:.4f} | Test loss = {test_loss:.4f}")

predict_t1()

predict_tH(H=3)
