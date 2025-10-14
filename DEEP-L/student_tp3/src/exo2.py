from utils import RNN, device,SampleMetroDataset
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
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
rnn = RNN(input_size=DIM_INPUT, latent_size=LATENT_SIZE, output_size=CLASSES, decode_activation=torch.nn.Softmax)
rnn = rnn.to(device)

# loss and optimizer
loss_module = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=rnn.parameters(), lr=learning_rate)

# tensorboard logging metrics
curr_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter("runs/run_exo2_"+curr_date)
#recall => writer.add_scalar('Loss/train', loss.item(), n) where loss is the output of the loss module

for i in n_iter:
    for x, y in data_train: # batches of shuffled data sequences
        # print(x.shape, y) => torch.Size([32, 20, 2]) and y = BATCH_SIZE tensor of labels 1 - 10
        # copy to device (gpu ideally)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        h = torch.zeros((x.size(0), LATENT_SIZE), device=device) # h0
        h = rnn(x, h)

        pred = rnn.decode(h[:-1]) # decode the last output
        loss = loss_module(pred, y)
        writer.add_scalar('Loss/train', loss.item(), i) # not sure if this is correct? mighht need to deal with batch

        loss.backward()
        optimizer.step()
    
    # eval


