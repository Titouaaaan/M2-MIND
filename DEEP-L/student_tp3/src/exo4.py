import string
import unicodedata
import torch
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader

from utils import RNN, device
import time

## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)
    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long),t])
        return t[:-1],t[1:]



#  TODO: 
# hyperparameters
PATH = "../data/"
batch_size = 32
DIM_INPUT = 1
DIM_OUTPUT = len(id2lettre) # "la dimension de sortie du RNN soit égale au nombre de symboles considéré"
EMBEDDING_DIM = DIM_OUTPUT//2 # would have to round if DIM_OUTPUT is odd
print(f'Embedding dim: {EMBEDDING_DIM}')
LATENT_SIZE = 30
lr = 0.001
n_iter = 50

print(f'Using device: {device}')

# load dtaset
data_trump = DataLoader(TrumpDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=1000), batch_size= batch_size, shuffle=True)

# setup the model and optimizer
model = RNN(EMBEDDING_DIM,LATENT_SIZE,DIM_OUTPUT,decode_activation=nn.Softmax(dim=1)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_module = nn.CrossEntropyLoss()

# embedding
# donc ici on veut apprednre une representation de nos symboles dans un espace plus peit que celui de base
# on va representer ca avec un couche lineaire pour apprendre ca
embedding = nn.Linear(DIM_OUTPUT, EMBEDDING_DIM).to(device)

train_losses = []

def train_model():
    model_name = "RNN_trump_exo4"
    for epoch in range(n_iter):
        model.train()
        epoch_loss = 0
        time_start = time.time()
        for x, y in data_trump:
            x = nn.functional.one_hot(x, num_classes=DIM_OUTPUT).float().to(device)
            x = embedding(x) 

            y = y.to(device).long()
            
            optimizer.zero_grad()

            h = torch.zeros((x.size(0), LATENT_SIZE), device=device)  
            h = model(x, h)
            y_hat = model.decode(h)

            y_hat = y_hat.transpose(0, 1).transpose(1, 2)
 
            loss = loss_module(y_hat, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print("batch loss:", loss.item())
        time_batch = time.time() - time_start
        print(f"Epoch {epoch} completed in {time_batch:.2f} seconds.")
        train_losses.append(epoch_loss / len(data_trump))
        print("step:", epoch)
        print("Loss_train:", float(train_losses[-1]))
    # save the model
    torch.save(model.state_dict(), f"{model_name}.pt")
    torch.save(embedding.state_dict(), f"{model_name}_embedding.pt")

train_model()

'''
Note: 
Epoch 0 completed in 209.74 seconds.
step: 0
Loss_train: 4.556160144591599

This was for the first iteration T_T

si y en a pour 4min environ par iteration et qu'il en faut  50 (j'image que moins que ca pourrait suffire)
on en aurait pour 3h de train total...
Je ferais peut etre pendant la nuit, mais pour le moment je laisse comme ca
'''

# TODO: fonction pour generer du texte a partir du modele 