
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *
import time

#  TODO: 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    #  TODO:  Implémenter maskedCrossEntropy sans aucune boucle, la CrossEntropy qui ne prend pas en compte les caractères de padding.
    # first we create the mask where its 1 if not padding, 0 else
    mask = target != padcar # 1 if target different than padcar else 0 -> [1, 1, 1, 1, 0, 0, 0, 0...]
    output *= mask
    return CrossEntropyLoss(output, target, reduce='None').mean()  # mean since we work over batches

class RNN(nn.Module):
    # data: X = length x batch x dim
        # Xt = batch x dim
        # Ht = batch x latent
        # Wi = dim x latent
        # Wh = latent x latent
        # bias = latent
        # so ze need two linear layers, one for Wi and one for Wh

    def __init__(self, input_size: int, latent_size: int, output_size: int, decode_activation: nn.Module | None):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.Wi = nn.Linear(in_features=self.input_size, out_features=self.latent_size, bias=True, device=device)
        self.Wh = nn.Linear(in_features=self.latent_size, out_features=self.latent_size, bias=False, device=device)
        self.d_a = decode_activation
        self.decoder = nn.Linear(in_features=latent_size, out_features=output_size, device=device)
    
    def one_step(self, x, h):
        # ht = input x latent
        ht = torch.tanh(self.Wi(x) + self.Wh(h))
        return ht

    def forward(self, x, h):
        # for each elem x in the batch X, we want to one_step it with h
        # output = batch_size x latent_size 
        x = x.transpose(0, 1)
        length = x.size(0)
        batch_size = x.size(1)
        h_output = torch.zeros(length, batch_size, self.latent_size).to(device) 
        for i in range(length): # loop over the x sequence
            h = self.one_step(x[i], h) # apply 
            h_output[i] = h # saves the current h (memory!) used to decode
        return h_output

    def decode(self, h):
        # yt = d(h) = latent x output
        if self.d_a is None:
            return self.decoder(h)
        else:
            return self.d_a(self.decoder(h))


class LSTM(RNN):
    #  TODO:  Implémenter un LSTM
    pass


class GRU(nn.Module):
    #  TODO:  Implémenter un GRU
    pass


#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot
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

# hyperparameters
DATA_PATH = "../student_tp3/data/trump_full_speech.txt" #run from the src folder
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
data_trump = DataLoader(TrumpDataset(open(DATA_PATH,"rb").read().decode(),maxlen=1000), batch_size= batch_size, shuffle=True)

# setup the model and optimizer
model = RNN(EMBEDDING_DIM,LATENT_SIZE,DIM_OUTPUT,decode_activation=nn.Softmax(dim=1)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_module = nn.CrossEntropyLoss()

# embedding
# donc ici on veut apprednre une representation de nos symboles dans un espace plus peit que celui de base
# on va representer ca avec un couche lineaire pour apprendre ca
embedding = nn.Embedding(DIM_OUTPUT, EMBEDDING_DIM).to(device)

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