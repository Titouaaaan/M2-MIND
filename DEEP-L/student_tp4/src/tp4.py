
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *
import time
from tqdm import tqdm

#  TODO: 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    #  TODO:  Implémenter maskedCrossEntropy sans aucune boucle, la CrossEntropy qui ne prend pas en compte les caractères de padding.
    seq_len, batch, vocab_size = output.size()
    output = output.reshape(-1, vocab_size)  # seq_len * batch, vocab_size
    target = target.reshape(-1)              # seq_len * batch

    mask = (target != padcar).float()     # seq_len * batch

    loss = nn.functional.cross_entropy(output, target, reduction='none')

    loss = (loss * mask).sum() / mask.sum()

    return loss
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
        # remove the transpose of x here since we already do it in the training loop (maybe here would be better idk?)
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
    # on va garder la meme structure que pour le RNN au niveau des fonctions
    def __init__(self, input_size: int, latent_size: int, output_size: int, device: str):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.device = device

        # on prepare les couches lineaires pour les gates
        self.W_z = nn.Linear(in_features=self.input_size+self.latent_size, out_features=self.latent_size, bias=True, device=device)
        self.W_r = nn.Linear(in_features=self.input_size+self.latent_size, out_features=self.latent_size, bias=True, device=device)
        self.W_h = nn.Linear(in_features=self.input_size+self.latent_size, out_features=self.latent_size, bias=True, device=device)

        #output gate:
        self.o_c = nn.Linear(in_features=latent_size, out_features=output_size, device=device)

    def one_step(self, x, h):
        # concatener x et h
        xh = torch.cat((x, h), dim=1)  # notre batch de donnee

        z_t = torch.sigmoid(self.W_z(xh))  # σ (Wz · [ht−1, xt])
        r_t = torch.sigmoid(self.W_r(xh))  # σ (Wr · [ht−1, xt])

        xh_combined = torch.cat((x, r_t * h), dim=1)  # keep or drop information -> like a drop gate
        h_tilde = torch.tanh(self.W_h(xh_combined))  # new hidden state candidate

        h_new = (1 - z_t) * h + z_t * h_tilde  # (1 − zt) ⊗ ht−1 + zt ⊗ tanh (W · [rt ⊗ ht−1, xt])
        return h_new, z_t, r_t
    
    def forward(self, x, h):
        # ici on part du principe que h sera sous la bonne forme (c.a.d zero dans le premier pass)
        seq_len, batch_size, _ = x.size()
        outputs = []

        # on veut track les gates pour plus tard
        z_t, r_t = None, None

        for i in range(seq_len):
            h_t, z_t, r_t = self.one_step(x[i], h)
            y_t = self.o_c(h_t)
            outputs.append(y_t.unsqueeze(0))  #sequence dimension
        return torch.cat(outputs, dim=0), h_t, z_t, r_t

    def decode(self, h):
        return self.o_c(h) # la couche lineaire qui sert de decoder

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
DATA_PATH = "student_tp3/data/trump_full_speech.txt" # double check path
batch_size = 32
DIM_INPUT = 1
DIM_OUTPUT = len(id2lettre) # "la dimension de sortie du RNN soit égale au nombre de symboles considéré"
EMBEDDING_DIM = DIM_OUTPUT # would have to round if DIM_OUTPUT is odd
print(f'Embedding dim: {EMBEDDING_DIM}')
LATENT_SIZE = 30
lr = 0.001
n_iter = 20

print(f'Using device: {device}')

# load dtaset
data_trump = DataLoader(TrumpDataset(open(DATA_PATH,"rb").read().decode(),maxlen=1000), batch_size= batch_size, shuffle=True)

# setup the model and optimizer
model = RNN(EMBEDDING_DIM,LATENT_SIZE,DIM_OUTPUT,decode_activation=nn.Softmax(dim=1)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# embedding
# donc ici on veut apprednre une representation de nos symboles dans un espace plus peit que celui de base
# on va representer ca avec un couche lineaire pour apprendre ca
embedding = nn.Embedding(DIM_OUTPUT, EMBEDDING_DIM).to(device)

train_losses = []

# also for convenience i lowered the latent space, n_iter and maxlen. We just can increase those params if running on gpu
def train_model():
    model_name = "RNN_trump_exo4"
    for epoch in range(n_iter):
        model.train()
        epoch_loss = 0.0
        time_start = time.time()

        # we do one p bar per epoch that way we can track it nicely
        progress_bar = tqdm(data_trump, desc=f"Epoch {epoch+1}/{n_iter}", leave=False)

        for x, y in progress_bar: # so we iterate over the pbar since it wraps the dataset
            x = x.to(device).long()
            x = embedding(x)
            x = x.transpose(0, 1)  # length x batch x emb_dim

            y = y.to(device).long()
            y = y.transpose(0, 1)  # length x batch
            
            #print(f'x shape: {x.shape}, y shape: {y.shape}')

            optimizer.zero_grad()

            h = torch.zeros((x.size(1), LATENT_SIZE), device=device)
            h = model(x, h)
            y_hat = model.decode(h)

            loss = maskedCrossEntropy(y_hat, y, padcar=0)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        time_epoch = time.time() - time_start
        avg_loss = epoch_loss / len(data_trump)
        train_losses.append(avg_loss)

        print(f"Epoch {epoch+1}/{n_iter} | Loss: {avg_loss:.4f} | Time: {time_epoch:.2f}s")

        # Save model every n epoch epoch for safety
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"{model_name}.pt")
            torch.save(embedding.state_dict(), f"{model_name}_embedding.pt")

    torch.save(model.state_dict(), f"{model_name}.pt")
    torch.save(embedding.state_dict(), f"{model_name}_embedding.pt")
    print(f"Model saved as {model_name}.pt")

# coudln't rly train on my laptop cuz it uses cpu, but gpu should be much faster, ill try at home
# donzo thanks to the almighty gpu

# ===================================
# questions sur LSTM et GRU
# ===================================

# rreminder of useful stuff from above:
# DATA_PATH = "student_tp3/data/trump_full_speech.txt"
DIM_INPUT = 1
DIM_OUTPUT = len(id2lettre) # "la dimension de sortie du RNN soit égale au nombre de symboles considéré"
EMBEDDING_DIM = DIM_OUTPUT # would have to round if DIM_OUTPUT is odd
LATENT_SIZE = 30
embedding = nn.Embedding(DIM_OUTPUT, EMBEDDING_DIM).to(device)

# data loader is this guy:
# data_trump = DataLoader(TrumpDataset(open(DATA_PATH,"rb").read().decode(),maxlen=1000), batch_size= batch_size, shuffle=True)
# device is already defined above too
log_dir="student_tp4/runs/GRU_Training"
lr = 0.001
num_epoch = 20
gru_model = GRU(input_size=EMBEDDING_DIM, latent_size=LATENT_SIZE, output_size=DIM_OUTPUT, device=device) 

def train_lstm_gru(model, dataloader, num_epoch, lr):

    writer = SummaryWriter(log_dir) # tensorboard logging
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()

    global_step = 0

    for epoch in range(num_epoch):
        model.train() # put in train mode to change params
        total_loss = 0.0 # init that 

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        first_batch = True
        for x, y in pbar:
            x = x.to(device).long()
            x = embedding(x)
            x = x.transpose(0, 1)  # length x batch x emb_dim

            y = y.to(device).long()
            y = y.transpose(0, 1)  # length x batch

            batch_size = x.size(1)

            # on oublie pas de mettre le premier etat cache a 0           
            h = torch.zeros(batch_size, model.latent_size, device=device)

            # forward pass
            outputs, h, z_t, r_t = model(x, h)

            loss = loss_function(outputs.reshape(-1, model.output_size), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip pour pas avoir de exploding gradients
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            writer.add_scalar("Loss/train", loss.item(), global_step)

            # logging des valeurs des gates
            if first_batch: # que sur le premier batch sinon y aura trop de trucs
                # track gradients 
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(f"gradients/{name}", param.grad.cpu().data.numpy(), global_step)
                writer.add_histogram("gates/z_t", z_t.detach().cpu().numpy(), global_step)
                writer.add_histogram("gates/r_t", r_t.detach().cpu().numpy(), global_step)
                first_batch = False

            pbar.set_postfix(loss=loss.item())

        if epoch % 5 == 0:
            torch.save(model.state_dict(), "gru_model.pt")
            torch.save(embedding.state_dict(), "gru_model_embedding.pt")


    writer.close()
    torch.save(model.state_dict(), "gru_model.pt")
    torch.save(embedding.state_dict(), "gru_model_embedding.pt")

TRAIN_RNN = False
TRAIN_GRU = True
TRAIN_LSTM = False

if __name__ == "__main__":
    if TRAIN_RNN:
        print("Training RNN model...")
        train_model()

    if TRAIN_GRU:
        print("Training GRU model...")
        train_lstm_gru(gru_model, data_trump, num_epoch, lr)
    if TRAIN_LSTM:
        pass #havent done lstm yet
