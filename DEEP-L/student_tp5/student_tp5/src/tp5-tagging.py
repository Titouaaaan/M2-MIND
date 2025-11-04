import itertools
import logging
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import List
import time
from conllu import parse_incr
logging.basicConfig(level=logging.INFO)

DATA_PATH = "../data/"


# Format de sortie décrit dans
# https://pypi.org/project/conllu/

class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement s'il n'est pas connu
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        """ oov : autorise ou non les mots OOV """
        self.oov =  oov
        self.id2word = [ "PAD"]
        self.word2id = { "PAD" : Vocabulary.PAD}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self,idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self,idx: List[int]):
        return [self.getword(i) for i in idx]



class TaggingDataset():
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        self.sentences = []

        for s in data:
                self.sentences.append(([words.get(token["form"], adding) for token in s], [tags.get(token["upostag"], adding) for token in s]))
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, ix):
        return self.sentences[ix]


def collate_fn(batch):
    """Collate using pad_sequence"""
    return tuple(pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2))

logging.info("Loading datasets...")
words = Vocabulary(True)
tags = Vocabulary(False)

data_file = open(DATA_PATH+"fr_gsd-ud-train.conllu",encoding="utf-8")
train_data = TaggingDataset(parse_incr(data_file), words, tags, True)

data_file = open(DATA_PATH+"fr_gsd-ud-dev.conllu",encoding='utf-8')
dev_data = TaggingDataset(parse_incr(data_file), words, tags, True)

data_file = open(DATA_PATH+"fr_gsd-ud-test.conllu",encoding="utf-8")
test_data = TaggingDataset(parse_incr(data_file), words, tags, False)


logging.info("Vocabulary size: %d", len(words))


BATCH_SIZE=100

train_loader = DataLoader(train_data, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)

def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            pred = model(x)                     # [seq_len, batch, num_tags]
            pred = pred.argmax(dim=-1)          # [seq_len, batch] — pick highest scoring tag

            # compare predictions to gold labels
            correct += (pred == y).sum().item()
            total += y.numel()

    acc = correct / total
    return acc

#  TODO:  Implémenter le modèle et la boucle d'apprentissage (en utilisant les LSTMs de pytorch)
class seq2seq(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedder = nn.Embedding(input_size, embedding_size, device=device) 
        self.lstm = nn.LSTM(embedding_size, hidden_size, device=device)
        self.f_h = nn.Linear(hidden_size, input_size, device=device)
    
    def forward(self, x):
        emb = self.embedder(x)
        y, _ = self.lstm(emb)
        output = self.f_h(y)
        return output

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device {device}')
seq2seq_model = seq2seq(len(words), 128, 64, device).to(device)

def train_seq2seq(model: seq2seq, train:DataLoader, dev: DataLoader, test: DataLoader, num_epoch: int):
    # train the tagger
    loss_module = nn.CrossEntropyLoss()
    writer = SummaryWriter('/student_tp5/runs') # tensorboard logging
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    global_step = 0

    for epoch in range(num_epoch):
        model.train() # put in train mode to change params
        total_loss = 0.0 # init that 

        pbar = tqdm(train, desc=f"Epoch {epoch+1}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device) # this is size seq x batch

            # print(x.shape, y.shape)

            pred = model(x) # torch.Size([58, 100, 44939]) => seq_len x batch_size x vocab_size
            # print(pred.shape)
            pred = pred.view(-1, pred.size(-1)) # flatten to seq_len * batch_size x vocab_size 
            y = y.view(-1)  # also flatten y otherwise shape mismatch
            # print(pred.shape, y.shape)
            loss = loss_module(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            writer.add_scalar("Loss train", loss.item(), global_step)

            pbar.set_postfix(loss=loss.item())
        train_acc = evaluate_accuracy(model, train_loader, device)
        test_acc = evaluate_accuracy(model, test_loader, device)
        writer.add_scalar("Accuracy train", train_acc, global_step)
        writer.add_scalar("Accuracy test", test_acc, global_step)
        tqdm.write(f"Epoch {epoch+1}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")
        
    writer.close()
    torch.save(model.state_dict(), 'tagger.pt') # this saves it in outside of the src folder

train_seq2seq(seq2seq_model, train_loader, dev_loader, test_loader, 30)
# the run is available in the 'runs' folder !

""" 
data_file = open(DATA_PATH+"fr_gsd-ud-train.conllu",encoding="utf-8")
train_data = TaggingDataset(parse_incr(data_file), words, tags, True)

data_file = open(DATA_PATH+"fr_gsd-ud-dev.conllu",encoding='utf-8')
dev_data = TaggingDataset(parse_incr(data_file), words, tags, True)

data_file = open(DATA_PATH+"fr_gsd-ud-test.conllu",encoding="utf-8")
test_data = TaggingDataset(parse_incr(data_file), words, tags, False)

model = seq2seq(len(words), 128, 64, device).to(device)
model.load_state_dict(torch.load("tagger.pt", map_location=device))
model.eval()

def test_tagger(model, dataloader, idx2word, idx2tag, device, num_examples=3):
    model.eval()
    shown = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=-1)  # seq_len x batch

            for b in range(x.shape[1]):
                tokens = [idx2word[i.item()] for i in x[:, b]]
                gold_tags = [idx2tag[i.item()] for i in y[:, b]]
                pred_tags = [idx2tag[i.item()] for i in pred[:, b]]

                print(f"Sentence {shown+1}")
                print("Words:     ", " ".join(tokens))
                print("Gold tags: ", " ".join(gold_tags))
                print("Pred tags: ", " ".join(pred_tags))
                print("-" * 100)

                shown += 1
                if shown >= num_examples:
                    return

test_tagger(model, train_loader, words.id2word, tags.id2word, device, num_examples=5) """