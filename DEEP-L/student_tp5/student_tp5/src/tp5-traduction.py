import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import unicodedata
import string
from tqdm import tqdm
from pathlib import Path
from typing import List
import pickle

import time
import re
from torch.utils.tensorboard import SummaryWriter




logging.basicConfig(level=logging.INFO)

FILE = "../../data/en-fra.txt"

writer = SummaryWriter('/student_tp5/runs')

def normalize(s):
    return re.sub(' +',' ', "".join(c if c in string.ascii_letters else " "
         for c in unicodedata.normalize('NFD', s.lower().strip())
         if  c in string.ascii_letters+" "+string.punctuation)).strip()


class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    PAD = 0
    EOS = 1
    SOS = 2
    OOVID = 3

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {"PAD": Vocabulary.PAD, "EOS": Vocabulary.EOS, "SOS": Vocabulary.SOS}
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

    def getword(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]



class TradDataset():
    def __init__(self,data,vocOrig,vocDest,adding=True,max_len=10):
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            self.sentences.append((torch.tensor([vocOrig.get(o) for o in orig.split(" ")]+[Vocabulary.EOS]),torch.tensor([vocDest.get(o) for o in dest.split(" ")]+[Vocabulary.EOS])))
    def __len__(self):return len(self.sentences)
    def __getitem__(self,i): return self.sentences[i]



def collate_fn(batch):
    orig,dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig),o_len,pad_sequence(dest),d_len


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open(FILE) as f:
    lines = f.readlines()

lines = [lines[x] for x in torch.randperm(len(lines))]
idxTrain = int(0.8*len(lines))

vocEng = Vocabulary(True)
vocFra = Vocabulary(True)
MAX_LEN=100
BATCH_SIZE=100

datatrain = TradDataset("".join(lines[:idxTrain]),vocEng,vocFra,max_len=MAX_LEN)
datatest = TradDataset("".join(lines[idxTrain:]),vocEng,vocFra,max_len=MAX_LEN)

train_loader = DataLoader(datatrain, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datatest, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)

#  TODO:  Implémenter l'encodeur, le décodeur et la boucle d'apprentissage
class Encoder(nn.Module):
    def __init__(self, input_vocab_size, hidden_size, embedding_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size)

    def forward(self, input):
        embedded = self.embedding(input)
        return self.gru(embedded)

class Decoder(nn.Module):
    def __init__(
        self, output_vocab_size, hidden_size, embedding_dim, max_length=MAX_LEN
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.embedding = nn.Embedding(output_vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.to_vocab = nn.Linear(hidden_size, output_vocab_size)

    def one_step(self, input, hidden):
        output = self.embedding(input)
        output, h = self.gru(output, hidden)
        output = self.to_vocab(output)
        return output, h

    def forward(self, encoder_outputs, encoder_hidden, lens_seq, target_tensor=None):
        batch_size = encoder_outputs.size(1)
        decoder_input = torch.empty(
            1, batch_size, dtype=torch.long, device=device
        ).fill_(Vocabulary.SOS)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(lens_seq):
            decoder_output, decoder_hidden = self.one_step(
                decoder_input, decoder_hidden
            )

            if target_tensor is not None:
                decoder_input = target_tensor[i, :].unsqueeze(0)  #teacher forcing
            else:
                #if not then the decoder preds are the used inputs
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach() 
            decoder_outputs.append(decoder_output) #save it

        decoder_outputs = torch.cat(decoder_outputs, dim=0)
        return decoder_outputs, decoder_hidden

def train_traduction(
    encoder, decoder, train_loader,
    n_epochs, teacher_forcing_rate=0.5, lr=0.002, max_grad_norm=1.0
):
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)

    encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        encoder_optimizer, mode='min', factor=0.5, patience=2
    )
    decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        decoder_optimizer, mode='min', factor=0.5, patience=2
    )

    criterion = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD)

    for epoch in range(n_epochs):
        encoder.train()
        decoder.train()
        total_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")

        for batch_idx, (src, src_len, tgt, tgt_len) in enumerate(progress):
            src, tgt = src.to(device), tgt.to(device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = encoder(src)

            #random teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_rate
            tgt_tensor = tgt if use_teacher_forcing else None

            decoder_outputs, _ = decoder(
                encoder_outputs,
                encoder_hidden,
                tgt.size(0),  # max target length
                target_tensor=tgt_tensor
            )

            loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), tgt.view(-1))

            loss.backward()

            # bonus clip
            nn.utils.clip_grad_norm_(encoder.parameters(), max_grad_norm)
            nn.utils.clip_grad_norm_(decoder.parameters(), max_grad_norm)

            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

            writer.add_scalar(
                "Loss/batch",
                loss.item(),
                epoch * len(train_loader) + batch_idx
            )

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        encoder_scheduler.step(avg_loss)
        decoder_scheduler.step(avg_loss)

        writer.add_scalar("Loss/train", avg_loss, epoch)

    writer.close()
    torch.save(encoder.state_dict(), "encoder.pt")
    torch.save(decoder.state_dict(), "decoder.pt")
    
    #save the vocabulary
    with open("vocEng.pkl", "wb") as f:
        pickle.dump(vocEng, f)
    with open("vocFra.pkl", "wb") as f:
        pickle.dump(vocFra, f)

train_traduction(
    Encoder(len(vocEng), 128, 64).to(device),
    Decoder(len(vocFra), 128, 64).to(device),
    train_loader,
    50,
)   

def translate_sentence(encoder, decoder, sentence, src_vocab, tgt_vocab, max_len=MAX_LEN):
    # make sure its in eval
    encoder.eval()
    decoder.eval()

    # very basic tokenizing on the sentence
    tokens = sentence.lower().strip().split()  
    src_indices = [src_vocab.get(tok, adding=False) for tok in tokens] # convert to indices
    src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(1).to(device)  # proper dim convertion

    with torch.no_grad():
        _, encoder_hidden = encoder(src_tensor) # we only need the hidden state

    decoder_input = torch.tensor([[Vocabulary.SOS]], device=device)  # start token
    decoder_hidden = encoder_hidden
    predicted_indices = []

    for _ in range(max_len):
        with torch.no_grad():
            decoder_output, decoder_hidden = decoder.one_step(decoder_input, decoder_hidden)
            # pick most probable token
            _ , topi = decoder_output.topk(1)
            next_idx = topi.item()
            if next_idx == Vocabulary.EOS:  # stop at EOS
                break
            predicted_indices.append(next_idx)
            # feed the predicted token as next input
            decoder_input = torch.tensor([[next_idx]], device=device)

    # convert back to words to print
    translated_tokens = [tgt_vocab.getword(idx) for idx in predicted_indices]
    translated_sentence = " ".join(translated_tokens)
    return translated_sentence

# make sure to load the vocab otherwise the indices could be wrong from one instance to another
with open("vocEng.pkl", "rb") as f:
    vocEng = pickle.load(f)
with open("vocFra.pkl", "rb") as f:
    vocFra = pickle.load(f)

# same arch is before
encoder = Encoder(len(vocEng), 128, 64).to(device)
decoder = Decoder(len(vocFra), 128, 64).to(device)

encoder.load_state_dict(torch.load("encoder.pt"))
decoder.load_state_dict(torch.load("decoder.pt"))

english_sentence = "Hello how are you" 
french_translation = translate_sentence(encoder, decoder, english_sentence, vocEng, vocFra)
print("Predicted French:", french_translation)

