import logging

from torch.nn.modules.pooling import MaxPool1d
logging.basicConfig(level=logging.INFO)

import heapq
from pathlib import Path
import gzip

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sentencepiece as spm

from tp7_preprocess import TextDataset

# Utiliser tp7_preprocess pour générer le vocabulaire BPE et
# le jeu de donnée dans un format compact

# --- Configuration

# Taille du vocabulaire
vocab_size = 1000
MAINDIR = Path(__file__).parent

# Chargement du tokenizer

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(f"wp{vocab_size}.model")
ntokens = len(tokenizer)

def loaddata(mode):
    with gzip.open(f"{mode}-{vocab_size}.pth", "rb") as fp:
        return torch.load(fp, weights_only=False)


train = loaddata("train")
TRAIN_BATCHSIZE=500
TEST_BATCHSIZE=500


# --- Chargements des jeux de données train, validation et test

val_size = 1000
test_size = 10000
train_size = len(train) - val_size -test_size
train, val, test = torch.utils.data.random_split(train, [train_size, val_size,test_size])

logging.info("Datasets: train=%d, val=%d, test=%d", train_size, val_size, len(test))
logging.info("Vocabulary size: %d", vocab_size)
train_iter = torch.utils.data.DataLoader(train, batch_size=TRAIN_BATCHSIZE, collate_fn=TextDataset.collate)
val_iter = torch.utils.data.DataLoader(val, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)
test_iter = torch.utils.data.DataLoader(test, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)


#  TODO: 
class SentimentCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, num_channels):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.conv1 = nn.Conv1d(embed_size, num_channels, kernel_size=5, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)

        # uncomment for better perf if cuda is available
        # self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=5, padding=1)
        # self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.global_pool = nn.AdaptiveMaxPool1d(1) # gives specific size no matter input

        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.pool1(torch.relu(self.conv1(x)))
        # x = self.pool2(torch.relu(self.conv2(x)))
        x = self.global_pool(x).squeeze(-1)  
        return self.fc(x)
    
# vocab size -> unique tokens in the tokenizer, so one vector per token
# embed size -> size of the embedding vector for each token, so how much info per token
# num classes -> number of output classes (positive, negative, neutral)
# num channels -> number of filters in the convolutional layer, so how many features to extract
model = SentimentCNN(vocab_size=ntokens, embed_size=128, num_classes=3, num_channels=100)

optim = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

writer = SummaryWriter()

def train_epoch(model, dataloader, optim, criterion, epoch):
    model.train()
    total_loss = 0
    # lets only use part of the train data for speed
    for i, (data, labels) in enumerate(dataloader):
        optim.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()
        total_loss += loss.item()
        if i == 100:
            break
        
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    print(f"Epoch {epoch}, Training Loss: {avg_loss}")
    logging.info(f"Epoch {epoch}, Training Loss: {avg_loss}")

def evaluate(model, dataloader, criterion, epoch, mode='Val'):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, labels in dataloader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / (len(dataloader.dataset))
    writer.add_scalar(f'Loss/{mode}', avg_loss, epoch)
    writer.add_scalar(f'Accuracy/{mode}', accuracy, epoch)
    print(f"Epoch {mode} {epoch}, {mode} Loss: {avg_loss}, {mode} Accuracy: {accuracy}")
    logging.info(f"Epoch {epoch}, {mode} Loss: {avg_loss}, {mode} Accuracy: {accuracy}")

NUM_EPOCHS = 10
for epoch in range(0, NUM_EPOCHS):
    train_epoch(model, train_iter, optim, criterion, epoch)
    evaluate(model, val_iter, criterion, epoch, mode='Val')
evaluate(model, test_iter, criterion, NUM_EPOCHS, mode='Test')
print("Training complete.")

# QUESTION 2
# this gives about 70 ish % accuracy on test and val after 10 epochs, a very tiny subset 
# of train data, trained on cpu and a super simple model
# so increasing model complexity, training time and data used should improve results
# ill have to test that but for sure will increase to well above 80% with those changes
# just need to study that the model doesnt always output the majority class cuz that would be a problem


