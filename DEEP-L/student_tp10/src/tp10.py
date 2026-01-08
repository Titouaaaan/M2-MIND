from utils import random_walk,construct_graph
import math
from tqdm import tqdm
import networkx as nx
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random
import torch
from torch.utils.tensorboard import SummaryWriter

import time
import logging

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

class TripletDataset(torch.utils.data.Dataset):
    ''' Returns a random triplet (anchor, positive, negative) for training '''
    def __init__(self, graph, k_neighbors):
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.k_neighbors = k_neighbors
        self.node2idx = {n: i for i, n in enumerate(self.nodes)} # need this to return proper id
    
    def __len__(self):
        return len(self.graph.nodes())
    
    def __getitem__(self, idx):
        ''' returns a random triplet (anchor, positive, negative) '''
        return self.getTriplet(idx)

    def getTriplet(self, idx):
        anchor = self.nodes[idx] # get node id from index

        # .neighbors returns nodes that are directly connected to our anchor node, so distance == 1
        neighbors = list(self.graph.neighbors(anchor)) # get neighbors of the anchor node
        if not neighbors:
            #print("No neighbors for node", anchor)
            # if there are no neighbors, return a random triplet
            return self.__getitem__(random.randint(0, len(self.nodes) - 1))

        positive = random.choice(neighbors) # get a random neighbor

        lengths = nx.single_source_shortest_path_length(self.graph, anchor, cutoff=self.k_neighbors)
        far_neighbors = {n for n, d in lengths.items() if 1 < d <= self.k_neighbors}  
        if not far_neighbors:
            #print("No far neighbors for node", anchor)
            # if there are no far neighbors, return a random triplet
            return self.__getitem__(random.randint(0, len(self.nodes) - 1))
        negative = random.choice(list(far_neighbors)) # get a random far neighbor  

        return (
            self.node2idx[anchor],
            self.node2idx[positive],
            self.node2idx[negative],
        )

if __name__=="__main__":
    PATH = "data/ml-latest-small/"
    logging.info("Constructing graph")
    movies_graph, movies = construct_graph(PATH + "movies.csv", PATH + "ratings.csv")
    logging.info("Sampling walks")
    walks = random_walk(movies_graph,5,10,1,1)
    nodes2id = dict(zip(movies_graph.nodes(),range(len(movies_graph.nodes()))))
    id2nodes = list(movies_graph.nodes())
    id2title = [movies[movies.movieId==idx].iloc[0].title for idx in id2nodes]
    
    dataset = TripletDataset(movies_graph, k_neighbors=3)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    embedder = nn.Embedding(len(movies_graph.nodes()), 64).to(device)

    # Training loop to learn embeddings
    logging.info("Starting training loop")
    n_epochs = 20
    optimizer = torch.optim.Adam(embedder.parameters(), lr=0.001)
    margin = 1.0
    criterion = nn.TripletMarginLoss(margin=1.0)
    for epoch in range(n_epochs):
        logging.info(f"Epoch {epoch+1}/{n_epochs}")
        avg_loss = 0.0
        embedder.train()
        # Iterate over the DataLoader
        for anchor, positive, negative in loader:

            # lets calculate the triplet loss
            anchor_emb = embedder(anchor.to(device))
            positive_emb = embedder(positive.to(device))
            negative_emb = embedder(negative.to(device))

            # normalize
            # i removed this because normalizing doesnt help in this case 
            # works way better without it
            """ anchor_emb = F.normalize(anchor_emb, p=2, dim=1)
            positive_emb = F.normalize(positive_emb, p=2, dim=1)
            negative_emb = F.normalize(negative_emb, p=2, dim=1) """

            # this is the manual version but lets use the pytorch verison for stability just in case
            # distance_pos = torch.norm(anchor_emb - positive_emb, p=2, dim=1)
            # distance_neg = torch.norm(anchor_emb - negative_emb, p=2, dim=1)
            # loss = torch.max(torch.tensor(0.0, device=device), distance_pos - distance_neg + margin)
            loss = criterion(anchor_emb, positive_emb, negative_emb)

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            avg_loss += loss.mean().item()
        logging.info(f"Epoch {epoch+1} completed")
        logging.info(f"Average Loss: {avg_loss/len(loader)}")
    logging.info("Training completed")

    # visualize embeddings with t-SNE in tensorboard
    embedder.eval()

    with torch.no_grad():
        embeddings = embedder.weight.detach().cpu().numpy()
    metadata = [id2title[i] for i in range(len(id2title))]
    writer = SummaryWriter("runs/movies_tsne")

    writer.add_embedding(
        mat=embeddings,
        metadata=metadata,
        tag="Movie embeddings"
    )

    writer.close()