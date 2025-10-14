import torch
import torch.nn as nn
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):
    # data: X = length x batch x dim
        # Xt = batch x dim
        # Ht = batch x latent
        # Wi = dim x latent
        # Wh = latent x latent
        # bias = latent
        # so ze need two linear layers, one for Wi and one for Wh

    def __init__(self, input_size: int, latent_size: int, output_size: int, decode_activation: nn.Module):
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
        ht = nn.Tanh(self.Wi(x) + self.Wh(h))
        return ht

    def forward(self, x, h):
        # for each elem x in the batch X, we want to one_step it with h
        # output = batch_size x latent_size 
        length = x.size(0)
        batch_size = x.size(1)
        h_output = torch.zeros(length, batch_size, self.latent_size).to(device) 
        for i in range(length): # loop over the x sequence
            h = self.one_step(x[i], h) # apply 
            h_output[i] = h # saves the current h (memory!) used to decode
        return h_output

    def decode(self, h):
        # yt = d(h) = latent x output
        return self.d_a(self.decoder(h))

        

class SampleMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data, length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.classes*self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## transformation de l'index 1d vers une indexation 3d
        ## renvoie une séquence de longueur length et l'id de la station.
        station = i // ((self.nb_timeslots-self.length) * self.nb_days)
        i = i % ((self.nb_timeslots-self.length) * self.nb_days)
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length),station],station

class ForecastMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data,length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## Transformation de l'indexation 1d vers indexation 2d
        ## renvoie x[d,t:t+length-1,:,:], x[d,t+1:t+length,:,:]
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length-1)],self.data[day,(timeslot+1):(timeslot+self.length)]

