import torch
from torch.utils.data import Dataset
import numpy as np
import os
from utils import Modulation, ofdm_simulate

class OFDMDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode

        # parameters
        self.K = 64
        self.CP = self.K // 4
        self.P = config.Pilots
        self.mu = 2
        self.allCarriers = np.arange(self.K) #indice

        if self.P < self.K:
            self.pilotCarriers = self.allCarriers[::self.K // self.P]
            self.dataCarriers = np.delete(self.allCarriers, self.pilotCarriers)
        else:
            self.pilotCarriers = self.allCarriers
            self.dataCarriers = []

        self.payloadBits_per_OFDM = self.K * self.mu

        # pilots
        Pilot_file_name = 'Pilot_' + str(self.P)
        if os.path.isfile(Pilot_file_name):
            self.pilot_bits = np.loadtxt(Pilot_file_name, delimiter=',')
        else:
            self.pilot_bits = np.random.binomial(1, 0.5, size=(self.K * self.mu))
            np.savetxt(Pilot_file_name, self.pilot_bits, delimiter=',')

        self.pilotValue = Modulation(self.pilot_bits, self.mu)


