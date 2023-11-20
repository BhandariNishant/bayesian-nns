'''
Dataset file.
'''
import torch
from torch.utils.data.dataset import Dataset

class NoisyXOR(Dataset):

    def __init__(self, num_samples=100):

        super(NoisyXOR, self).__init__()

        self.num_samples = num_samples
        x = torch.rand((num_samples, 2))
        y = torch.zeros((num_samples, 1))

        # assign labels according to XOR condition
        y[(x[:, 0] < 0.5) & (x[:, 1] > 0.5)] = 1
        y[(x[:, 0] > 0.5) & (x[:, 1] < 0.5)] = 1

        x[x > 0.5] += 0.08  # move the class blocks slightly further from each other
        x += torch.rand(
            (num_samples, 2)
        ) / 8  # add some noise at the edges of each class block

        self.X = x
        self.y = y

    def __len__(self):
        
        return self.num_samples

    def __getitem__(self, index):

        return (self.X[index], self.y[index])