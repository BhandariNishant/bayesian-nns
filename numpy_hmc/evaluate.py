import torch
import pickle
from autograd import numpy as np
from numpy_hmc.hmc import weight_unpack, sigmoid

'''
Very weird but allows for relative imports.
'''
import sys
sys.path.append("..")

from data import NoisyXOR

def evaluate():

    with open('numpy_hmc/weights/params_16_num_samples_1000.pickle', 'rb') as handle:
        weights = pickle.load(handle)

    w1, b1, w2, b2 = weight_unpack(weights, hid_size=4)

    data = NoisyXOR(100)

    logit = np.dot(np.maximum(np.dot(data.X.numpy(), w1) + b1, 0), w2) + b2
    op = sigmoid(logit)

    op[op > 0.5] = 1
    op[op < 0.5] = 0
    
    return np.mean((op == data.y.numpy()))*100

if __name__ == '__main__':

    #NOTE to self : run as `python -m numpy_hmc.evaluate`
    #Allows for relative imports
    
    torch.manual_seed(1)
    acc1 = evaluate()
    torch.manual_seed(2)
    acc2 = evaluate()
    torch.manual_seed(3)
    acc3 = evaluate()

    print('Accuracy 1: ', acc1)
    print('Accuracy 2: ', acc2)
    print('Accuracy 3: ', acc3)
    print('Mean accuracy accross 3 seeds: ', np.mean([acc1, acc2, acc3]))