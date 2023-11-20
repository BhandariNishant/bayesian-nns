import pickle
import torch
torch.manual_seed(1)

import sys
sys.path.append("..")

from data import NoisyXOR

def evaluate(iter, val, data):
    # Load the posterior samples from the saved pickle file
    with open('posterior_saved.pickle', 'rb') as handle:
        posterior_samples = pickle.load(handle)

    # Extract the weights and biases from the posterior samples
    weight0 = posterior_samples["weight_0"][iter]
    bias0=posterior_samples["bias_0"][iter]
    weight1=posterior_samples["weight_1"][iter]
    bias1=posterior_samples["bias_1"][iter]
    
    # If Validation then generate new data or use existing training data
    if(val):
        data = NoisyXOR(1000)

    # Make predictions
    op = torch.relu(data.X @ weight0 + bias0)
    op = torch.sigmoid(op @ weight1 + bias1)
    
    op[op >= 0.5] = 1
    op[op < 0.5] = 0 

    # Return the accuracy
    return(torch.mean((op == data.y).to(torch.float32)) * 100)