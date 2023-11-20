from pyro_hmc.MLP import model
from pyro.infer import HMC, MCMC
import pickle

import sys
sys.path.append("..")

from data import NoisyXOR

def train(num_data, num_samples, step_size, warmup_steps, traj_len):

    # Generate the training data
    data = NoisyXOR(num_samples=num_data)

    # Define the architecture of the Bayesian neural network
    input_size = 2
    hidden_sizes = [4]  # Specify the sizes of hidden layers
    output_size = 1

    # Run HMC to sample from the posterior distribution
    hmc_kernel = HMC(model, step_size=step_size, trajectory_length=traj_len)
    mcmc_run = MCMC(hmc_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_run.run(data, input_size, hidden_sizes, output_size)

    # Save the posterior samples
    posterior_samples = mcmc_run.get_samples()
    with open('posterior_saved.pickle', 'wb') as handle:
        pickle.dump(posterior_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return data