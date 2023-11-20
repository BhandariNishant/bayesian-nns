'''
Based on https://colindcarroll.com/2019/04/11/hamiltonian-monte-carlo-from-scratch/
and 
https://colindcarroll.com/2019/04/06/exercises-in-automatic-differentiation-using-autograd-and-jax/
'''
# import numpy as np
import pickle
import scipy.stats as st
from autograd import grad
import autograd.numpy as np
import matplotlib.pyplot as plt

'''
Very weird but allows for relative imports.
'''
import sys
sys.path.append("..")

from data import NoisyXOR
from tqdm import tqdm

np.random.seed(14)

def sigmoid(x):
    '''
    Autograd wrapper doesn't have sigmoid, so we define it here.
    '''
    return 1/(1 + np.exp(-x))

def weight_unpack(params, hid_size=4):
    '''
    Unpacks the weights and biases from the numpy array.
    '''
    w1 = params[:2*hid_size].reshape(2, hid_size)
    b1 = params[2*hid_size:3*hid_size].reshape(1, hid_size)
    w2 = params[3*hid_size:4*hid_size].reshape(hid_size, 1)
    b2 = params[-1].reshape(1, 1)

    return w1, b1, w2, b2

def evaluate(params, data : NoisyXOR, eval = False):
    '''
    Evaluates the accuracy of the model.
    Different dataset every time.
    '''
    w1, b1, w2, b2 = weight_unpack(params, hid_size=4)

    if eval:
        data = NoisyXOR(100)

    logit = np.dot(np.maximum(np.dot(data.X.numpy(), w1) + b1, 0), w2) + b2
    op = sigmoid(logit)

    op[op > 0.5] = 1
    op[op < 0.5] = 0
    
    return np.mean((op == data.y.numpy()))*100

def model(params, data : NoisyXOR):
    '''
    Computes the negative log likelihood of the model.
    '''
    w1, b1, w2, b2 = weight_unpack(params, hid_size=4)

    logit = np.dot(np.maximum(np.dot(data.X.numpy(), w1) + b1, 0), w2) + b2
    prob = sigmoid(logit)

    prob_plus = np.multiply(prob.squeeze(1), (data.y.squeeze(1).numpy() == 1.))
    prob_neg = np.multiply(1-prob.squeeze(1), (data.y.squeeze(1).numpy() == 0.))
    prob = prob_plus + prob_neg

    neg_log_prob = -np.sum(np.log(prob + 1e-10))

    return neg_log_prob

def negative_log_prob(prior_mu, prior_sigma, model, data : NoisyXOR):
    '''
    Computes the negative log posterior.
    '''

    def posterior(q):
        numerator = np.exp((-(q-prior_mu)**2)/(2*(prior_sigma**2)))
        denominator = np.sqrt(2*np.pi)*prior_sigma

        neg_log_likelihood = model(q, data)
        
        return -np.sum(np.log(numerator/denominator)) + neg_log_likelihood
    
    return posterior

def hamiltonian_monte_carlo(data, model=model, n_samples=50, negative_log_prob=negative_log_prob, initial_position=None, path_len=1, step_size=1e-3):
    """Run Hamiltonian Monte Carlo sampling.

    Parameters
    ----------
    data : NoisyXOR
        The dataset to train on.
    args : argparse.ArgumentParser
        The arguments passed to the script.
    model : callable
        The model to sample from. This takes in a single argument (the parameters of the model) and returns the log likelihood.
    n_samples : int
        Number of samples to return
    negative_log_prob : callable
        The negative log probability to sample from
    initial_position : np.array
        A place to start sampling from.
    path_len : float
        How long each integration path is. Smaller is faster and more correlated.
    step_size : float
        How long each integration step is. Smaller is slower and more accurate.

    Returns
    -------
    np.array
        Array of length `n_samples`.
    """
    # store
    neg_prob = negative_log_prob(np.zeros_like(initial_position), np.ones_like(initial_position), model, data)

    # autograd magic
    dVdq = grad(neg_prob)

    # collect all our samples in a list
    samples = [initial_position]

    # Keep a single object for momentum resampling
    momentum = st.norm(0, 1)

    # If initial_position is a 10d vector and n_samples is 100, we want
    # 100 x 10 momentum draws. We can do this in one call to momentum.rvs, and
    # iterate over rows
    size = (n_samples,) + initial_position.shape[:1]
    train_acc = [0]
    val_acc = [0]
    with tqdm(momentum.rvs(size=size)) as pbar: #desc=f"HMC, training accuracy : {train_acc[-1] :.4f}"
        for p0 in pbar:
            # Integrate over our path to get a new position and momentum
            q_new, p_new = leapfrog(
                samples[-1],
                p0,
                dVdq,
                path_len=path_len,
                step_size=step_size,
            )

            # Check Metropolis acceptance criterion
            start_log_p = neg_prob(samples[-1]) - np.sum(momentum.logpdf(p0)) #negative_log_prob(samples[-1])
            new_log_p = neg_prob(q_new) - np.sum(momentum.logpdf(p_new)) #negative_log_prob(q_new)
            if np.log(np.random.randn()) < start_log_p - new_log_p:
                samples.append(q_new)
            else:
                samples.append(np.copy(samples[-1]))

            train_acc.append(evaluate(samples[-1], data))
            val_acc.append(evaluate(samples[-1], data, eval=True))
            pbar.set_postfix(train_accuracy=train_acc[-1], val_accuracy=val_acc[-1])

    # Plotting
    #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 16))
    plt.plot(val_acc, label='Validation accuracy')
    plt.plot(train_acc, label='Training accuracy')
    plt.legend()
    #ax2.plot(samples[:][-1], label='Final layer bias plot')
    plt.savefig('numpy_hmc/hmc.png')
    plt.close('all')

    return np.array(samples[1:])

def leapfrog(q, p, dVdq, path_len, step_size):
    """Leapfrog integrator for Hamiltonian Monte Carlo.

    Parameters
    ----------
    q : np.floatX
        Initial position
    p : np.floatX
        Initial momentum
    dVdq : callable
        Gradient of the velocity
    path_len : float
        How long to integrate for
    step_size : float
        How long each integration step should be

    Returns
    -------
    q, p : np.floatX, np.floatX
        New position and momentum
    """
    q, p = np.copy(q), np.copy(p)

    p -= step_size * dVdq(q) / 2  # half step
    for _ in range(int(path_len/step_size) - 1): #/step_size
        q += step_size * p  # whole step
        p -= step_size * dVdq(q)  # whole step
    q += step_size * p  # whole step
    p -= step_size * dVdq(q) / 2  # half step

    # momentum flip at end
    return q, -p

if __name__ == '__main__':
    
    #NOTE to self : run as `python -m numpy_hmc.hmc`
    #Allows for relative imports

    total_params = 16
    num_train_samples = 1000
    data = NoisyXOR(num_train_samples)
    init = np.random.npr.normal(np.zeros(total_params), np.ones(total_params), total_params)
    samples = hamiltonian_monte_carlo(data=data, initial_position=init)
    with open(f'numpy_hmc/weights/params_{total_params}_num_samples_{num_train_samples}.pickle', 'wb') as handle:
        pickle.dump(samples[-1], handle, protocol=pickle.HIGHEST_PROTOCOL)