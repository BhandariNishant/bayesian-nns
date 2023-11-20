# PoPL (Principles of Programming Languages), CS F301 Course Project

## Title : Comparative Study between PyRo and other DL frameworks
## Members : Karan Bania (2021A7PS2582G), Nishant Bhandari (2021A7PS2046G)

### Software used (major parts): 
1. <a href="https://github.com/HIPSautograd">`autograd` library</a>
2. <a href="https://pyro.ai/">the `PyRo` framework</a>
3. <a href="https://numpy.org/">`numpy` library</a>
4. <a href="https://pytorch.org/">`PyTorch` library</a>

### PoPL aspects: (<u>Underlined</u> library is the better one)
### (ease-of-use) Defining the prior (<u>PyRo</u>/Numpy)
In numpy, the prior must be hard-coded at all places <br>
```
init = np.random.npr.normal(np.zeros(total_params), np.ones(total_params), total_params)
```
In Pyro, the prior can be easily changed by just changing the dist.Normal <br>
```
 weights.append(pyro.sample(f'weight_0', dist.Normal(torch.zeros(input_size, output_size), torch.ones(input_size, output_size))))
```
### (ease-of-use) Using the actual Hamiltonian Monte Carlo algorithm (<u>PyRo</u>/Numpy)
In numpy, the code is 100+ lines <br>
In PyRo, the code is 3 lines <br>
```
hmc_kernel = HMC(model, step_size=step_size, trajectory_length=traj_len)
mcmc_run = MCMC(hmc_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
mcmc_run.run(data, input_size, hidden_sizes, output_size)
```

### (ease-of-use) Changing Model architecture (<u>Pyro</u>/Numpy)
In numpy, the architecture must be hard-coded and is awkward to change
```
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
```
In Pyro, the layer dimensions can be changed by changing the size as well as additional layers can be easily added by appending to hidden_sizes
```
input_size = 2
hidden_sizes = [4]  # Specify the sizes of hidden layers
output_size = 1
```

### (ease-of-use) Plotting results (Pyro/<u>Numpy</u>)
In Numpy, we have access to intermediate configurations of the model easily, few lines of code to plot stuff,
```
plt.plot(val_acc, label='Validation accuracy')
plt.plot(train_acc, label='Training accuracy')
plt.legend()
```
In Pyro, we had to make artificial changes and sample first evaluate later to plot
```
for i in range (num_samples):
```

### (reliability) Default implementation vs Unsafe Self Implementation (<u>Pyro</u>/Numpy)
In Numpy, the sigmoid implementation can lead to overflows
```
def sigmoid(x):
    '''
    Autograd wrapper doesn't have sigmoid, so we define it here.
    '''
    return 1/(1 + np.exp(-x))
```
In Pyro, all this is taken care of internally<br>
```
pyro.sample("obs", dist.Bernoulli(logits=output), obs=data.y)
```

### To reproduce results:
#### For numpy run - 
```
cd code-orig
python -m numpy_hmc.hmc
```
#### For pyro run -
```
cd code-orig
python -m pyro_hmc.experiment
```

### Possible future directions -
#### It would be compelling to compare these two paradigms on even more DL algorithms like CNNs and RNNs. Although scaling MCMC to this would be a problem.

#### Note : there is no `/tests` folder because results have been compiled in the `/doc` folder,there is also no `/code-external` folder because we have mentioned all our references in individual `.py` files.