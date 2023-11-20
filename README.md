# PoPL (Principles of Programming Languages), CS F301 Course Project

## Title : Comparative Study between PyRo and other DL frameworks
## Members : Karan Bania (2021A7PS2582G), Nishant Bhandari (2021A7PS2046G)

### Software used (major parts): 
1. <a href="https://github.com/HIPSautograd">`autograd` library</a>
2. <a href="https://pyro.ai/">the `PyRo` framework</a>
3. <a href="https://numpy.org/">`numpy` library</a>

### PoPL aspects: (<b>Bold</b> library is the better one)
### (ease-of-use) Defining the prior (<b>PyRo</b>)
In numpy, the prior must be hard-coded at all places <br>
```
init = np.random.npr.normal(np.zeros(total_params), np.ones(total_params), total_params)
```
In Pyro, <br>
```
code
```
### (ease-of-use) Using the actual Hamiltonian Monte Carlo algorithm (<b>PyRo</b>)
In numpy, the code is 100+ lines <br>
In Pyro, the code is 3 lines <br>
```
code
```
<br>

### (ease-of-use) Changing Model architecture (<b>Pyro<b>)
In Numpy, the architecture must be hard-coded and is awkward to change
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
In Pyro,
```
code
```

### (ease-of-use) Plotting results (<b>Numpy</b>)
In Numpy, we have access to intermediate configurations of the model easily, few lines of code to plot stuff,
```
plt.plot(val_acc, label='Validation accuracy')
plt.plot(train_acc, label='Training accuracy')
plt.legend()
```
In Pyro, we had to make artificial changes and sample first evaluate later to plot
```
code
```

### (reliability) Default implementation vs Unsafe Self Implementation (<b>Pyro</b>)
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
Insert bernoulli code here
```

#### Note : there is no `/tests` folder because results have been compiled in the `/doc` folder.