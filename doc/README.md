# PoPL (Principles of Programming Languages), CS F301 Course Project

## Title : Comparative Study between PyRo and other DL frameworks
## Members : Karan Bania (2021A7PS2582G), Nishant Bhandari (2021A7PS2046G)

### Software used (major parts): 
1. <a href="https://github.com/HIPSautograd">`autograd` library</a>
2. <a href="https://pyro.ai/">the `PyRo` framework</a>
3. <a href="https://numpy.org/">`numpy` library</a>
4. <a href="https://pytorch.org/">`PyTorch` library</a>

### PoPL aspects: (<b>Bold</b> library is the better one)
1. ease-of-use, <br>
    a. defining the prior (<b>Pyro</b>) <br>
        In Numpy,<br>
        ```
        init = np.random.npr.normal(np.zeros(total_params), np.ones(total_params), total_params)
        ```
        <br>
        In Pyro,
        <br>
    b. using the actual Hamiltonian Monte Carlo algorithm (<b>Pyro</b>)<br>
        In Numpy, the code is 100+ lines <br>
        In Pyro, 
        <br>
    c. changing the architecture (<b>Pyro</b>) <br>
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
        In Pyro, <br>
    d. plotting results (<b>Numpy</b>) <br>
        In Numpy, we have access to intermediate configurations of the model easily, few lines of code to plot stuff,
        ```
        plt.plot(val_acc, label='Validation accuracy')
        plt.plot(train_acc, label='Training accuracy')
        plt.legend()
        ```
        In Pyro, <br>
    2. reliability, <br>
    a. default implementation vs unsafe self implementation (<b>Pyro</b>) <br>
        In Numpy, the sigmoid implementation can lead to overflows<br>
        ```
        def sigmoid(x):
            '''
            Autograd wrapper doesn't have sigmoid, so we define it here.
            '''
            return 1/(1 + np.exp(-x))
        ```
        In Pyro, <br>
    b. 