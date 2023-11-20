import argparse
import torch
from pyro_hmc.hmc import train
from pyro_hmc.evaluate import evaluate
import matplotlib.pyplot as plt
import pickle

def experiment():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples",type = int, help = "number of samples")
    parser.add_argument("--step_size",type = float, help = "step size")
    parser.add_argument("--num_steps",type = int, help = "number of steps")
    parser.add_argument("--warmup_steps",type = int, help = "number of warmup steps")
    parser.add_argument("--num_data",type = int, help = "number of data")
    args = parser.parse_args()

    # Default values for hyperparameters
    traj_len = 1
    num_samples = 50
    step_size = 0.01
    num_steps = 1
    warmup_steps = 10
    num_data = 1000

    # Update values if provided
    if args.num_samples:
        num_samples = args.num_samples
    if args.step_size:
        step_size = args.step_size
    if args.num_steps:
        num_steps = args.num_steps
    if args.warmup_steps:
        warmup_steps = args.warmup_steps

    # Print the values
    print("num_samples ", num_samples)
    print("step_size ", step_size)
    print("num_steps ", num_steps)
    print("warmup_steps ", warmup_steps)
    print("num_data ", num_data)

    # Train Model
    training_data = train(num_data, num_samples, step_size, warmup_steps, traj_len)

    # Plot the training data
    y = training_data.y.numpy().squeeze()
    plt.plot(training_data.X.numpy()[:, 0][y == 1], training_data.X.numpy()[:, 1][y == 1], 'o', label='+')
    plt.plot(training_data.X.numpy()[:, 0][y == 0], training_data.X.numpy()[:, 1][y == 0], 'x', label='-')
    plt.savefig('pyro_hmc/data.png')
    plt.close('all')

    # Evaluate Model
    train_acc = []
    val_acc = []
    for i in range (num_samples):
        print("Iteration", i+1, end = ": ")

        train_acc.append(evaluate(data=training_data,iter=i, val=False))
        print("Training accuracy =", train_acc[-1], end = ", ")

        val_acc.append(evaluate(data=training_data, iter=i,val=True))
        print("Validation accuracy =", val_acc[-1])

    # Plot the accuracies
    plt.plot(val_acc, label='Validation accuracy')
    plt.plot(train_acc, label='Training accuracy')
    plt.legend()
    plt.savefig("pyro_hmc/accuracy.png")
    plt.close('all')

    # Extract final weights and biases after training is complete
    with open('posterior_saved.pickle', 'rb') as handle:
        posterior_samples = pickle.load(handle)

    weight0 = posterior_samples["weight_0"][-1]
    bias0=posterior_samples["bias_0"][-1]
    weight1=posterior_samples["weight_1"][-1]
    bias1=posterior_samples["bias_1"][-1]

    # Make final predictions
    op = torch.relu(training_data.X @ weight0 + bias0)
    op = torch.sigmoid(op @ weight1 + bias1)
    op[op >= 0.5] = 1
    op[op < 0.5] = 0 
    op=op.squeeze()

    # Plot the final predictions
    plt.plot(training_data.X.numpy()[:, 0][op == 1], training_data.X.numpy()[:, 1][op == 1], 'o', label='+')
    plt.plot(training_data.X.numpy()[:, 0][op == 0], training_data.X.numpy()[:, 1][op == 0], 'x', label='-')
    plt.savefig('pyro_hmc/predictions.png')
    plt.close('all')


if __name__ == '__main__':
    experiment()