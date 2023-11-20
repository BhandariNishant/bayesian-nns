import torch
import pyro
import pyro.distributions as dist
torch.manual_seed(1)

def model(data, input_size, hidden_sizes, output_size):
    # Prior distributions for weights and biases
    weights = []
    biases = []

    # Sample the weights and biases
    if(len(hidden_sizes)==0):
        weights.append(pyro.sample(f'weight_0', dist.Normal(torch.zeros(input_size, output_size), torch.ones(input_size, output_size))))
        biases.append(pyro.sample(f'bias_0', dist.Normal(torch.zeros(input_size), torch.ones(input_size))))

        x = data.X
        x = torch.matmul(x, weights[0]) + biases[0]
        x = torch.relu(x)

    else:
        weights.append(pyro.sample(f'weight_0', dist.Normal(torch.zeros(input_size, hidden_sizes[0]), torch.ones(input_size, hidden_sizes[0]))))
        biases.append(pyro.sample(f'bias_0', dist.Normal(torch.zeros(hidden_sizes[0]), torch.ones(hidden_sizes[0]))))

        for i in range(1, len(hidden_sizes)):
            weights.append(pyro.sample(f'weight_{i}', dist.Normal(torch.zeros(hidden_sizes[i-1], hidden_sizes[i]), torch.ones(hidden_sizes[i-1], hidden_sizes[i]))))
            biases.append(pyro.sample(f'bias_{i}', dist.Normal(torch.zeros(hidden_sizes[i]), 10*torch.ones(hidden_sizes[i]))))

        weights.append(pyro.sample(f'weight_{len(hidden_sizes)}', dist.Normal(torch.zeros(hidden_sizes[len(hidden_sizes)-1], output_size), torch.ones(hidden_sizes[len(hidden_sizes)-1], output_size))))
        biases.append(pyro.sample(f'bias_{len(hidden_sizes)}', dist.Normal(torch.zeros(output_size), torch.ones(output_size))))

        # Define the neural network structure
        x = data.X
        for i in range(len(hidden_sizes)):
            x = torch.matmul(x, weights[i]) + biases[i]
            x = torch.relu(x)
        
    output = torch.matmul(x, weights[-1]) + biases[-1]

    # Observe the data
    pyro.sample("obs", dist.Bernoulli(logits=output), obs=data.y)