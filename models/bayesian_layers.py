import torch
import torch.nn as nn
import torch.distributions as dist

# Define a Bayesian Linear layer
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_mu=0, prior_sigma=0.2):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Parameters for the mean and variance of the weights
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(prior_mu, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-3, -2))  # Start with a small rho
        
        # Parameters for the mean and variance of the bias
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(prior_mu, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-3, -2))
        
        # Prior distribution
        self.prior_weight = dist.Normal(prior_mu, prior_sigma)
        self.prior_bias = dist.Normal(prior_mu, prior_sigma)

    def forward(self, x):
        # Sample weights and biases using the reparameterization trick
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))  # Softplus to ensure positivity
        weight_eps = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + weight_sigma * weight_eps
        
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        bias_eps = torch.randn_like(self.bias_mu)
        bias = self.bias_mu + bias_sigma * bias_eps
        
        # Linear operation with sampled weights and biases
        return torch.nn.functional.linear(x, weight, bias)

    def kl_divergence(self):
        # Calculate KL divergence for weights and biases with respect to the prior
        weight_kl = dist.kl_divergence(dist.Normal(self.weight_mu, torch.log1p(torch.exp(self.weight_rho))), self.prior_weight).sum()
        bias_kl = dist.kl_divergence(dist.Normal(self.bias_mu, torch.log1p(torch.exp(self.bias_rho))), self.prior_bias).sum()
        return weight_kl + bias_kl
    

class BayesianConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, prior_mu=0, prior_sigma=0.1):
        super(BayesianConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Parameters for weights and biases
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size).normal_(prior_mu, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size).uniform_(-3, -2))
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels).normal_(prior_mu, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_channels).uniform_(-3, -2))
        
        # Prior distributions
        self.prior_weight = dist.Normal(prior_mu, prior_sigma)
        self.prior_bias = dist.Normal(prior_mu, prior_sigma)

    def forward(self, x):
        # Reparameterization trick
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight_eps = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + weight_sigma * weight_eps
        
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        bias_eps = torch.randn_like(self.bias_mu)
        bias = self.bias_mu + bias_sigma * bias_eps
        
        return F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding)

    def kl_divergence(self):
        # KL divergence for weights and biases
        weight_kl = dist.kl_divergence(dist.Normal(self.weight_mu, torch.log1p(torch.exp(self.weight_rho))), self.prior_weight).sum()
        bias_kl = dist.kl_divergence(dist.Normal(self.bias_mu, torch.log1p(torch.exp(self.bias_rho))), self.prior_bias).sum()
        return weight_kl + bias_kl