import torch
from torch import nn, Tensor

from model_constants import *
from conditional_analyzer import ConditionalAnalyzer

class PolicyPredictor(nn.Module):
    def __init__(self, conditional_analyzer: ConditionalAnalyzer, num_mixture_components: int) -> None:
        super().__init__()
        self.conditional_analyzer = conditional_analyzer
        self.input_dim = conditional_analyzer.output_dim
        self.num_mixture_components = num_mixture_components
        self.num_distribution_parameters = num_mixture_components * NUM_PARAMETERS_PER_MOVEMENT_DIST

        # our movement and rotation policy should be dependent on the action we're taking e.g. shield vs attack
        # so we scale the number of distribution parameters produced by the number of actions
        # +1 for the actual action logits themselves
        self.output_dim = NUM_ACTIONS * (1 + self.num_distribution_parameters)
        self.output_split = [NUM_ACTIONS, NUM_ACTIONS * self.num_distribution_parameters]
        assert sum(self.output_split) == self.output_dim, "The split must sum to the output dimension"

        self.MLP = nn.Sequential([
            nn.Linear(self.input_dim, 90),
            nn.ReLU(),
            nn.Linear(90, 100),
            nn.ReLU(),
            nn.Linear(100, 110),
            nn.ReLU(),
            nn.Linear(110, self.output_dim)
        ])

        self.relu = nn.ReLU() # extra ReLU to generate parameters for Beta distributions

    def forward(self, states: Tensor, strategies: Tensor, mode: str) -> ActionPolicy:
        n = states.shape[0] # batch size
        analysis = self.conditional_analyzer(states, strategies, mode)
        mlp_output = self.MLP(analysis)
        logits, parameters = torch.split(mlp_output, self.output_split, dim = 1)

        assert logits.shape == (n, NUM_ACTIONS)
        parameters = parameters.reshape(n, NUM_ACTIONS, self.num_mixture_components, NUM_PARAMETERS_PER_MOVEMENT_DIST)

        # beta parameters must be positive
        beta_parameters = self.relu(parameters) + BETA_EPSILON
        return ActionPolicy(logits, beta_parameters)