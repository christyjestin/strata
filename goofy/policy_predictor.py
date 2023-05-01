import torch
from torch import nn

from model_constants import *
from conditional_analyzer import ConditionalAnalyzer

class PolicyPredictor(nn.Module):
    def __init__(self, conditional_analyzer: ConditionalAnalyzer, num_distribution_parameters = 8):
        super().__init__()
        self.state_dim = STATE_DIM
        self.strategy_dim = STRATEGY_DIM
        self.conditional_analyzer = conditional_analyzer
        self.input_dim = conditional_analyzer.output_dim
        self.num_distribution_parameters = num_distribution_parameters
        # our movement and rotation policy should be dependent on the action we're taking e.g. shield vs attack
        # so we scale the number of distribution parameters produced by the number of actions
        self.output_dim = NUM_ACTIONS * (1 + self.num_distribution_parameters)
        self.output_split = [NUM_ACTIONS, NUM_ACTIONS * self.num_distribution_parameters]
        assert sum(self.output_split) == self.output_dim, "The split must sum to the output dimension"

        # TODO: figure out whether to use separate normalizing flows and policy predictors
        # for adversary vs hero 
        self.MLP = nn.Sequential([
            nn.Linear(self.input_dim, 80),
            nn.ReLU(),
            nn.Linear(80, 70),
            nn.ReLU(),
            nn.Linear(70, 60),
            nn.ReLU(),
            nn.Linear(60, self.output_dim)
        ])

    def forward(self, states, strategies, in_search_mode):
        n = states.shape[0] # batch size
        analysis = self.conditional_analyzer(states, strategies, in_search_mode)
        mlp_output = self.MLP(analysis)
        logits, parameters = torch.split(mlp_output, self.output_split, dim = 1)

        assert logits.shape == (n, NUM_ACTIONS)
        parameters = parameters.reshape(n, NUM_ACTIONS, self.num_distribution_parameters)
        return SearchPolicy(logits, parameters)