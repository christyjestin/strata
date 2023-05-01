import torch
from torch import nn

from model_constants import *
from conditional_analyzer import ConditionalAnalyzer

class ValuePredictor(nn.Module):
    def __init__(self, conditional_analyzer: ConditionalAnalyzer, time_embedding_dim):
        super().__init__()
        self.state_dim = STATE_DIM
        self.strategy_dim = STRATEGY_DIM
        self.conditional_analyzer = conditional_analyzer
        self.input_dim = conditional_analyzer.output_dim
        self.time_embedding_dim = time_embedding_dim
        self.time_horizon = TIME_HORIZON

        self.time_embedding = nn.Embedding(self.time_horizon, self.time_embedding_dim)

        self.MLP = nn.Sequential([
            nn.Linear(self.input_dim, 80),
            nn.ReLU(),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 11)
        ])

    def forward(self, states, strategies, remaining_time_horizon, in_search_mode):
        assert in_search_mode or states.shape[0] == remaining_time_horizon.shape[0], \
                                        "states and time horizon must align in backprop mode"

        n = states.shape[0] # batch size
        if in_search_mode:
            assert isinstance(remaining_time_horizon, int), "remaining time horizon must be an integer in search mode"
            remaining_time_horizon = torch.full((n,), remaining_time_horizon)

        analysis = self.conditional_analyzer(states, strategies, in_search_mode)
        embedded_time = self.time_embedding(remaining_time_horizon)
        x = torch.cat((analysis, embedded_time), dim = 1)
        mlp_outputs = self.MLP(x)

        # TODO: talk to Jacob and consider making this 1 by 1 plus 1 for dims
        x1, x2, b = torch.split(mlp_outputs, [5, 5, 1], dim = 1)
        # matrix implementation of x1 @ x2.T plus a bias where x1 and x2 are row vectors
        # the idea here is to allow scaling since value will (to some extent) scale with time
        output = torch.sum(x1 * x2, dim = 1, keepdim = True) + b
        return output

    def encode_time(self, remaining_time_horizon):
        assert len(remaining_time_horizon.shape) == 1, "time horizon must be 1d for proper indexing into embeddings"
        return self.time_embedding(remaining_time_horizon - 1) # subtract off one to zero index