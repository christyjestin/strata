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

        self.time_embedding = nn.Embedding(self.time_horizon - 1, self.time_embedding_dim)

        self.MLP = nn.Sequential([
            nn.Linear(self.input_dim, 80),
            nn.ReLU(),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 11)
        ])

    def forward(self, states, strategies, mode, remaining_time_horizon = None):
        assert mode in [SEARCH_MODE, BACKPROP_MODE]
        n = states.shape[0] # batch size

        analysis = self.conditional_analyzer(states, strategies, mode)
        # construct time embedding
        if mode == SEARCH_MODE:
            assert isinstance(remaining_time_horizon, int), "remaining time horizon must be an integer in search mode"
            embedded_time = self.embed_time(torch.full((n,), remaining_time_horizon))
        else:
            assert remaining_time_horizon is None, "the predictor will construct the remaining time horizon in backprop mode"
            # repeat time horizon - 1 times unless there aren't enough states left
            repeats = torch.clip(torch.arange(n, 0, -1), max = self.time_horizon - 1)
            # expand analysis to project over every possible remaining time horizon
            analysis = torch.repeat_interleave(analysis, repeats, dim = 0)
            # construct corresponding time horizons in ascending order
            remaining_time_horizon = torch.concatenate([torch.arange(r) + 1 for r in repeats])
            embedded_time = self.embed_time(remaining_time_horizon)

        mlp_outputs = self.MLP(torch.cat((analysis, embedded_time), dim = 1))
        # TODO: talk to Jacob and consider making this (1 @ 1 + 1) instead of (5 @ 5 + 1) for dims
        x1, x2, b = torch.split(mlp_outputs, [5, 5, 1], dim = 1)
        # matrix implementation of x1 @ x2.T plus a bias where x1 and x2 are row vectors
        # the idea here is to allow scaling since value will (to some extent) scale with time
        output = torch.sum(x1 * x2, dim = 1, keepdim = True) + b
        assert output.shape == (n, 1) or mode == BACKPROP_MODE
        return output

    def embed_time(self, remaining_time_horizon):
        assert len(remaining_time_horizon.shape) == 1, "time horizon must be 1d for proper indexing into embeddings"
        assert all(remaining_time_horizon < self.time_horizon), \
            "remaining time horizon must be less than the full time horizon"
        assert all(remaining_time_horizon > 0), "remaining time horizon must be at least 1"
        return self.time_embedding(remaining_time_horizon - 1) # subtract off one to zero index