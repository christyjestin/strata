import torch
from torch import nn, Tensor
from typing import Union

from model_constants import *
from unconditional_analyzer import UnconditionalAnalyzer

class StrategyDistiller(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.strategy_dim = STRATEGY_DIM

        self.state_analyzer = UnconditionalAnalyzer()

        # shallow MLP to map the concatenation of analysis and action to the input for the LSTM
        self.MLP = nn.Sequential([
            nn.Linear((2 * self.state_analyzer.embedding_dim) + self.action_dim, 25),
            nn.ReLU(),
            nn.Linear(25, self.strategy_dim)
        ])

        self.LSTM = nn.LSTM(input_size = self.strategy_dim, hidden_size = self.strategy_dim)

    def forward(self, strategy: Strategy, states: Tensor, actions: Tensor, mode: str) -> Union[Strategy, Tensor]:
        assert states.shape[0] == actions.shape[0], "states and actions must align"
        assert len(states.shape) == 2 and states.shape[1] == self.state_dim
        assert len(actions.shape) == 2 and actions.shape[1] == self.action_dim
        assert mode in [SEARCH_MODE, BACKPROP_MODE]
        assert mode != SEARCH_MODE or states.shape[0] == 1, "there should only be a single state in search mode"

        state_analysis = self.state_analyzer(states)
        LSTM_input = self.MLP(torch.cat((state_analysis, actions), dim = 1))
        output, (final_hidden, final_cell) = self.LSTM(LSTM_input, (strategy.hidden, strategy.cell))

        # we should return both hidden and cell state in search mode because the cell state will be relevant
        # in future calls; in backprop mode just return the output (i.e. the series of hidden states)
        return Strategy(final_hidden, final_cell) if mode == SEARCH_MODE else output

    # helper function to init both hidden and cell states as all zeros
    def init_strategy(self) -> Strategy:
        return Strategy(torch.zeros((1, self.strategy_dim)), torch.zeros((1, self.strategy_dim)))