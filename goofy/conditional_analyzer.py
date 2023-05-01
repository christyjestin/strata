import torch
from torch import nn

from model_constants import *

class ConditionalAnalyzer(nn.Module):
    def __init__(self, num_heads = 5):
        super().__init__()
        self.player_dim = PLAYER_TOKEN_LENGTH
        self.weapon_dim = WEAPON_TOKEN_LENGTH
        self.state_dim = STATE_DIM
        self.strategy_dim = STRATEGY_DIM
        self.embedding_dim = self.strategy_dim
        self.output_dim = 3 * self.embedding_dim

        assert (self.embedding_dim % num_heads) == 0, "num_heads must evenly divide the embedding dimension"

        # one extra binary dimension to indicate which player the tokens belong to
        self.player_projection_head = nn.Linear(self.player_dim + 1, self.embedding_dim)
        self.weapon_projection_head = nn.Linear(self.weapon_dim + 1, self.embedding_dim)

        # attend strategy and player tokens to all tokens
        self.full_attn = nn.MultiheadAttention(self.embedding_dim, num_heads = num_heads, batch_first = True)
        # attend only the strategy and player tokens to each other
        self.partial_attn = nn.MultiheadAttention(self.embedding_dim, num_heads = num_heads, batch_first = True)

    def forward(self, states, strategies, in_search_mode):
        assert len(states.shape) == 2 and states.shape[1] == self.state_dim
        assert len(strategies.shape) == 2 and strategies.shape[1] == self.strategy_dim
        assert in_search_mode or states.shape[0] == strategies.shape[0], "states and strategies must align in backprop mode"

        n = states.shape[0] # batch size
        # in search mode, there's a single strategy and many states, so we need to expand strategies to match states
        if in_search_mode:
            assert strategies.shape[0] == 1
            strategies = strategies.expand(n, -1)
        strategies = torch.unsqueeze(strategies, dim = 1) # n x 1 x e

        # preprocessing
        hero_state, adversary_state = torch.split(states, 2, dim = 1)
        _, hero_token, hero_weapon_tokens = torch.split(hero_state, STATE_SPLIT, dim = 1)
        _, adversary_token, adversary_weapon_tokens = torch.split(adversary_state, STATE_SPLIT, dim = 1)
        hero_weapon_tokens = hero_weapon_tokens.reshape(n, NUM_WEAPON_TOKENS, self.weapon_dim) # n x (w * d_w) -> n x w x d_w
        adversary_weapon_tokens = adversary_weapon_tokens.reshape(n, NUM_WEAPON_TOKENS, self.weapon_dim)

        # add 1's for tokens belonging to hero and 0's for tokens belonging to adversary
        hero_token = torch.cat((hero_token, torch.ones((n, 1))), dim = 1)
        adversary_token = torch.cat((adversary_token, torch.zeros((n, 1))), dim = 1)
        hero_weapon_tokens = torch.cat((hero_weapon_tokens, torch.ones((n, NUM_WEAPON_TOKENS, 1))), dim = 2)
        adversary_weapon_tokens = torch.cat((adversary_weapon_tokens, torch.zeros((n, NUM_WEAPON_TOKENS, 1))), dim = 2)

        # project player and weapon tokens to match the embedding dim
        projected_player_tokens = self.player_projection_head(torch.stack((hero_token, adversary_token)), dim = 1) # n x 2 x e
        weapon_tokens = torch.cat((hero_weapon_tokens, adversary_weapon_tokens), dim = 1) # n x 2w x (d_w + 1)
        projected_weapon_tokens = self.weapon_projection_head(weapon_tokens) # n x 2w x e

        q = torch.cat((strategies, projected_player_tokens), dim = 1) # n x 3 x e
        kv = torch.cat((q, projected_weapon_tokens), dim = 1) # n x (3 + 2w) x e
        # attend strategy and player tokens to every token (including weapons)
        x = self.full_attn(q, kv, kv, need_weights = False)
        x = x + q # residual connection
        # attend strategy and player tokens to only each other
        output = self.partial_attn(x, x, x, need_weights = False)

        return output.reshape((n, 3 * self.embedding_dim))
