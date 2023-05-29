import torch
from torch import nn

from model_constants import *

class UnconditionalAnalyzer(nn.Module):
    def __init__(self, embedding_dim = 10, num_heads = 2):
        super().__init__()
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.player_dim = PLAYER_TOKEN_LENGTH
        self.weapon_dim = WEAPON_TOKEN_LENGTH
        self.embedding_dim = embedding_dim

        assert (embedding_dim % num_heads) == 0, 'the number of heads must be divisible by the embedding dim'

        # one extra binary dimension to indicate which player the tokens belong to
        self.player_projection_head = nn.Linear(self.player_dim + 1, self.embedding_dim)
        self.weapon_projection_head = nn.Linear(self.weapon_dim + 1, self.embedding_dim)

        # attends player tokens to all tokens (including themselves)
        self.attn = nn.MultiheadAttention(self.embedding_dim, num_heads = num_heads, batch_first = True)

    def forward(self, states):
        assert len(states.shape) == 2 and states.shape[1] == self.state_dim

        n = states.shape[0] # batch size

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
        weapon_tokens = torch.cat((hero_weapon_tokens, adversary_weapon_tokens), dim = 1)
        projected_weapon_tokens = self.weapon_projection_head(weapon_tokens) # n x 2w x e

        q = projected_player_tokens
        kv = torch.cat((projected_player_tokens, projected_weapon_tokens), dim = 1) # n x (2 + 2w) x e
        # attend player tokens to every token (including weapons)
        output = self.attn(q, kv, kv, need_weights = False) # n x 2 x e
        return output.reshape(n, -1) # flatten attn outputs