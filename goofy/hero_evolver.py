import torch
from torch import nn
import torch.nn.functional as F

from model_constants import *

# TODO: set global low precision fp dtype

# first value is health
state_split = [1, PLAYER_TOKEN_LENGTH, NUM_WEAPON_TOKENS * WEAPON_TOKEN_LENGTH]

# trim a few unnecessary positions from a player token to create a token with same length as weapons
def trim_token(token):
    assert len(token.shape) == 1, "player token must be one dimensional"
    return token[PLAYER_TRIM_INDICES]

# create a new dimension at dim and expand it to have value n
def expand_new_dim(tensor, n, dim = 0):
    tensor = torch.unsqueeze(tensor, dim = dim)
    shape = [-1] * len(tensor.shape)
    shape[dim] = n
    return tensor.expand(*shape)

# using final dimension as logits, argmax to determine the class and then encode as a one hot vector
def argmax_logits_to_one_hot(tensor, num_classes):
    tensor = torch.argmax(tensor, dim = -1, keepdim = False)
    return F.one_hot(tensor, num_classes = num_classes)

class HeroEvolver(nn.Module):
    def __init__(self, weapon_pos_embedding_dim = 3, num_heads = 3):
        assert (weapon_pos_embedding_dim + WEAPON_TOKEN_LENGTH) % num_heads == 0, \
                                "num_heads must divide total weapon embedding dim"

        self.action_dim = ACTION_DIM
        self.player_dim = PLAYER_TOKEN_LENGTH
        self.weapon_dim = WEAPON_TOKEN_LENGTH
        self.weapon_pos_embedding_dim = weapon_pos_embedding_dim
        self.weapon_MLP_input_dim = 3 * (self.weapon_dim + self.weapon_pos_embedding_dim) + self.action_dim

        # weapon prediction layers
        self.weapon_pos_embedding_module = nn.Embedding(NUM_WEAPON_TOKENS, self.weapon_pos_embedding_dim)
        self.weapon_self_attn = nn.MultiheadAttention(self.weapon_dim + self.weapon_pos_embedding_dim, num_heads)
        self.weapon_to_player_attn = nn.MultiheadAttention(self.weapon_dim + self.weapon_pos_embedding_dim, num_heads,
                                                           kdim = self.player_dim, vdim = self.player_dim)
        self.weapon_MLP = nn.Sequential([
            nn.Linear(self.weapon_MLP_input_dim, 15),
            nn.ReLU(),
            nn.Linear(15, self.weapon_dim)
        ])

        # hero prediction layers
        self.hero_MLP = nn.Sequential([
            nn.Linear(self.player_dim + self.action_dim, 15),
            nn.ReLU(),
            nn.Linear(15, 12),
            nn.ReLU(),
            nn.Linear(12, self.player_dim)
        ])

        # adversary health prediction layers
        self.player_to_weapon_attn = nn.MultiheadAttention(self.weapon_dim, num_heads = 2)
        self.player_self_attn = nn.MultiheadAttention(self.weapon_dim, num_heads = 2)
        self.adversary_health_MLP = nn.Sequential([
            nn.Linear(self.weapon_dim + self.action_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        ])


    def forward(self, partial_s, actions):
        # preprocessing
        hero_state, adversary_state = torch.split(partial_s, 2)
        hero_health, hero_token, hero_weapon_tokens = torch.split(hero_state, state_split)
        adversary_health, adversary_token, adversary_weapon_tokens = torch.split(adversary_state, state_split)
        hero_weapon_tokens = hero_weapon_tokens.reshape(NUM_WEAPON_TOKENS, -1)

        # use respective forward functions
        hero_weapon_preds = self.weapon_forward(hero_weapon_tokens, hero_token, adversary_token, actions)
        hero_token_preds = self.hero_forward(hero_token, actions)
        adversary_health_preds = self.adversary_health_forward(self, adversary_health, hero_weapon_tokens, 
                                                               hero_token, adversary_token, actions)

        # replicate the parts of state that don't evolve to match the number of actions
        b = actions.shape[0]
        hero_health = expand_new_dim(hero_health, b, dim = 0)
        adversary_token = expand_new_dim(adversary_token, b, dim = 0)
        adversary_weapon_tokens = expand_new_dim(adversary_weapon_tokens, b, dim = 0)

        # repackage state
        hero_state_preds = torch.cat((hero_health, hero_token_preds, hero_weapon_preds), dim = 1)
        adversary_state_preds = torch.cat((adversary_health_preds, adversary_token, adversary_weapon_tokens), dim = 1)
        return torch.cat((hero_state_preds, adversary_state_preds), dim = 1)


    def weapon_forward(self, hero_weapon_tokens, hero_token, adversary_token, actions):
        assert hero_weapon_tokens.shape == (NUM_WEAPON_TOKENS, WEAPON_TOKEN_LENGTH)
        assert hero_token.shape == (PLAYER_TOKEN_LENGTH,)
        assert adversary_token.shape == (PLAYER_TOKEN_LENGTH,)
        assert len(actions.shape) == 2 and actions.shape[1] == self.action_dim

        # concat pos embeddings instead of adding since all of x's dimensions are saturated with info
        x = torch.cat((hero_weapon_tokens, self.weapon_pos_embedding), dim = 1)

        # weapon to weapon self attention
        self_attn_outputs = self.weapon_self_attn(x, x, x, need_weights = False)

        # weapon to player cross attention
        player_tokens = torch.stack((hero_token, adversary_token))
        cross_attn_outputs = self.weapon_to_player_attn(x, player_tokens, player_tokens, need_weights = False)

        # match every pair of weapon token and action and pass this to the MLP
        b = actions.shape[0]
        attn_outputs = torch.cat((x, self_attn_outputs, cross_attn_outputs), dim = 1)
        expanded_attn_outputs = expand_new_dim(attn_outputs, b, dim = 0) # w x d -> b x w x d
        expanded_actions = expand_new_dim(actions, NUM_WEAPON_TOKENS, dim = 1) # b x a -> b x w x a
        mlp_inputs = torch.cat((expanded_attn_outputs, expanded_actions), dim = 2) # b x w x (d + a)
        mlp_outputs = self.weapon_MLP(mlp_inputs)

        # take argmax of logits to classify weapon type
        weapon_types = argmax_logits_to_one_hot(mlp_outputs[:, :, -NUM_WEAPON_TYPES:], num_classes = NUM_WEAPON_TOKENS)
        weapon_outputs = torch.cat((mlp_outputs[:, :, :-NUM_WEAPON_TYPES], weapon_types), dim = 2)
        return weapon_outputs.reshape(b, -1)


    def hero_forward(self, hero_token, actions):
        b = actions.shape[0]
        expanded_hero_token = expand_new_dim(hero_token, b, dim = 0)
        mlp_outputs = self.hero_MLP(torch.cat((expanded_hero_token, actions), dim = 1))
        hero_modes = argmax_logits_to_one_hot(mlp_outputs[:, -NUM_MODES:], num_classes = NUM_MODES)
        return torch.cat((mlp_outputs[:, :-NUM_MODES], hero_modes), dim = 1)


    def adversary_health_forward(self, adversary_health, hero_weapon_tokens, hero_token, adversary_token, actions):
        # player to weapon cross attention
        trimmed_token = trim_token(adversary_token).reshape(1, -1)
        cross_attn_outputs = self.weapon_to_player_attn(trimmed_token, hero_weapon_tokens, hero_weapon_tokens, need_weights = False)

        # player to player self attention
        x = torch.stack((trim_token(hero_token), trim_token(adversary_token)))
        self_attn_outputs = self.player_self_attn(x, x, x, need_weights = False)

        # global pool self attention outputs and pass everything to MLP
        attn_outputs = torch.cat((torch.squeeze(cross_attn_outputs, dim = 0), torch.mean(self_attn_outputs, dim = 0)))
        b = actions.shape[0]
        expanded_attn_outputs = expand_new_dim(attn_outputs, b, dim = 0)
        mlp_input = torch.cat((expanded_attn_outputs, actions), dim = 1)
        return adversary_health + self.adversary_health_MLP(mlp_input)

    @property # helper function since we always use all embeddings
    def weapon_pos_embedding(self):
        return self.weapon_pos_embedding_module(torch.arange(NUM_WEAPON_TOKENS))