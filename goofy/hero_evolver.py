import torch
from torch import nn, optim
import torch.nn.functional as F
from collections import namedtuple
from itertools import chain

from model_constants import *

# TODO: set global low precision fp dtype

HeroEvolverLoss = namedtuple('HeroEvolverLoss', ['hero_token', 'weapon_token', 'adversary_health'])

# first value is health
STATE_SPLIT = [1, PLAYER_TOKEN_LENGTH, NUM_WEAPON_TOKENS * WEAPON_TOKEN_LENGTH]

# trim a few unnecessary positions from a player token to create a token with same length as weapons
def trim_token(token):
    assert len(token.shape) == 2 and token.shape[1] == PLAYER_TOKEN_LENGTH, "player token must be two dimensional"
    return token[:, PLAYER_TRIM_INDICES]

# using final dimension as logits, argmax to determine the class and then encode as a one hot vector
def argmax_logits_to_one_hot(tensor, num_classes):
    tensor = torch.argmax(tensor, dim = -1, keepdim = False)
    return F.one_hot(tensor, num_classes = num_classes)

class HeroEvolver(nn.Module):
    def __init__(self, weapon_pos_embedding_dim = 3, num_heads = 3, hero_token_cross_entropy_lambda = 1, 
                 weapon_token_cross_entropy_lambda = 1):
        super().__init__()
        self.action_dim = ACTION_DIM
        self.player_dim = PLAYER_TOKEN_LENGTH
        self.weapon_dim = WEAPON_TOKEN_LENGTH
        self.state_dim = 2 * sum(STATE_SPLIT)
        self.weapon_pos_embedding_dim = weapon_pos_embedding_dim
        self.weapon_MLP_input_dim = 3 * (self.weapon_dim + self.weapon_pos_embedding_dim) + self.action_dim

        assert (weapon_pos_embedding_dim + self.weapon_dim) % num_heads == 0, \
                                "num_heads must evenly divide the total dimension of the weapon attention embedding"

        # weapon prediction layers
        self.weapon_pos_embedding_module = nn.Embedding(NUM_WEAPON_TOKENS, self.weapon_pos_embedding_dim)
        self.weapon_self_attn = nn.MultiheadAttention(self.weapon_dim + self.weapon_pos_embedding_dim, num_heads, 
                                                      batch_first = True)
        self.weapon_to_player_attn = nn.MultiheadAttention(self.weapon_dim + self.weapon_pos_embedding_dim, num_heads,
                                                           kdim = self.player_dim, vdim = self.player_dim, batch_first = True)
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
        self.player_to_weapon_attn = nn.MultiheadAttention(self.weapon_dim, num_heads = 2, batch_first = True)
        self.player_self_attn = nn.MultiheadAttention(self.weapon_dim, num_heads = 2, batch_first = True)
        self.adversary_health_MLP = nn.Sequential([
            nn.Linear(self.weapon_dim + self.action_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        ])

        # lambda values, loss functions, and optimizers
        self.hero_token_CE_lambda = hero_token_cross_entropy_lambda
        self.weapon_token_CE_lambda = weapon_token_cross_entropy_lambda
        self.CE_loss = nn.CrossEntropyLoss()
        self.hero_token_L2_loss_weight = torch.tensor(PLAYER_L2_WEIGHT).reshape(1, -1)
        self.weapon_token_L2_loss_weight = torch.tensor(WEAPON_L2_WEIGHT).reshape(1, -1)

        attn_models = [self.weapon_self_attn, self.weapon_to_player_attn, self.player_to_weapon_attn, self.player_self_attn]
        MLP_models = [self.weapon_MLP, self.hero_MLP, self.adversary_health_MLP]
        all_models = [self.weapon_pos_embedding_module, *attn_models, *MLP_models]
        params = [model.parameters() for model in all_models]
        self.optimizer = optim.Adam(chain(*params), lr = 1e-3)

    def forward(self, partial_next_states, actions, in_search_mode):
        assert len(partial_next_states.shape) == 2 and partial_next_states.shape[1] == self.state_dim
        assert len(actions.shape) == 2 and actions.shape[1] == self.action_dim
        assert in_search_mode or partial_next_states.shape[0] == actions.shape[0], \
                                            "states and actions must align in backprop mode"

        # preprocessing
        hero_state, adversary_state = torch.split(partial_next_states, 2, dim = 1)
        hero_health, hero_token, hero_weapon_tokens = torch.split(hero_state, STATE_SPLIT, dim = 1)
        adversary_health, adversary_token, adversary_weapon_tokens = torch.split(adversary_state, STATE_SPLIT, dim = 1)
        hero_weapon_tokens = hero_weapon_tokens.reshape(-1, NUM_WEAPON_TOKENS, self.weapon_dim)

        # use respective forward functions
        hero_weapon_preds = self.weapon_forward(hero_weapon_tokens, hero_token, adversary_token, actions, 
                                                in_search_mode = in_search_mode)
        hero_token_preds = self.hero_forward(hero_token, actions, in_search_mode = in_search_mode)
        adversary_health_preds = self.adversary_health_forward(self, adversary_health, hero_weapon_tokens, hero_token, 
                                                               adversary_token, actions, in_search_mode = in_search_mode)

        # in search mode, we're considering one state and b actions, so we need to replicate the parts of 
        # state that don't evolve to match the number of actions
        # in backprop mode, we're considering n states and n corresponding actions, so no changes are necessary
        if in_search_mode:
            b = actions.shape[0] # branching number
            hero_health = hero_health.expand(b, -1)
            adversary_token = adversary_token.expand(b, -1)
            adversary_weapon_tokens = adversary_weapon_tokens.expand(b, -1)

        # repackage state
        hero_state_preds = torch.cat((hero_health, hero_token_preds, hero_weapon_preds), dim = 1)
        adversary_state_preds = torch.cat((adversary_health_preds, adversary_token, adversary_weapon_tokens), dim = 1)
        next_state_preds = torch.cat((hero_state_preds, adversary_state_preds), dim = 1)

        return next_state_preds


    # predicts next state for all weapon tokens
    # one state paired with each of b actions in search mode and n state/action pairs in backprop mode
    def weapon_forward(self, hero_weapon_tokens, hero_token, adversary_token, actions, in_search_mode):
        assert len(hero_weapon_tokens.shape) == 3 and \
                hero_weapon_tokens.shape[1:] == (NUM_WEAPON_TOKENS, self.weapon_dim)
        assert len(hero_token.shape) == 2 and hero_token.shape[1] == self.player_dim
        assert len(adversary_token.shape) == 2 and adversary_token.shape[1] == self.player_dim
        assert len(actions.shape) == 2 and actions.shape[1] == self.action_dim
        assert in_search_mode or hero_token.shape[0] == actions.shape[0], "states and actions must align in backprop mode"

        n = hero_weapon_tokens.shape[0] # batch size
        b = actions.shape[0] # branching number in search mode and batch size in backprop mode

        # concat pos embeddings instead of adding since all of x's dimensions are saturated with info
        x = torch.cat((hero_weapon_tokens, self.weapon_pos_embedding.expand(n, -1, -1)), dim = 2) # n x w x d_w

        # weapon to weapon self attention
        self_attn_outputs = self.weapon_self_attn(x, x, x, need_weights = False) # n x w x d_w

        # weapon to player cross attention
        player_tokens = torch.stack((hero_token, adversary_token), dim = 1) # n x 2 x d_p
        cross_attn_outputs = self.weapon_to_player_attn(x, player_tokens, player_tokens, need_weights = False) # n x w x d_w

        attn_outputs = torch.cat((x, self_attn_outputs, cross_attn_outputs), dim = 2)

        # in search mode, there is only one state, so we need to expand it to line up with the number of actions
        if in_search_mode:
            assert n == 1
            attn_outputs = attn_outputs.expand(b, -1, -1)

        actions = actions.expand(-1, NUM_WEAPON_TOKENS, -1) # b x w x a
        mlp_inputs = torch.cat((attn_outputs, actions), dim = 2) # b x w x (d + a)
        mlp_outputs = self.weapon_MLP(mlp_inputs) # b x w x o

        if in_search_mode:
            # take argmax of logits to classify weapon type in search mode
            weapon_types = argmax_logits_to_one_hot(mlp_outputs[:, :, -NUM_WEAPON_TYPES:], num_classes = NUM_WEAPON_TYPES)
            weapon_outputs = torch.cat((mlp_outputs[:, :, :-NUM_WEAPON_TYPES], weapon_types), dim = 2)
        else:
            # preserve logits for cross entropy loss in backprop mode
            weapon_outputs = mlp_outputs

        # repack all weapon tokens into one row
        weapon_outputs = weapon_outputs.reshape(b, -1)
        assert weapon_outputs.shape == (b, self.weapon_dim)

        return weapon_outputs


    # see weapon_forward for explanation about search mode and backprop mode
    def hero_forward(self, hero_token, actions, in_search_mode):
        assert len(hero_token.shape) == 2 and hero_token.shape[1] == self.player_dim
        assert len(actions.shape) == 2 and actions.shape[1] == self.action_dim
        assert in_search_mode or hero_token.shape[0] == actions.shape[0], "states and actions must align in backprop mode"

        n = hero_token.shape[0] # batch size
        b = actions.shape[0] # branching number in search mode and batch size in backprop mode

        # in search mode, there is only one state, so we need to expand it to line up with the number of actions
        if in_search_mode:
            assert n == 1
            hero_token = hero_token.expand(b, -1)

        mlp_outputs = self.hero_MLP(torch.cat((hero_token, actions), dim = 1))
        if in_search_mode:
            # take argmax of logits to classify player mode in search mode
            hero_modes = argmax_logits_to_one_hot(mlp_outputs[:, -NUM_MODES:], num_classes = NUM_MODES)
            hero_outputs = torch.cat((mlp_outputs[:, :-NUM_MODES], hero_modes), dim = 1)
        else:
            # preserve logits for cross entropy loss in backprop mode
            hero_outputs = mlp_outputs

        assert hero_outputs.shape == (b, self.player_dim)
        return hero_outputs


    # see weapon_forward for explanation about search mode and backprop mode
    def adversary_health_forward(self, adversary_health, hero_weapon_tokens, hero_token, adversary_token, actions, 
                                 in_search_mode):
        assert len(adversary_health.shape) == 2 and adversary_health.shape[1] == 1
        assert len(hero_weapon_tokens.shape) == 3 and hero_weapon_tokens.shape[1:] == (NUM_WEAPON_TOKENS, self.weapon_dim)
        assert len(hero_token.shape) == 2 and hero_token.shape[1] == self.player_dim
        assert len(adversary_token.shape) == 2 and adversary_token.shape[1] == self.player_dim
        assert len(actions.shape) == 2 and actions.shape[1] == self.action_dim
        assert in_search_mode or hero_token.shape[0] == actions.shape[0], "states and actions must align in backprop mode"

        n = hero_token.shape[0]
        b = actions.shape[0]

        # player to weapon cross attention
        trimmed_token = trim_token(adversary_token)
        cross_attn_outputs = self.weapon_to_player_attn(trimmed_token, hero_weapon_tokens, hero_weapon_tokens, 
                                                        need_weights = False)

        # player to player self attention
        x = torch.cat((trim_token(hero_token), trim_token(adversary_token)), dim = 0)
        self_attn_outputs = self.player_self_attn(x, x, x, need_weights = False)

        # global pool self attention outputs and pass everything to MLP
        self_attn_outputs = torch.mean(self_attn_outputs, dim = 0, keepdim = True)
        attn_outputs = torch.cat((cross_attn_outputs, self_attn_outputs), dim = 1)

        # in search mode, there is only one state, so we need to expand it to line up with the number of actions
        if in_search_mode:
            assert n == 1
            attn_outputs = attn_outputs.expand(b, -1)

        # mlp outputs the change in health rather than the raw value (so we add this to the current health value)
        mlp_input = torch.cat((attn_outputs, actions), dim = 1)
        adversary_health_outputs = adversary_health + self.adversary_health_MLP(mlp_input)
        assert adversary_health_outputs.shape == (b, 1)
        return adversary_health_outputs

    @property # helper function since we always use all embeddings
    def weapon_pos_embedding(self):
        return self.weapon_pos_embedding_module(torch.arange(NUM_WEAPON_TOKENS))


    # computes loss, back propagates, and steps with the optimizer
    # returns a namedtuple with the 3 components of hero evolver loss
    def backward_pass(self, partial_next_states, actions, next_states):
        next_state_preds = self.forward(partial_next_states, actions, in_search_mode = False)
        loss, loss_breakdown = self.loss_function(next_state_preds, next_states)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_breakdown

    # helper function for computing loss
    def extract_mutable_state(self, next_states):
        hero_state, adversary_state = torch.split(next_states, 2, dim = 1)
        _, hero_token, hero_weapon_tokens = torch.split(hero_state, STATE_SPLIT, dim = 1)
        adversary_health, _, _ = torch.split(adversary_state, STATE_SPLIT, dim = 1)
        hero_weapon_tokens = hero_weapon_tokens.reshape(-1, NUM_WEAPON_TOKENS, self.weapon_dim)
        return (hero_weapon_tokens, hero_token, adversary_health)

    def loss_function(self, next_state_preds, next_states):
        # split up state
        weapon_tokens, hero_token, adversary_health = self.extract_mutable_state(next_states)
        weapon_token_preds, hero_token_preds, adversary_health_preds = self.extract_mutable_state(next_state_preds)

        # call respective loss functions
        hero_token_loss = self.compute_hero_token_loss(hero_token_preds, hero_token)
        weapon_token_loss = self.compute_weapon_token_loss(weapon_token_preds, weapon_tokens)
        adversary_health_loss = self.compute_adversary_health_loss(adversary_health_preds, adversary_health)

        total_loss = hero_token_loss + weapon_token_loss + adversary_health_loss
        return (total_loss, HeroEvolverLoss(hero_token_loss.item(), weapon_token_loss.item(), adversary_health_loss.item()))

    # note that weight must be a 2d row vector with the same final dimension as preds and target
    def weighted_L2_loss(self, preds, target, weight):
        assert preds.shape == target.shape, "preds and target must be the same shape"
        assert len(weight.shape) == 2 and weight.shape[0] == 1, "weight must be a 2d row vector"
        assert preds.shape[-1] == weight.shape[-1], "weight must align with feature dimension of preds and target"
        return torch.mean(((preds - target) ** 2) @ weight.T)

    # cross entropy for one hot part of token and weighted L2 for the rest
    def compute_hero_token_loss(self, preds, target):
        assert preds.shape == target.shape, "preds and target must be the same shape"
        assert len(preds.shape) == 2 and preds.shape[1] == self.player_dim

        L2_loss = self.weighted_L2_loss(preds[:, :-NUM_MODES], target[:, :-NUM_MODES], self.hero_token_L2_loss_weight)
        CE_loss = self.CE_loss(preds[:, -NUM_MODES:], target[:, -NUM_MODES:])
        return L2_loss + self.hero_token_CE_lambda * CE_loss

    # cross entropy for one hot part of token and weighted L2 for the rest
    def compute_weapon_token_loss(self, preds, target):
        assert preds.shape == target.shape, "preds and target must be the same shape"
        assert len(preds.shape) and preds.shape[1:] == (NUM_WEAPON_TOKENS, self.weapon_dim)

        L2_loss = self.weighted_L2_loss(preds[:, :, :-NUM_WEAPON_TYPES], target[:, :, :-NUM_WEAPON_TYPES], 
                                     self.weapon_token_L2_loss_weight)
        CE_loss = self.CE_loss(preds[:, :, -NUM_WEAPON_TYPES:], target[:, :, -NUM_WEAPON_TYPES:])
        return L2_loss + self.weapon_token_CE_lambda * CE_loss

    # standard L2
    def compute_adversary_health_loss(self, preds, target):
        return torch.mean((preds - target) ** 2)