import torch
from torch import nn, optim
import torch.nn.functional as F
from itertools import chain

from model_constants import *

# TODO: set global low precision fp dtype

# trim a few unnecessary positions from a player token to create a token with same length as weapons
def trim_token(token):
    assert len(token.shape) == 2 and token.shape[1] == PLAYER_TOKEN_LENGTH, "player token must be two dimensional"
    return token[:, PLAYER_TRIM_INDICES]

# using final dimension as logits, argmax to determine the class and then encode as a one hot vector
def argmax_logits_to_one_hot(tensor, num_classes):
    tensor = torch.argmax(tensor, dim = -1, keepdim = False)
    return F.one_hot(tensor, num_classes = num_classes)

class Evolver(nn.Module):
    def __init__(self, weapon_pos_embedding_dim = 3, player_pos_embedding_dim = 2, num_heads = 3, 
                 player_token_cross_entropy_lambda = 1, weapon_token_cross_entropy_lambda = 1):
        super().__init__()
        self.action_dim = ACTION_DIM
        self.player_dim = PLAYER_TOKEN_LENGTH
        self.weapon_dim = WEAPON_TOKEN_LENGTH
        self.state_dim = STATE_DIM
        self.weapon_pos_embedding_dim = weapon_pos_embedding_dim
        self.player_pos_embedding_dim = player_pos_embedding_dim
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

        # player prediction layers
        self.player_MLP = nn.Sequential([
            nn.Linear(self.player_dim + self.action_dim, 15),
            nn.ReLU(),
            nn.Linear(15, 12),
            nn.ReLU(),
            nn.Linear(12, self.player_dim)
        ])

        # opponent health prediction layers
        self.player_pos_embedding_module = nn.Embedding(2, self.player_pos_embedding_dim)
        self.player_to_weapon_attn = nn.MultiheadAttention(self.weapon_dim, num_heads = 2, batch_first = True)
        self.player_self_attn = nn.MultiheadAttention(self.weapon_dim + self.player_pos_embedding_dim, num_heads = 2, 
                                                      batch_first = True)
        self.opponent_health_MLP = nn.Sequential([
            nn.Linear((2 * self.weapon_dim) + self.player_pos_embedding_dim + self.action_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        ])

        # lambda values, loss functions, and optimizers
        self.player_token_CE_lambda = player_token_cross_entropy_lambda
        self.weapon_token_CE_lambda = weapon_token_cross_entropy_lambda
        self.CE_loss = nn.CrossEntropyLoss()
        self.player_token_L2_loss_weight = torch.tensor(PLAYER_L2_WEIGHT).reshape(1, -1)
        self.weapon_token_L2_loss_weight = torch.tensor(WEAPON_L2_WEIGHT).reshape(1, -1)

        attn_models = [self.weapon_self_attn, self.weapon_to_player_attn, self.player_to_weapon_attn, self.player_self_attn]
        MLP_models = [self.weapon_MLP, self.player_MLP, self.opponent_health_MLP]
        all_models = [self.weapon_pos_embedding_module, *attn_models, *MLP_models]
        params = [model.parameters() for model in all_models]
        self.optimizer = optim.Adam(chain(*params), lr = 1e-3)

    # predicts next state given current state and action
    def forward(self, states, actions, misaligned_dims, is_adversary_step, mode):
        assert len(states.shape) == 2 and states.shape[1] == self.state_dim # n x s
        assert len(actions.shape) == 2 and actions.shape[1] == self.action_dim # nb x a
        assert misaligned_dims or states.shape[0] == actions.shape[0], \
                                            "states and actions must align unless stated otherwise"
        assert mode in [SEARCH_MODE, BACKPROP_MODE]

        # reverse the player and opponent states if you're evolving the adversary
        if is_adversary_step:
            states = self.flip_state(states)

        n = states.shape[0] # batch size
        assert actions.shape[0] % n == 0, "the number of actions must be evenly divisible by the number of states"
        b = actions.shape[0] // n # branching number

        # preprocessing
        player_state, opponent_state = torch.split(states, 2, dim = 1)
        player_health, player_token, player_weapon_tokens = torch.split(player_state, STATE_SPLIT, dim = 1)
        opponent_health, opponent_token, opponent_weapon_tokens = torch.split(opponent_state, STATE_SPLIT, dim = 1)

        # use respective forward functions
        player_weapon_preds = self.weapon_forward(player_weapon_tokens, player_token, opponent_token, actions, 
                                                misaligned_dims = misaligned_dims, mode = mode)
        player_token_preds = self.player_forward(player_token, actions, misaligned_dims = misaligned_dims, mode = mode)
        opponent_health_preds = self.opponent_health_forward(opponent_health, player_weapon_tokens, player_token, 
                                                               opponent_token, actions, misaligned_dims = misaligned_dims)

        # replicate the parts of state that don't evolve to match the number of actions
        if misaligned_dims:
            player_health = torch.repeat_interleave(player_health, b, dim = 0)
            opponent_token = torch.repeat_interleave(opponent_token, b, dim = 0)
            opponent_weapon_tokens = torch.repeat_interleave(opponent_weapon_tokens, b, dim = 0)

        # repackage state
        player_state_preds = torch.cat((player_health, player_token_preds, player_weapon_preds), dim = 1)
        opponent_state_preds = torch.cat((opponent_health_preds, opponent_token, opponent_weapon_tokens), dim = 1)
        next_state_preds = torch.cat((player_state_preds, opponent_state_preds), dim = 1)
        assert next_state_preds.shape == (actions.shape[0], self.state_dim) # nb x s

        # re-reverse the player and opponent states if you're evolving the adversary
        if is_adversary_step:
            next_state_preds = self.flip_state(next_state_preds)

        return next_state_preds


    # predicts next state for all weapon tokens
    # note that w always refers to NUM_WEAPON_TOKENS
    def weapon_forward(self, player_weapon_tokens, player_token, opponent_token, actions, misaligned_dims, mode):
        assert len(player_weapon_tokens.shape) == 2 and \
                player_weapon_tokens.shape[1] == NUM_WEAPON_TOKENS * self.weapon_dim
        assert len(player_token.shape) == 2 and player_token.shape[1] == self.player_dim
        assert len(opponent_token.shape) == 2 and opponent_token.shape[1] == self.player_dim
        assert len(actions.shape) == 2 and actions.shape[1] == self.action_dim
        assert misaligned_dims or player_token.shape[0] == actions.shape[0], \
                                        "states and actions must align unless stated otherwise"

        n = player_token.shape[0] # batch size
        nb = actions.shape[0]
        assert nb % n == 0, "the number of actions must be evenly divisible by the number of states"
        b = nb // n # branching number

        # split up weapons so that they're a sequence of weapon tokens
        player_weapon_tokens = player_weapon_tokens.reshape(n, NUM_WEAPON_TOKENS, self.weapon_dim)

        # concat pos embeddings instead of adding since all of x's dimensions are saturated with info
        x = torch.cat((player_weapon_tokens, self.weapon_pos_embedding.expand(n, -1, -1)), dim = 2) # n x w x d_w

        # weapon to weapon self attention
        self_attn_outputs = self.weapon_self_attn(x, x, x, need_weights = False) # n x w x d_w

        # weapon to player cross attention
        player_tokens = torch.stack((player_token, opponent_token), dim = 1) # n x 2 x d_p
        cross_attn_outputs = self.weapon_to_player_attn(x, player_tokens, player_tokens, need_weights = False) # n x w x d_w

        attn_outputs = torch.cat((x, self_attn_outputs, cross_attn_outputs), dim = 2) # n x w x (2 * d_w)

        # replicate state to match the number of actions
        if misaligned_dims:
            attn_outputs = torch.repeat_interleave(attn_outputs, b, dim = 0) # nb x w x (2 * d_w)

        actions = actions.expand(-1, NUM_WEAPON_TOKENS, -1) # nb x a -> nb x w x a
        mlp_inputs = torch.cat((attn_outputs, actions), dim = 2) # nb x w x (d + a)
        mlp_outputs = self.weapon_MLP(mlp_inputs) # nb x w x (d + a) -> nb x w x o

        if mode == SEARCH_MODE:
            # take argmax of logits to classify weapon type in search mode
            weapon_types = argmax_logits_to_one_hot(mlp_outputs[:, :, -NUM_WEAPON_TYPES:], num_classes = NUM_WEAPON_TYPES)
            weapon_outputs = torch.cat((mlp_outputs[:, :, :-NUM_WEAPON_TYPES], weapon_types), dim = 2)
        else:
            # preserve logits for cross entropy loss in backprop mode
            weapon_outputs = mlp_outputs

        # repack all weapon tokens into one row
        weapon_outputs = weapon_outputs.flatten(end_dim = 1)
        assert weapon_outputs.shape == (nb, NUM_WEAPON_TOKENS * self.weapon_dim)
        return weapon_outputs

    # predicts next state for player token
    def player_forward(self, player_token, actions, misaligned_dims, mode):
        assert len(player_token.shape) == 2 and player_token.shape[1] == self.player_dim
        assert len(actions.shape) == 2 and actions.shape[1] == self.action_dim
        assert misaligned_dims or player_token.shape[0] == actions.shape[0], \
                                    "states and actions must align unless stated otherwise"
        assert mode in [SEARCH_MODE, BACKPROP_MODE]

        n = player_token.shape[0] # batch size
        nb = actions.shape[0]
        assert nb % n == 0, "the number of actions must be evenly divisible by the number of states"
        b = nb // n # branching number

        # replicate state to match the number of actions
        if misaligned_dims:
            player_token = torch.repeat_interleave(player_token, b, dim = 0)

        mlp_outputs = self.player_MLP(torch.cat((player_token, actions), dim = 1))
        if mode == SEARCH_MODE:
            # take argmax of logits to classify the player's mode in search mode
            player_modes = argmax_logits_to_one_hot(mlp_outputs[:, -NUM_PLAYER_MODES:], num_classes = NUM_PLAYER_MODES)
            player_outputs = torch.cat((mlp_outputs[:, :-NUM_PLAYER_MODES], player_modes), dim = 1)
        else:
            # preserve logits for cross entropy loss in backprop mode
            player_outputs = mlp_outputs

        assert player_outputs.shape == (nb, self.player_dim)
        return player_outputs


    def opponent_health_forward(self, opponent_health, player_weapon_tokens, player_token, opponent_token, actions, 
                                 misaligned_dims):
        assert len(opponent_health.shape) == 2 and opponent_health.shape[1] == 1
        assert len(player_weapon_tokens.shape) == 2 and \
            player_weapon_tokens.shape[1] == NUM_WEAPON_TOKENS * self.weapon_dim
        assert len(player_token.shape) == 2 and player_token.shape[1] == self.player_dim
        assert len(opponent_token.shape) == 2 and opponent_token.shape[1] == self.player_dim
        assert len(actions.shape) == 2 and actions.shape[1] == self.action_dim
        assert misaligned_dims or player_token.shape[0] == actions.shape[0], \
                                        "states and actions must align unless stated otherwise"

        n = player_token.shape[0] # batch size
        nb = actions.shape[0]
        assert nb % n == 0, "the number of actions must be evenly divisible by the number of states"
        b = nb // n # branching number

        # player to weapon cross attention
        trimmed_token = trim_token(opponent_token)
        cross_attn_outputs = self.weapon_to_player_attn(trimmed_token, player_weapon_tokens, player_weapon_tokens, 
                                                        need_weights = False)

        # player to player self attention
        x = torch.cat((trim_token(player_token), trim_token(opponent_token)), dim = 0)
        x = torch.cat((x, self.player_pos_embedding), dim = 1)
        self_attn_outputs = self.player_self_attn(x, x, x, need_weights = False)

        # global pool self attention outputs and pass everything to MLP
        self_attn_outputs = torch.mean(self_attn_outputs, dim = 0, keepdim = True)
        attn_outputs = torch.cat((cross_attn_outputs, self_attn_outputs), dim = 1)

        # replicate state to match the number of actions
        if misaligned_dims:
            attn_outputs = torch.repeat_interleave(attn_outputs, b, dim = 0)

        mlp_input = torch.cat((attn_outputs, actions), dim = 1)
        # the output of the mlp is the change in health rather than the raw final value
        opponent_health_outputs = opponent_health + self.opponent_health_MLP(mlp_input)

        assert opponent_health_outputs.shape == (nb, 1)
        return opponent_health_outputs

    @property # helper function since we always use all embeddings
    def weapon_pos_embedding(self):
        return self.weapon_pos_embedding_module(torch.arange(NUM_WEAPON_TOKENS))

    @property # helper function since we always use all embeddings
    def player_pos_embedding(self):
        return self.player_pos_embedding_module(torch.arange(2))

    # computes loss, back propagates, and steps with the optimizer
    # returns a namedtuple with the 3 components of evolver loss
    def backward_pass(self, states, actions, next_states, evolving_adversary):
        # reverse the player and opponent states if you're evolving the adversary
        # there's no need to flip back because we won't ever look at the outputs of the forward pass
        if evolving_adversary:
            states = self.flip_state(states)
            next_states = self.flip_state(next_states)

        next_state_preds = self.forward(states, actions, misaligned_dims = False)
        loss, loss_breakdown = self.loss_function(next_state_preds, next_states)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_breakdown

    # helper function for computing loss
    def extract_mutable_state(self, next_states):
        player_state, opponent_state = torch.split(next_states, 2, dim = 1)
        _, player_token, player_weapon_tokens = torch.split(player_state, STATE_SPLIT, dim = 1)
        opponent_health, _, _ = torch.split(opponent_state, STATE_SPLIT, dim = 1)
        player_weapon_tokens = player_weapon_tokens.reshape(-1, NUM_WEAPON_TOKENS, self.weapon_dim)
        return (player_weapon_tokens, player_token, opponent_health)

    def loss_function(self, next_state_preds, next_states):
        assert next_state_preds.shape == next_states.shape
        assert len(next_states.shape) == 2 and next_states.shape[1] == self.state_dim

        # split up state
        weapon_tokens, player_token, opponent_health = self.extract_mutable_state(next_states)
        weapon_token_preds, player_token_preds, opponent_health_preds = self.extract_mutable_state(next_state_preds)

        # call respective loss functions
        player_token_loss = self.compute_player_token_loss(player_token_preds, player_token)
        weapon_token_loss = self.compute_weapon_token_loss(weapon_token_preds, weapon_tokens)
        opponent_health_loss = self.compute_opponent_health_loss(opponent_health_preds, opponent_health)

        total_loss = player_token_loss + weapon_token_loss + opponent_health_loss
        return (total_loss, EvolverLoss(player_token_loss.item(), weapon_token_loss.item(), opponent_health_loss.item()))

    # note that weight must be a 2d row vector with the same final dimension as preds and target
    def weighted_L2_loss(self, preds, target, weight):
        assert preds.shape == target.shape, "preds and target must be the same shape"
        assert len(weight.shape) == 2 and weight.shape[0] == 1, "weight must be a 2d row vector"
        assert preds.shape[-1] == weight.shape[-1], "weight must align with feature dimension of preds and target"
        return torch.mean(((preds - target) ** 2) @ weight.T)

    # cross entropy for modes (one hot part of token) and weighted L2 for the rest
    def compute_player_token_loss(self, preds, target):
        assert preds.shape == target.shape, "preds and target must be the same shape"
        assert len(preds.shape) == 2 and preds.shape[1] == self.player_dim

        L2_loss = self.weighted_L2_loss(preds[:, :-NUM_PLAYER_MODES], target[:, :-NUM_PLAYER_MODES], self.player_token_L2_loss_weight)
        CE_loss = self.CE_loss(preds[:, -NUM_PLAYER_MODES:], target[:, -NUM_PLAYER_MODES:])
        return L2_loss + self.player_token_CE_lambda * CE_loss

    # cross entropy for weapon type (one hot part of token) and weighted L2 for the rest
    def compute_weapon_token_loss(self, preds, target):
        assert preds.shape == target.shape, "preds and target must be the same shape"
        assert len(preds.shape) and preds.shape[1:] == (NUM_WEAPON_TOKENS, self.weapon_dim)

        L2_loss = self.weighted_L2_loss(preds[:, :, :-NUM_WEAPON_TYPES], target[:, :, :-NUM_WEAPON_TYPES], 
                                     self.weapon_token_L2_loss_weight)
        CE_loss = self.CE_loss(preds[:, :, -NUM_WEAPON_TYPES:], target[:, :, -NUM_WEAPON_TYPES:])
        return L2_loss + self.weapon_token_CE_lambda * CE_loss

    # standard L2
    def compute_opponent_health_loss(self, preds, target):
        return torch.mean((preds - target) ** 2)

    # flips state from (player state, opponent state) to (opponent state, player state) along dim 1
    def flip_state(self, state):
        assert len(state.shape) == 2 and state.shape[1] == self.state_dim
        a, b = torch.split(state, 2, dim = 1)
        return torch.cat((b, a), dim = 1)