import numpy as np
import torch
from torch import nn

from policy_predictor import PolicyPredictor
from value_predictor import ValuePredictor
from conditional_analyzer import ConditionalAnalyzer
from evolver import Evolver

from model_constants import *

class AdaptiveEvolver(nn.Module):
    # TODO: add checks on init parameters
    def __init__(self, search_depth, trajectory_count, branching_number, bloom_factor):
        super().__init__()
        self.state_dim = STATE_DIM
        self.policy_dim = None
        self.strategy_dim = STRATEGY_DIM
        self.action_dim = ACTION_DIM

        # M: memory module that distills the opponent's strategy into a latent vector
        self.memory = None
        # c_t: latent vector that serves as a running tally of adversary's strategy
        self.adversary_strategy = None
        # CA: conditional analyzer that is used in both policy and value prediction but never directly called
        self.conditional_analyzer = ConditionalAnalyzer()
        # E: predicts the next state by evolving only the portions of state that belong to one of the players
        # i.e. this model approximates the one step for a single player
        # takes the current state and the action (or predicted action for adversary) as input
        self.evolver = Evolver()
        # HP: predicts a search policy for the hero's next actions given the state and the adversary's strategy vector
        self.hero_policy = PolicyPredictor(self.conditional_analyzer)
        # AP: predicts the adversary's next action given the state adn the adversary's strategy vector
        self.adversary_policy = PolicyPredictor(self.conditional_analyzer)
        # V: predicts the value of the current state over a finite time horizon h given adversary's strategy vector
        self.value_predictor = ValuePredictor(self.conditional_analyzer)
        # h: time horizon over which we want to maximize value
        self.time_horizon = TIME_HORIZON
        # k: length of state/action trajectory that is considered (not only including initial state)
        self.search_depth = search_depth
        # n: number of state/action trajectories that are kept after each round of pruning
        self.trajectory_count = trajectory_count
        # b: the number of actions that are considered from each state during the search
        self.branching_number = branching_number
        # f: the number of actions considered for the very first action is bloom factor x trajectory count
        self.bloom_factor = bloom_factor

        self.value_loss = nn.MSELoss()
        self.optimizer = None

    # produces an action a_t when given a state s_t
    @torch.no_grad() # backward pass only happens at the end of the game so no need to build computational graph here
    def forward(self, s_t: torch.Tensor):
        s_t = s_t.reshape(1, self.state_dim) # state needs to be 2d for later functions

        # TODO: revisit after writing LSTM
        # only update adversary strategy with the true current state s_t
        self.adversary_strategy = self.memory(self.adversary_strategy, s_t)

        # first level of tree search is handled separately outside of the loop since we want to setup initial actions
        candidate_next_states, candidate_actions, candidate_values = self.expand_search_tree(s_t, s_t, depth = 0)

        # grab the n state action pairs with the highest projected total values
        indices = torch.argsort(candidate_values, descending = True)[:self.trajectory_count]
        # we'll continue to evaluate these n best initial actions by rolling out state/action trajectories from them
        candidate_initial_actions = candidate_actions[indices]
        # this will always be the final state for each of the candidate state/action trajectories
        # we need to store these states to do the next level of rollout
        candidate_final_states = candidate_next_states[indices]

        # we want to keep track of what the initial action was for each of our top trajectories
        # the candidate trajectories and candidate initial actions line up perfectly at the beginning
        # but this will change as we continue rollout
        initial_action_idx_for_candidates = np.arange(self.trajectory_count)

        # subsequent levels of tree search are handled in the loop
        for i in range(1, self.search_depth):
            # during rollout, we don't care about any actions other than the first
            candidate_next_states, _, candidate_values = self.expand_search_tree(s_t, candidate_final_states, depth = i)

            # find the new best candidate trajectories; then store their initial actions and final states
            indices = torch.argsort(candidate_values, descending = True)[:self.trajectory_count]
            candidate_final_states = candidate_next_states[indices]
            # repeat this array before indexing to reflect how the search tree branches out from each trajectory
            # note that branches from the same trajectory are contiguous (hence np.repeat instead of np.tile)
            initial_action_idx_for_candidates = np.repeat(initial_action_idx_for_candidates, self.branching_number)[indices]

            # stop early if every trajectory has the same initial action since this will still be the case as we go deeper
            if len(np.unique(initial_action_idx_for_candidates)) == 1:
                return candidate_initial_actions[initial_action_idx_for_candidates[0]]

            # write down the best trajectory at the end of the loop
            if i == self.search_depth - 1:
                # TODO: consider switching to plurality vote instead of argmax
                # index before argmaxing so that you're lined up with the initial_action_idx array
                best_trajectory_index = torch.argmax(candidate_values[indices]).item()

        best_initial_action_index = initial_action_idx_for_candidates[best_trajectory_index]
        return candidate_initial_actions[best_initial_action_index]

    # expands the search tree by one level
    def expand_search_tree(self, initial_state: torch.Tensor, states: torch.Tensor, depth: int):
        assert len(states.shape) == 2 and states.shape[1] == self.state_dim # states is n x s
        assert initial_state.shape == (1, self.state_dim)
        search_policies = self.hero_policy(states, self.adversary_strategy, mode = SEARCH_MODE) # n x p
        # sample b actions per policy/state and evolve state accordingly
        b = (self.bloom_factor * self.trajectory_count) if depth == 0 else self.branching_number
        candidate_actions = self.sample_actions(search_policies, num_samples = b, for_adversary = False) # nb x a
        partial_next_states = self.evolver(states, candidate_actions, misaligned_dims = True, 
                                           is_adversary_step = False, mode = SEARCH_MODE) # E(n x s, nb x a) -> nb x s

        # predict what action the adversary will take in response and evolve state accordingly
        adversary_policies = self.adversary_policy(partial_next_states, self.adversary_strategy, 
                                                   mode = SEARCH_MODE) # nb x p
        adversary_actions = self.sample_actions(adversary_policies, num_samples = 1, for_adversary = True) # nb x a
        candidate_next_states = self.evolver(partial_next_states, adversary_actions, misaligned_dims = False, 
                                             is_adversary_step = True, mode = SEARCH_MODE) # E(nb x s, nb x a) -> nb x s

        # compute value accumulated so far by comparing each next state to the initial state
        v_accumulated = self.reward(initial_state, candidate_next_states) # (1 x s, nb x s) -> (nb,)
        # project the value of each state over the remaining time horizon
        v_projected = self.value_predictor(candidate_next_states, self.adversary_strategy, mode = SEARCH_MODE, 
                                           remaining_time_horizon = self.time_horizon - depth - 1) # nb x 1
        candidate_values = v_accumulated + v_projected.flatten() # nb
        return candidate_next_states, candidate_actions, candidate_values # (nb x s, nb x a, nb)

    # sample num_samples unique actions from the policy pi
    def sample_actions(self, pi, num_samples: int, for_adversary: bool):
        assert len(pi.shape) == 2 and pi.shape[1] == self.policy_dim
        pass

    # computes the reward by calculating the change in health differential from s_t to s_t' where t' >= t
    def reward(self, initial_state: torch.Tensor, later_states: torch.Tensor):
        return self.health_differential(later_states) - self.health_differential(initial_state)

    # computes hero health minus adversary health at some time t given state s_t
    # assumes 2d input for (potentially) multiple states
    def health_differential(self, s: torch.Tensor):
        assert len(s.shape) == 2 and s.shape[1] == self.state_dim
        return s[:, HERO_HEALTH_INDEX] - s[:, ADVERSARY_HEALTH_INDEX]

    def compute_action_log_prob(self, pi, a):
        pass

    def backward(self, hero_states, adversary_states, hero_actions, adversary_actions):
        n = hero_actions.shape[0]
        # there should be one extra hero state
        # hero state -> hero action -> adversary state -> adversary action -> hero state -> ...
        assert hero_states.shape == (n + 1, self.state_dim)
        assert adversary_states.shape == (n, self.state_dim)
        assert hero_actions.shape == (n, self.action_dim)
        assert adversary_actions.shape == (n, self.action_dim)

        # run the memory module
        # TODO: implement this
        strategies = None

        # compute NLL loss for policy predictors
        hero_pis = self.hero_policy(hero_states[:-1], strategies, BACKPROP_MODE)
        adversary_pis = self.adversary_policy(adversary_states, strategies, BACKPROP_MODE)
        hero_policy_NLL = -self.compute_action_log_prob(hero_pis, hero_actions)
        adversary_policy_NLL = -self.compute_action_log_prob(adversary_pis, adversary_actions)

        # compute loss for evolver
        hero_evolver_preds = self.evolver(hero_states[:-1], hero_actions, misaligned_dims = False, 
                                          is_adversary_step = False, mode = BACKPROP_MODE)
        adversary_evolver_preds = self.evolver(adversary_states, adversary_actions, misaligned_dims = False, 
                                               is_adversary_step = True, mode = BACKPROP_MODE)
        HE_loss, HE_loss_breakdown = self.evolver.loss(hero_evolver_preds, adversary_states, 
                                                       evolving_adversary = False)
        AE_loss, AE_loss_breakdown = self.evolver.loss(adversary_evolver_preds, hero_states[1:], 
                                                       evolving_adversary = True)

        # compute loss for value predictor
        value_preds = self.value_predictor(hero_states[:-1], strategies, mode = BACKPROP_MODE).flatten()
        # the health differential is our score, and the change in score is the reward between states
        health_diffs = self.health_differential(hero_states)
        rewards = health_diffs[1:] - health_diffs[:-1]
        # cumulatively sum rewards to get values over different time horizons
        value_targets = np.concatenate([np.cumsum(rewards[i : i + self.time_horizon - 1]) for i in range(n)])
        assert value_preds.shape == value_targets.shape, "preds and targets must align"
        value_loss = self.value_loss(value_preds, value_targets)

        # combine loss
        loss = hero_policy_NLL + adversary_policy_NLL + HE_loss + AE_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            'hero policy': hero_policy_NLL,
            'adversary policy': adversary_policy_NLL,
            'hero weapon': HE_loss_breakdown.weapon_token,
            'hero player': HE_loss_breakdown.player_token,
            'hero opponent health': HE_loss_breakdown.opponent_health,
            'adversary weapon': AE_loss_breakdown.weapon_token,
            'adversary player': AE_loss_breakdown.player_token,
            'adversary opponent health': AE_loss_breakdown.opponent_health,
            'value': value_loss
        }