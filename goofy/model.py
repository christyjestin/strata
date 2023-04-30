import numpy as np
import torch
from torch import nn

# TODO: double check imports and add type hints

class AdaptiveEvolver(nn.Module):
    # TODO: add checks on init parameters
    def __init__(self):
        super().__init__()
        self.state_dim = None
        self.policy_dim = None
        self.strategy_dim = None

        # M: memory module that distills the opponent's strategy into a latent vector
        self.memory = None
        # c_t: latent vector that serves as a running tally of adversary's strategy
        self.adversary_strategy = None
        # AE: predicts the partial next state by evolving only the portions of state that belong to adversary or
        # are shared; takes the current state and the latent vector for adversary's strategy as input
        self.adversary_evolver = None
        # PP: predicts a search policy for next actions given the partial next state and the adversary's strategy
        self.policy_predictor = None
        # HE: predicts the full next state by evolving only the portions of state that belong to the hero or are shared;
        # takes the partial next state and the chosen action as input
        self.hero_evolver = None
        # V: predicts the value of the current state over a finite time horizon h given adversary's strategy
        self.value_predictor = None
        # h: time horizon over which we want to maximize value
        self.time_horizon = None
        # k: length of state/action trajectory that is considered (not only including initial state)
        self.search_depth = None
        # n: number of state/action trajectories that are kept after each round of pruning
        self.trajectory_count = None
        # b: the number of actions that are considered from each state during the search
        self.branching_number = None
        # f: the number of actions considered for the very first action is bloom factor x trajectory count
        self.bloom_factor = None

        self.adversary_evolver_loss = None
        self.hero_evolver_loss = None
        self.value_predictor_loss = None

        # need a lambda value for each loss that involves the strategy vector since this means that the loss backprops
        # to the memory module and we need a way to weight the different losses that are relevant to the memory module
        self.adversary_evolver_lambda = None
        self.value_predictor_lambda = None
        self.policy_predictor_lambda = None

    # TODO: ensure that computational graph isn't being built; in particular that the no grad decorator also
    # applies to the nested forward calls
    # produces an action a_t when given a state s_t
    @torch.no_grad() # backward pass only happens at the end of the game so no need to build computational graph here
    def forward(self, s_t: torch.Tensor):
        assert s_t.shape == (self.state_dim,)

        # only update adversary strategy with the true current state s_t
        self.adversary_strategy = self.memory(self.adversary_strategy, s_t)

        # first level of tree search is handled separately outside of the loop
        # partially evolve state to produce a search policy
        partial_next_s = self.adversary_evolver(self.adversary_strategy, s_t)
        search_policy = self.policy_predictor(self.adversary_strategy, partial_next_s)

        # grab b actions from search policy and compute projected values for all b
        args = (search_policy, self.trajectory_count * self.bloom_factor, partial_next_s, s_t, self.time_horizon - 1)
        candidate_next_states, candidate_actions, candidate_values = self.sample_and_compute_values(*args)

        # grab the n state action pairs with the highest projected total values
        indices = torch.argsort(candidate_values, descending = True)[:self.trajectory_count]
        candidate_initial_actions = candidate_actions[indices]
        candidate_states = candidate_next_states[indices]

        # we want to keep track of what the initial action was for each of our top trajectories
        # at the beginning, the candidate trajectories and candidate initial actions line up perfectly
        initial_action_idx_for_candidates = np.arange(self.trajectory_count)

        # subsequent levels of tree search are handled in the loop
        for i in range(1, self.search_depth):
            candidate_next_states = torch.empty((self.trajectory_count, self.branching_number, self.state_dim))
            candidate_values = torch.empty((self.trajectory_count, self.branching_number))

            # TODO: consider vectorizing this loop or at least the adversary evolver and policy predictor steps
            # generate b new nodes for each of the candidate trajectories
            for state_index in range(self.trajectory_count):
                # same as first level of tree search but the value of b and the remaining time horizon have changed
                partial_next_s = self.adversary_evolver(self.adversary_strategy, candidate_states[state_index])
                search_policy = self.policy_predictor(self.adversary_strategy, partial_next_s)
                args = (search_policy, self.branching_number, partial_next_s, s_t, self.time_horizon - 1 - i)
                # we don't need to track actions anymore since these are no longer the initial actions
                candidate_next_states[state_index], _, candidate_values[state_index] = self.sample_and_compute_values(*args)

            # find the best new candidates and store their initial actions and their next states
            indices = torch.argsort(candidate_values.flatten(), descending = True)[:self.trajectory_count]
            candidate_states = candidate_next_states.flatten(end_dim = 1)[indices]
            initial_action_idx_for_candidates = np.repeat(initial_action_idx_for_candidates, self.branching_number)[indices]

            # stop early if every trajectory has the same initial action since this will remain the case as we go deeper
            if len(np.unique(initial_action_idx_for_candidates)) == 1:
                return candidate_initial_actions[initial_action_idx_for_candidates[0]]

            # write down the best trajectory at the end of the loop
            if i == self.search_depth - 1:
                best_trajectory_index = torch.argmax(candidate_values.flatten()[indices])

        best_initial_action_index = initial_action_idx_for_candidates[best_trajectory_index]
        return candidate_initial_actions[best_initial_action_index]

    # sample b actions from the search policy pi and compute their total projected values
    def sample_and_compute_values(self, pi, b, partial_next_s, initial_s, remaining_time_horizon):
        candidate_actions = self.sample(pi, b) # shape is b x a
        candidate_next_states = self.hero_evolver(partial_next_s, candidate_actions) # shape is b x s
        # compute value accumulated so far by comparing each next state to the initial state
        v_accumulated = self.reward(initial_s, candidate_next_states) # shape is (b,)
        # project the value of each state over the remaining time horizon; shape is (b,)
        v_projected = self.value_predictor(self.adversary_strategy, candidate_next_states, remaining_time_horizon)
        candidate_values = v_accumulated + v_projected
        return candidate_next_states, candidate_actions, candidate_values

    # sample num_samples unique actions from the policy pi
    def sample(self, pi, num_samples):
        assert pi.shape == (self.policy_dim,)
        assert isinstance(num_samples, int)
        pass

    # computes the reward by calculating the change in health differential from s_t to s_t' where t' >= t
    def reward(self, initial_state, next_states):
        return self.health_differential(next_states) - self.health_differential(initial_state.unsqueeze(0))

    # computes hero health - adversary health at some time t given state s_t
    # assumes 2d input for (potentially) multiple states
    def health_differential(self, s):
        assert s.shape[1] == self.state_dim and len(s.shape) == 2
        pass

    def batch_value_prediction(self, strategy_history, state_history):
        # TODO: in likely event of batched refactoring, move asserts to backward function
        n = strategy_history.shape[0]
        assert strategy_history.shape == (n, self.strategy_dim)
        assert state_history.shape == (n, self.state_dim)
        v_projected = torch.empty((n,), requires_grad = True)
        # TODO: make a decision about whether to vectorize this loop
        for i in range(n):
            v_projected[i] = self.value_predictor(strategy_history[i], state_history[i:i+1], remaining_time_horizon = n - i)
        return v_projected

    def compute_log_prob(self, pi, a):
        pass

    def backward(self, s, partial_next_s, a, next_s, reward_history, strategy_history, state_history):
        # TODO: refactor for backprop on full game instead of single step
        # TODO: add asserts
        # TODO: review all places where strategy enters the equation
        next_s_pred = self.hero_evolver(partial_next_s, torch.unsqueeze(a, 0))
        partial_next_s_pred = self.adversary_evolver(self.adversary_strategy, s)
        v_pred = self.batch_value_prediction(strategy_history, state_history)
        search_policy = self.policy_predictor(self.adversary_strategy, partial_next_s)

        # cumulatively sum the rewards to find the value over increasing longer time horizons
        v = torch.tensor(np.cumsum(np.flip(reward_history)))

        hero_loss = self.hero_evolver_loss(next_s_pred, next_s)
        # multiply all losses that involve strategy (and in turn the memory module) by corresponding lambda
        adversary_loss = self.adversary_evolver_lambda * self.adversary_evolver(partial_next_s_pred, partial_next_s)
        # TODO: consider weighting longer time horizon predictions less in the loss because it's harder to see far out
        # into the future; if you do this, the lower weights should only apply to the very longest time horizons
        value_loss = self.value_predictor_lambda * self.value_predictor_loss(v_pred, v)
        policy_loss = self.policy_predictor_lambda * -self.compute_log_prob(search_policy, a)

        hero_loss.backward()
        adversary_loss.backward()
        value_loss.backward()
        policy_loss.backward()
