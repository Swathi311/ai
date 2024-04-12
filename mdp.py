import numpy as np

class MarkovDecisionProcess:
    def __init__(self, num_states, num_actions, transition_probabilities, rewards, discount_factor=0.9, tolerance=1e-6):
        self.num_states = num_states
        self.num_actions = num_actions
        self.transition_probabilities = transition_probabilities
        self.rewards = rewards
        self.discount_factor = discount_factor
        self.tolerance = tolerance

    def value_iteration(self):
        V = np.zeros(self.num_states)
        while True:
            V_new = np.zeros(self.num_states)
            for s in range(self.num_states):
                Q = np.zeros(self.num_actions)
                for a in range(self.num_actions):
                    for s_prime in range(self.num_states):
                        Q[a] += self.transition_probabilities[s][a][s_prime] * (self.rewards[s][a][s_prime] + self.discount_factor * V[s_prime])
                V_new[s] = np.max(Q)
            if np.max(np.abs(V - V_new)) < self.tolerance:
                break
            V = V_new
        policy = np.zeros(self.num_states, dtype=int)
        for s in range(self.num_states):
            Q = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                for s_prime in range(self.num_states):
                    Q[a] += self.transition_probabilities[s][a][s_prime] * (self.rewards[s][a][s_prime] + self.discount_factor * V[s_prime])
            policy[s] = np.argmax(Q)
        return V, policy

# Example usage
num_states = 3
num_actions = 2
transition_probabilities = np.array([[[0.7, 0.3, 0.0],
                                      [0.1, 0.9, 0.0]],
                                     [[0.0, 0.8, 0.2],
                                      [0.0, 0.0, 1.0]],
                                     [[0.8, 0.1, 0.1],
                                      [0.0, 0.0, 1.0]]])
rewards = np.array([[[1, 0, 0],
                     [2, 0, 0]],
                    [[0, 0, 0],
                     [0, 0, 1]],
                    [[-1, 0, 0],
                     [0, 0, -1]]])
mdp = MarkovDecisionProcess(num_states, num_actions, transition_probabilities, rewards)
optimal_values, optimal_policy = mdp.value_iteration()
print("Optimal values:", optimal_values)
print("Optimal policy:", optimal_policy)