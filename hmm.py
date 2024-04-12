import numpy as np

class HiddenMarkovModel:
    def __init__(self, states, observations, initial_probabilities, transition_matrix, emission_matrix):
        self.states = states
        self.observations = observations
        self.initial_probabilities = initial_probabilities
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix

    def _forward_algorithm(self, observation_sequence):
        T = len(observation_sequence)
        N = len(self.states)
        alpha = np.zeros((T, N))

        #Initialization
        alpha[0] = self.initial_probabilities * self.emission_matrix[:, self.observations.index(observation_sequence[0])]

        #recursion
        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = np.sum(alpha[t-1] * self.transition_matrix[:, j]) * self.emission_matrix[j, self.observations.index(observation_sequence[t])]

        return alpha


    def predict_sequence_probability(self, observation_sequence):
        alpha = self._forward_algorithm(observation_sequence)
        return np.sum(alpha[-1])

    def predict_state_sequence(self, observation_sequence):
        T = len(observation_sequence)
        N = len(self.states)
        delta = np.zeros((T, N))
        psi = np.zeros((T, N))

        delta[0] = self.initial_probabilities * self.emission_matrix[:, self.observations.index(observation_sequence[0])]

        for t in range(1, T):
            for j in range(N):
                delta[t, j] = np.max(delta[t-1] * self.transition_matrix[:, j]) * self.emission_matrix[j, self.observations.index(observation_sequence[t])]
                psi[t, j] = np.argmax(delta[t-1] * self.transition_matrix[:, j])

        #backtracking
        state_sequence = [np.argmax(delta[-1])]
        for t in range(T-2, -1, -1):
            state_sequence.insert(0, int(psi[t+1, state_sequence[0]]))

        return [self.states[i] for i in state_sequence]

states = ['Sunny', 'Rainy']
observations = ['Walk', 'Shop', 'Clean']
initial_probabilities = np.array([0.6, 0.4])
transition_matrix = np.array([[0.7, 0.3],
                               [0.4, 0.6]])
emission_matrix = np.array([[0.1, 0.4, 0.5],
                             [0.6, 0.3, 0.1]])

hmm = HiddenMarkovModel(states, observations, initial_probabilities, transition_matrix, emission_matrix)

observation_sequence = ['Walk', 'Shop', 'Clean']
probability = hmm.predict_sequence_probability(observation_sequence)
state_sequence = hmm.predict_state_sequence(observation_sequence)
print("Probability of observing sequence {} is {:.4f}".format(observation_sequence, probability))
print("Most likely state sequence:", state_sequence)