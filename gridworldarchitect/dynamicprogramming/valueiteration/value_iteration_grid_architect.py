import numpy as np

class ValueIterationGridArchitect:
    def __init__(self, env):
        self.env = env
        self.P = env.P
        self.state_number = len(env.P)
        self.action_number = env.action_space.n
        self.action_value = np.random.uniform(low=10, high=20, size=(self.state_number, self.action_number))
        self.state_value = np.random.uniform(low=10, high=20, size=self.state_number)
        self.iteration = 0
        self.previous_state_value = None
        self.discount_factor = 0.9
        self.policy = np.ones((self.state_number, self.action_number)) / self.action_number


    def train(self):
        are_values_converged = False
        while not are_values_converged:
            self._evaluate_policy()
            are_values_converged = self._is_converged()
            print(f"Iteration: {self.iteration}")
            for state in range(self.state_number):
                print(f"State: {state} - {self.env._state_to_pos(state)}, State Value: {self.state_value[state]:.4f}, Action Value: {self.action_value[state]}, Policy: {self.policy[state]}")
            self.iteration += 1
        print("Policy Iteration Converged!")

    def play(self, state):
        return int(np.argmax(self.policy[state]))

    def _is_converged(self):
        for state in range(self.state_number):
            if self.previous_state_value is None or abs(self.state_value[state] - self.previous_state_value[state]) > 1e-4:
                return False
        return True

    def _evaluate_policy(self):
        self.previous_state_value = self.state_value.copy()
        next_state_value = self.state_value.copy()
        for state in range(self.state_number):
            for action in range(self.action_number):
                action_value = 0
                for probability, next_state, reward, done in self.P[state][action]:
                    vs_next = 0 if done else self.previous_state_value[next_state]
                    action_value += probability * (reward + self.discount_factor * vs_next)
                self.action_value[state][action] = action_value
            next_state_value[state] = np.max(self.action_value[state])
            best_action = np.argmax(self.action_value[state])
            self.policy[state] = np.eye(self.action_number)[best_action]
        self.state_value = next_state_value