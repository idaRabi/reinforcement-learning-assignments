import numpy as np

class PolicyIterationGridArchitect:
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
        # with the below policy initialization, the algorithm doesn't work because its sum is not 1.
        # self.policy = np.random.uniform(low=0, high=1, size=(self.state_number, self.action_number))


    def train(self):
        is_policy_stable = False
        while not is_policy_stable:
            self._evaluate_policy()
            is_policy_stable = self._improve_policy()
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
        while not self._is_converged():
            self.previous_state_value = self.state_value.copy()
            next_state_value = self.state_value.copy()
            for state in range(self.state_number):
                state_value = 0
                for action in range(self.action_number):
                    action_value = 0
                    for probability, next_state, reward, done in self.P[state][action]:
                        vs_next = 0 if done else self.previous_state_value[next_state]
                        action_value += probability * (reward + self.discount_factor * vs_next)
                    action_value = action_value * self.policy[state][action]
                    state_value += action_value
                    self.action_value[state][action] = action_value
                next_state_value[state] = state_value
            self.state_value = next_state_value

    def _improve_policy(self):
        old_best_policy = self.policy.copy()
        for state in range(self.state_number):
            best_action = np.argmax(self.action_value[state])
            for action in range(self.action_number):
                if action == best_action:
                    self.policy[state][best_action] = 0.95
                    # self.policy[state][best_action] = 1
                else:
                    self.policy[state][action] = 0.05 / (self.action_number - 1)
                    # self.policy[state][action] = 0
        return np.array_equal(self.policy, old_best_policy)