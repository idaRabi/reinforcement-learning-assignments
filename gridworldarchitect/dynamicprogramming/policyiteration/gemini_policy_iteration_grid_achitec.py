import numpy as np


class PolicyIterationGridArchitectByGemini:
    def __init__(self, env):
        self.env = env
        self.P = env.P
        self.state_number = len(env.P)
        self.action_number = env.action_space.n
        # Initialize values to 0.0 or small random numbers
        self.state_value = np.zeros(self.state_number)

        # Policy: start with uniform random probability
        self.policy = np.ones((self.state_number, self.action_number)) / self.action_number

        self.iteration = 0
        self.previous_state_value = np.zeros(self.state_number)

        # Gamma (discount factor) usually high (0.9 - 0.99) for pathfinding
        self.gamma = 0.9
        self.theta = 1e-4  # Convergence threshold

    def train(self):
        # We need a loop that iterates until policy is stable
        policy_stable = False
        while not policy_stable:
            self._evaluate_policy()
            policy_stable = self._improve_policy()

            self.iteration += 1
            print(f"Iteration: {self.iteration}")
            # Optional: print debug info for state 0 or others
            # print(f"State 0 Value: {self.state_value[0]:.4f}")

        print("Policy Iteration Converged!")

    def play(self, state):
        # Return index of best action
        return np.argmax(self.policy[state])

    def _evaluate_policy(self):
        """
        Policy Evaluation: Iterate Bellman expectation equation until V converges for the *current* policy.
        """
        while True:
            delta = 0
            # Create a copy so updates are synchronous (optional but cleaner)
            new_state_values = np.zeros_like(self.state_value)

            for state in range(self.state_number):
                v_s = 0
                for action in range(self.action_number):
                    action_prob = self.policy[state][action]
                    if action_prob > 0:
                        # Compute Q(s,a)
                        q_sa = 0
                        for probability, next_state, reward, done in self.P[state][action]:
                            # If done, next state value is 0
                            v_next = 0 if done else self.state_value[next_state]
                            q_sa += probability * (reward + self.gamma * v_next)

                        v_s += action_prob * q_sa

                new_state_values[state] = v_s
                delta = max(delta, abs(v_s - self.state_value[state]))

            self.state_value = new_state_values

            if delta < self.theta:
                break

    def _improve_policy(self):
        """
        Policy Improvement: Make policy greedy with respect to V.
        Returns True if policy is stable (didn't change), False otherwise.
        """
        policy_stable = True

        for state in range(self.state_number):
            old_best_action = np.argmax(self.policy[state])

            # Compute Q-values for all actions based on current V
            action_values = np.zeros(self.action_number)

            for action in range(self.action_number):
                q_sa = 0
                for probability, next_state, reward, done in self.P[state][action]:
                    v_next = 0 if done else self.state_value[next_state]
                    q_sa += probability * (reward + self.gamma * v_next)
                action_values[action] = q_sa

            # Greedily choose best action
            best_action = np.argmax(action_values)

            # Update policy to be deterministic (greedy)
            # You can keep it slightly stochastic (epsilon-soft) if desired,
            # but standard Policy Iteration usually converts to deterministic.
            new_policy_dist = np.eye(self.action_number)[best_action]
            self.policy[state] = new_policy_dist

            if best_action != old_best_action:
                policy_stable = False

        return policy_stable