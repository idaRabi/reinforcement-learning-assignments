No network access is available here, so I can't install gymnasium in this sandbox — but the code is complete and ready to use on your machine. Let me verify it's syntactically valid at least:Here's what the environment includes:

**Grid cells:** Empty `.`, Wall `#`, Start `S`, Goal `G`, and Pit `O` — defined via a simple 2D list so you can design any map you want.

**Observation space:** A single integer — the flattened state index `row * n_cols + col`. Exactly what you need for tabular DP.

**Action space:** Discrete(4) — UP, DOWN, LEFT, RIGHT. Moving into a wall or boundary leaves the agent in place.

**Rewards:**
- Every step: `living_penalty` (default -1)
- Reaching goal: `+10` (terminates)
- Falling into pit: `-10` (terminates)

**Stochastic mode:** When `stochastic=True`, with `slip_prob` probability the agent slides in a random perpendicular direction — great for testing policy iteration under uncertainty.

**`env.P` transition table:** Pre-computed as `P[state][action] = [(prob, next_state, reward, done), ...]`. This is the classic MDP model format used by DP algorithms like Value Iteration and Policy Iteration — you just iterate over `env.P` without ever calling `step()`.

**To get started:**
```bash
pip install gymnasium
python gridworld_architect.py   # runs the demo
```

**For your DP agent**, the key object is `env.P`:
```python
env = GridworldArchitectEnv()
# Value Iteration loop skeleton:
V = {s: 0 for s in range(env.observation_space.n)}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        for prob, s_next, reward, done in env.P[s][a]:
            ...  # Bellman update here
```