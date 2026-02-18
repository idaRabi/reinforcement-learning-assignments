"""
Gridworld Architect - A Gymnasium Environment for Dynamic Programming
======================================================================

Environment Description:
------------------------
The Gridworld Architect is an NxM grid where an agent navigates from a
start position to a goal while avoiding walls and collecting optional rewards.

State Space:
    - Agent's (row, col) position on the grid
    - Total states: N * M (minus wall cells)

Action Space (Discrete 4):
    0 - UP
    1 - DOWN
    2 - LEFT
    3 - RIGHT

Transition Dynamics:
    - Deterministic by default (stochastic mode available)
    - Moving into a wall or out of bounds: agent stays in place
    - Each step incurs a living penalty (default -1)
    - Reaching the goal yields +10 reward and terminates episode
    - Falling into a pit yields -10 reward and terminates episode

Observation:
    - Integer state index (row * num_cols + col)

Rendering:
    - 'human' mode: prints the grid to the terminal
    - 'rgb_array' mode: returns an RGB numpy array

Usage Example:
--------------
    import gymnasium as gym
    from gridworld_architect import GridworldArchitectEnv

    env = GridworldArchitectEnv(
        grid_map=None,        # use default map or supply custom one
        n_rows=5,
        n_cols=5,
        stochastic=False,     # set True for wind/slipping
        slip_prob=0.1,
    )
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(0)  # move UP
    env.render()
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, List, Tuple


# Cell type constants
EMPTY  = 0
WALL   = 1
START  = 2
GOAL   = 3
PIT    = 4

# Default 5x5 map legend:
#   0 = empty, 1 = wall, 2 = start, 3 = goal, 4 = pit
DEFAULT_MAP = [
    [2, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 4, 3],
]

CELL_RENDER = {
    EMPTY: ".",
    WALL:  "#",
    START: "S",
    GOAL:  "G",
    PIT:   "O",
}

# Actions
UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3
ACTION_DELTAS = {
    UP:    (-1,  0),
    DOWN:  ( 1,  0),
    LEFT:  ( 0, -1),
    RIGHT: ( 0,  1),
}
ACTION_NAMES = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}


class GridworldArchitectEnv(gym.Env):
    """
    Gridworld Architect Environment compatible with Gymnasium.

    Parameters
    ----------
    grid_map : list[list[int]] or None
        2D list of cell types. If None, a default 5×5 map is used.
    n_rows : int
        Number of rows (used only when grid_map is None and a random map is generated).
    n_cols : int
        Number of columns (same caveat as n_rows).
    stochastic : bool
        If True, with probability slip_prob the agent moves in a random
        perpendicular direction instead of the intended one.
    slip_prob : float
        Probability of slipping (only relevant when stochastic=True).
    living_penalty : float
        Reward applied at every non-terminal step (typically negative).
    goal_reward : float
        Reward received when reaching the goal cell.
    pit_penalty : float
        Reward (penalty) received when falling into a pit cell.
    max_steps : int
        Maximum steps per episode before truncation.
    render_mode : str or None
        'human' for terminal rendering, 'rgb_array' for image array.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        grid_map: Optional[List[List[int]]] = None,
        n_rows: int = 5,
        n_cols: int = 5,
        stochastic: bool = False,
        slip_prob: float = 0.1,
        living_penalty: float = -1.0,
        goal_reward: float = 10.0,
        pit_penalty: float = -10.0,
        max_steps: int = 200,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # ── Grid setup ──────────────────────────────────────────────────
        if grid_map is not None:
            self.grid = np.array(grid_map, dtype=np.int32)
        else:
            self.grid = np.array(DEFAULT_MAP, dtype=np.int32)

        self.n_rows, self.n_cols = self.grid.shape
        self._validate_grid()

        # ── Parameters ──────────────────────────────────────────────────
        self.stochastic    = stochastic
        self.slip_prob     = slip_prob
        self.living_penalty = living_penalty
        self.goal_reward   = goal_reward
        self.pit_penalty   = pit_penalty
        self.max_steps     = max_steps
        self.render_mode   = render_mode

        # ── Special cell locations ───────────────────────────────────────
        self.start_pos = self._find_cells(START)[0]
        self.goal_pos  = self._find_cells(GOAL)[0]
        self.pit_positions = set(map(tuple, self._find_cells(PIT)))

        # ── Gym spaces ──────────────────────────────────────────────────
        n_states = self.n_rows * self.n_cols
        self.observation_space = spaces.Discrete(n_states)
        self.action_space      = spaces.Discrete(4)

        # ── Internal state ───────────────────────────────────────────────
        self._agent_pos: Tuple[int, int] = self.start_pos
        self._steps: int = 0

        # ── Pre-compute transition table (for DP use) ───────────────────
        # P[state][action] = list of (prob, next_state, reward, terminated)
        self.P = self._build_transition_table()

    # ────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ────────────────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._agent_pos = self.start_pos
        self._steps = 0
        obs  = self._pos_to_state(self._agent_pos)
        info = {"agent_pos": self._agent_pos}
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        self._steps += 1
        next_pos, reward, terminated = self._transition(self._agent_pos, action)
        self._agent_pos = next_pos

        truncated = (self._steps >= self.max_steps) and not terminated
        obs  = self._pos_to_state(self._agent_pos)
        info = {
            "agent_pos":  self._agent_pos,
            "steps":      self._steps,
            "action_name": ACTION_NAMES[action],
        }

        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        # 'human' mode
        print(self._render_text())

    def close(self):
        pass

    # ────────────────────────────────────────────────────────────────────
    # Helper: state ↔ position conversion
    # ────────────────────────────────────────────────────────────────────

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.n_cols + pos[1]

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        return (state // self.n_cols, state % self.n_cols)

    # ────────────────────────────────────────────────────────────────────
    # Transition logic
    # ────────────────────────────────────────────────────────────────────

    def _transition(
        self, pos: Tuple[int, int], action: int
    ) -> Tuple[Tuple[int, int], float, bool]:
        """
        Apply action (with optional stochasticity) and return
        (next_pos, reward, terminated).
        """
        if self.stochastic and self.np_random.random() < self.slip_prob:
            # Slip to one of the two perpendicular directions
            perp = self._perpendicular_actions(action)
            action = self.np_random.choice(perp)

        dr, dc = ACTION_DELTAS[action]
        nr, nc = pos[0] + dr, pos[1] + dc

        # Boundary / wall check → stay in place
        if not self._is_walkable(nr, nc):
            nr, nc = pos

        next_pos = (nr, nc)
        terminated = False
        reward = self.living_penalty

        if next_pos == self.goal_pos:
            reward     = self.goal_reward
            terminated = True
        elif next_pos in self.pit_positions:
            reward     = self.pit_penalty
            terminated = True

        return next_pos, reward, terminated

    def _is_walkable(self, r: int, c: int) -> bool:
        if r < 0 or r >= self.n_rows or c < 0 or c >= self.n_cols:
            return False
        return self.grid[r, c] != WALL

    @staticmethod
    def _perpendicular_actions(action: int) -> List[int]:
        return {
            UP:    [LEFT, RIGHT],
            DOWN:  [LEFT, RIGHT],
            LEFT:  [UP, DOWN],
            RIGHT: [UP, DOWN],
        }[action]

    # ────────────────────────────────────────────────────────────────────
    # Transition table  P[s][a] — useful for DP algorithms
    # ────────────────────────────────────────────────────────────────────

    def _build_transition_table(self):
        """
        Build a full MDP transition table.

        Returns
        -------
        dict: P[state][action] = list of (probability, next_state, reward, terminated)
        """
        P = {}
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                pos   = (r, c)
                state = self._pos_to_state(pos)
                P[state] = {}

                # Wall cells are not reachable but include them for completeness
                for action in range(4):
                    transitions = []

                    if not self._is_walkable(r, c):
                        # Absorbing: agent can't be here
                        transitions.append((1.0, state, 0.0, False))
                        P[state][action] = transitions
                        continue

                    # Terminal absorbing states
                    if pos == self.goal_pos or pos in self.pit_positions:
                        transitions.append((1.0, state, 0.0, True))
                        P[state][action] = transitions
                        continue

                    if self.stochastic:
                        # Intended action: (1 - slip_prob)
                        # Two perpendicular actions: slip_prob / 2 each
                        perp = self._perpendicular_actions(action)
                        action_probs = {
                            action:    1.0 - self.slip_prob,
                            perp[0]:   self.slip_prob / 2,
                            perp[1]:   self.slip_prob / 2,
                        }
                    else:
                        action_probs = {action: 1.0}

                    for a, prob in action_probs.items():
                        dr, dc = ACTION_DELTAS[a]
                        nr, nc = r + dr, c + dc
                        if not self._is_walkable(nr, nc):
                            nr, nc = r, c
                        next_pos   = (nr, nc)
                        next_state = self._pos_to_state(next_pos)

                        if next_pos == self.goal_pos:
                            reward, done = self.goal_reward, True
                        elif next_pos in self.pit_positions:
                            reward, done = self.pit_penalty, True
                        else:
                            reward, done = self.living_penalty, False

                        transitions.append((prob, next_state, reward, done))

                    P[state][action] = transitions

        return P

    # ────────────────────────────────────────────────────────────────────
    # Rendering
    # ────────────────────────────────────────────────────────────────────

    def _render_text(self) -> str:
        rows = []
        rows.append("┌" + "───┬" * (self.n_cols - 1) + "───┐")
        for r in range(self.n_rows):
            row_str = "│"
            for c in range(self.n_cols):
                if (r, c) == self._agent_pos:
                    cell = "A"
                else:
                    cell = CELL_RENDER[self.grid[r, c]]
                row_str += f" {cell} │"
            rows.append(row_str)
            if r < self.n_rows - 1:
                rows.append("├" + "───┼" * (self.n_cols - 1) + "───┤")
        rows.append("└" + "───┴" * (self.n_cols - 1) + "───┘")
        legend = "  Legend: A=Agent  S=Start  G=Goal  O=Pit  #=Wall  .=Empty"
        rows.append(legend)
        rows.append(f"  Step: {self._steps}  Pos: {self._agent_pos}")
        return "\n".join(rows)

    def _render_rgb(self) -> np.ndarray:
        """Return an RGB array (H, W, 3) representing the grid."""
        cell_size = 40
        H = self.n_rows * cell_size
        W = self.n_cols * cell_size
        img = np.ones((H, W, 3), dtype=np.uint8) * 255  # white background

        COLORS = {
            EMPTY: (230, 230, 230),
            WALL:  (50,  50,  50),
            START: (100, 180, 100),
            GOAL:  (80,  160, 220),
            PIT:   (220, 80,  80),
        }
        AGENT_COLOR = (255, 200, 0)

        for r in range(self.n_rows):
            for c in range(self.n_cols):
                top  = r * cell_size
                left = c * cell_size
                color = COLORS[self.grid[r, c]]
                img[top:top + cell_size, left:left + cell_size] = color
                # Grid lines
                img[top, left:left + cell_size]  = (0, 0, 0)
                img[top:top + cell_size, left]   = (0, 0, 0)

        # Draw agent
        ar, ac = self._agent_pos
        center_r = ar * cell_size + cell_size // 2
        center_c = ac * cell_size + cell_size // 2
        radius   = cell_size // 3
        rr, cc   = np.ogrid[:H, :W]
        mask = (rr - center_r) ** 2 + (cc - center_c) ** 2 <= radius ** 2
        img[mask] = AGENT_COLOR

        return img

    # ────────────────────────────────────────────────────────────────────
    # Utilities
    # ────────────────────────────────────────────────────────────────────

    def _find_cells(self, cell_type: int) -> List[Tuple[int, int]]:
        positions = list(zip(*np.where(self.grid == cell_type)))
        return [tuple(int(x) for x in p) for p in positions]

    def _validate_grid(self):
        starts = self._find_cells(START)
        goals  = self._find_cells(GOAL)
        assert len(starts) == 1, f"Grid must have exactly one START cell (S), found {len(starts)}."
        assert len(goals)  == 1, f"Grid must have exactly one GOAL cell (G), found {len(goals)}."

    # ────────────────────────────────────────────────────────────────────
    # Convenience: print the MDP model (useful for small grids / DP)
    # ────────────────────────────────────────────────────────────────────

    def print_transition_table(self, state: Optional[int] = None):
        """Pretty-print the transition table for debugging / DP inspection."""
        states = [state] if state is not None else range(self.n_rows * self.n_cols)
        for s in states:
            pos = self._state_to_pos(s)
            print(f"\nState {s:3d}  pos={pos}  cell={CELL_RENDER[self.grid[pos]]}")
            for a in range(4):
                transitions = self.P[s][a]
                print(f"  Action {ACTION_NAMES[a]:5s}: ", end="")
                for prob, ns, r, done in transitions:
                    print(f"(p={prob:.2f} → s'={ns:3d}, r={r:+.1f}, done={done})", end="  ")
                print()


# ════════════════════════════════════════════════════════════════════════
# Gymnasium registration
# ════════════════════════════════════════════════════════════════════════

gym.register(
    id="GridworldArchitect-v0",
    entry_point="gridworld_architect:GridworldArchitectEnv",
)


# ════════════════════════════════════════════════════════════════════════
# Quick demo (run this file directly)
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Gridworld Architect — Environment Demo")
    print("=" * 60)

    env = GridworldArchitectEnv(render_mode="human", stochastic=False)
    obs, info = env.reset()

    print(f"\nObservation space : {env.observation_space}")
    print(f"Action space      : {env.action_space}")
    print(f"Grid shape        : {env.n_rows}×{env.n_cols}")
    print(f"Start position    : {env.start_pos}")
    print(f"Goal  position    : {env.goal_pos}")
    print(f"Pit   positions   : {env.pit_positions}")
    print(f"\nInitial obs (state index): {obs}\n")

    # Walk a fixed path (not optimal — just for demonstration)
    demo_actions = [DOWN, RIGHT, DOWN, RIGHT, DOWN, RIGHT, DOWN, RIGHT]

    for step, action in enumerate(demo_actions):
        print(f"\n─── Step {step + 1}: Action = {ACTION_NAMES[action]} ───")
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward={reward:+.1f}  Terminated={terminated}  Truncated={truncated}")
        if terminated or truncated:
            print("\nEpisode finished!")
            break

    print("\n" + "=" * 60)
    print("  Transition table sample (state 0):")
    print("=" * 60)
    env.print_transition_table(state=0)

    print("\n" + "=" * 60)
    print("  Custom map example (7×7 with extra pits)")
    print("=" * 60)

    custom_map = [
        [2, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 1, 4, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 4],
        [0, 0, 0, 1, 0, 0, 3],
    ]
    env2 = GridworldArchitectEnv(grid_map=custom_map, render_mode="human", stochastic=True)
    env2.reset()
    print(f"Custom env — {env2.n_rows}×{env2.n_cols} grid, stochastic=True")
