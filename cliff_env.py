import numpy as np


class CliffWalkingEnv:
    """Cliff Walking environment (Sutton & Barto Example 6.6).

    Grid of shape (n_rows, n_cols).
    - Start: bottom-left  (n_rows-1, 0)
    - Goal:  bottom-right (n_rows-1, n_cols-1)
    - Cliff: bottom row between Start and Goal

    Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    """

    def __init__(self, n_rows=4, n_cols=12):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.start = (n_rows - 1, 0)
        self.goal = (n_rows - 1, n_cols - 1)
        self.cliff = [(n_rows - 1, c) for c in range(1, n_cols - 1)]
        self.n_states = n_rows * n_cols
        self.n_actions = 4
        self.state = self.pos_to_state(self.start)

    # ── coordinate helpers ──────────────────────────────────────────────

    def pos_to_state(self, pos):
        r, c = pos
        return r * self.n_cols + c

    def state_to_pos(self, s):
        return (s // self.n_cols, s % self.n_cols)

    # ── standard env interface ──────────────────────────────────────────

    def reset(self):
        """Reset to start state and return start state id."""
        self.state = self.pos_to_state(self.start)
        return self.state

    def step(self, action):
        """Apply action, return (next_state, reward, done, info)."""
        r, c = self.state_to_pos(self.state)
        if action == 0:    # UP
            r = max(r - 1, 0)
        elif action == 1:  # RIGHT
            c = min(c + 1, self.n_cols - 1)
        elif action == 2:  # DOWN
            r = min(r + 1, self.n_rows - 1)
        elif action == 3:  # LEFT
            c = max(c - 1, 0)

        new_pos = (r, c)

        if new_pos in self.cliff:
            # Fell off cliff: -100 penalty and reset to Start
            self.state = self.pos_to_state(self.start)
            return self.state, -100, False, {}

        if new_pos == self.goal:
            # Reached Goal: episode ends
            self.state = self.pos_to_state(new_pos)
            return self.state, -1, True, {}

        # Normal step
        self.state = self.pos_to_state(new_pos)
        return self.state, -1, False, {}

    # ── visualisation ───────────────────────────────────────────────────

    def render_policy_with_path(self, Q, file=None, algorithm='Q-learning'):
        """Visualize the greedy policy with arrows and the optimal trajectory.

        The optimal path is traced greedily (argmax Q) from Start until Goal
        (or a loop / max-step limit is hit).  The visited row(s) are highlighted
        with a dashed blue rectangle, mimicking Sutton & Barto Fig. 6.4.

        Args:
            Q (np.ndarray): Q-table, shape (n_states, n_actions)
            file (str | None): save path; if None, calls plt.show()
            algorithm (str): displayed in the figure title
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib import rcParams
        rcParams['font.family'] = 'DejaVu Sans'
        rcParams['axes.unicode_minus'] = False

        # Arrow direction vectors (in display coords after invert_yaxis)
        # action 0=UP → dy negative (upward on screen), 1=RIGHT, 2=DOWN, 3=LEFT
        ARROW = {0: (0, -0.30), 1: (0.30, 0), 2: (0, 0.30), 3: (-0.30, 0)}

        policy = np.argmax(Q, axis=1).reshape(self.n_rows, self.n_cols)
        cliff_set = set(self.cliff)

        fig, ax = plt.subplots(figsize=(self.n_cols * 0.85, self.n_rows * 0.9 + 0.5))
        ax.set_xlim(0, self.n_cols)
        ax.set_ylim(0, self.n_rows)
        ax.invert_yaxis()   # row 0 at top, row n_rows-1 at bottom
        ax.set_aspect('equal')

        # ── 1. Cell background colours ────────────────────────────────
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                fc = '#BDE0F7' if (r, c) in cliff_set else 'white'
                ax.add_patch(patches.Rectangle(
                    (c, r), 1, 1,
                    linewidth=0.8, edgecolor='#aaaaaa', facecolor=fc
                ))

        # ── 2. Trace greedy path from Start ───────────────────────────
        path_cells = []
        state = self.pos_to_state(self.start)
        visited = set()
        self.reset()
        max_trace = self.n_rows * self.n_cols * 3
        for _ in range(max_trace):
            r, c = self.state_to_pos(state)
            path_cells.append((r, c))
            if (r, c) == self.goal:
                break
            if state in visited:
                break           # loop — stop gracefully
            visited.add(state)
            action = int(np.argmax(Q[state]))
            state, _, done, _ = self.step(action)
            if done:
                nr, nc = self.state_to_pos(state)
                path_cells.append((nr, nc))
                break
        self.reset()            # restore env state

        # ── 3. Dashed blue rectangle around the path row band ─────────
        path_rows = sorted({r for r, c in path_cells
                            if (r, c) not in cliff_set
                            and (r, c) != self.start
                            and (r, c) != self.goal})
        if path_rows:
            r_min, r_max = path_rows[0], path_rows[-1]
            ax.add_patch(patches.FancyBboxPatch(
                (0, r_min), self.n_cols, (r_max - r_min + 1),
                boxstyle='square,pad=0',
                linewidth=2.2, edgecolor='#1565C0',
                linestyle='--', facecolor='none', zorder=4
            ))

        # ── 4. Policy arrows ──────────────────────────────────────────
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if (r, c) in cliff_set or (r, c) == self.goal:
                    continue
                a = policy[r, c]
                dx, dy = ARROW[a]
                cx, cy = c + 0.5, r + 0.5
                ax.annotate(
                    '', xy=(cx + dx, cy + dy), xytext=(cx - dx, cy - dy),
                    arrowprops=dict(arrowstyle='->', color='black',
                                   lw=1.3, mutation_scale=11),
                    zorder=3
                )

        # ── 5. Optimal path line ──────────────────────────────────────
        if len(path_cells) > 1:
            xs = [c + 0.5 for _, c in path_cells]
            ys = [r + 0.5 for r, _ in path_cells]
            ax.plot(xs, ys, color='#1565C0', linewidth=2.2,
                    zorder=5, alpha=0.80, label='Optimal path')

        # ── 6. Start / Goal / Cliff labels ────────────────────────────
        sc, sr = self.start[1], self.start[0]
        gc, gr = self.goal[1], self.goal[0]

        # Start: up-arrow + text
        ax.annotate('', xy=(sc + 0.5, sr + 0.22), xytext=(sc + 0.5, sr + 0.78),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.8,
                                   mutation_scale=13), zorder=6)
        ax.text(sc + 0.5, sr + 0.85, 'Start', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold')

        # Goal: text only
        ax.text(gc + 0.5, gr + 0.5, 'Goal', ha='center', va='center',
                fontsize=8.5, fontweight='bold')

        # Cliff label
        cliff_cs = [c for _, c in self.cliff]
        cliff_mid = (min(cliff_cs) + max(cliff_cs)) / 2 + 0.5
        cliff_r = self.cliff[0][0]
        ax.text(cliff_mid, cliff_r + 0.5, 'Cliff', ha='center', va='center',
                fontsize=11, fontweight='bold', color='#1565C0', alpha=0.70)

        # ── 7. Axes formatting ─────────────────────────────────────────
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(1.0)
            sp.set_edgecolor('#888888')
        ax.set_title(f'{algorithm} policy', fontsize=13, fontweight='bold', pad=8)

        plt.tight_layout()
        if file:
            plt.savefig(file, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
