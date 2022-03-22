import string
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from texttable import Texttable

np.set_printoptions(precision=5, suppress=True)


class SimpleMaze():
    """
    Create a simple grid maze with walls, goals and pitfalls.

    Maze Symbols:
        #  : wall or obstacle (the maze has walls around it as default)
        .  : empty grid cell
        *  : general goal
        A-Z: special goal/state
        !  : pitfall
        @  : agent origin(starting state)

    Actions:
        N: North, S: South, W: West, E: East

    Rewards should be a dictionary with these symbols:
        *       : goal reward,
        movement: negative reward for each move,
        hit_wall: negative reward for hitting the wall,
        !       : pitfall punishment
        A-Z     : special rewards
    """

    def __init__(
        self, plan: List[str], rewards: Dict[str, int] = {'*': 10},
        action_prob=0.8, terminal_markers='*', save_fig=False
    ):

        self.plan = np.array([list(row) for row in plan], dtype=str)
        self.shape = self.plan.shape
        self.rewards = rewards
        self.terminal_markers = terminal_markers
        self.save_fig = save_fig
        # default four actions defined by movement deltas:
        # as default each action may fail into two prependicular actions with equal prob.
        self.actions = {
            'N': np.array([-1, 0]),
            'S': np.array([1, 0]),
            'W': np.array([0, -1]),
            'E': np.array([0, 1])
        }
        self.action_prob = action_prob
        self.action_noise_prob = (1.0 - self.action_prob) / 2
        self.policy_symbols = {
            'N': '↑',
            'S': '↓',
            'W': '←',
            'E': '→',
            'J': '↷',
            '#': '⬣',
            '*': '★',
            '!': '☢'
        }

        self.end_at_terminal = True
        self.num_states: int = self.shape[0] * self.shape[1]
        self.current_state: tuple = (0, 0)
        self._set_agent_origin()
        self._jump_cells: Dict[tuple, tuple] = {}
        self._create_plot()
        self.fig_output = Path('./fig_output')
        if self.save_fig:
            self.fig_output.mkdir(parents=True, exist_ok=True)

    def set_jump_cell(self, origin: tuple, to: tuple):
        self._jump_cells[origin] = to

    def is_terminal(self, state):
        # Check if the given state is a terminal state.
        return self.plan[state] in self.terminal_markers

    def run_value_iteration(self, gamma=0.9, max_error=0.00001, max_iterations=1000, normal=False) -> tuple:
        """
        Performs value iteration algorithm on this maze and returns state utilities and policy.
        """
        symbols = '!*#'  # state's symbols with fixed/no reward.
        err_coeff = (1 - gamma) / gamma

        discounted_rewards = np.zeros(self.shape)
        policy = np.empty(self.shape, dtype=np.str)

        current_utilities = self._get_utilities_at_0()
        updating_utilities = current_utilities.copy()
        print('Utilities at 0:')
        self._print_table(current_utilities)
        self._plot_heatmap(current_utilities, policy)

        print('\nbeginning Value Iteration:')
        it = 1
        while it <= max_iterations:
            print('\n  itertion #{0}:'.format(it))
            delta = 0.
            # iterate over all states.
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    state = (i, j)

                    # if state is a terminal state(goal or pitfall) or obstacle then don't update it:
                    if self.plan[state] in symbols:
                        policy[state] = self.policy_symbols.get(
                            self.plan[state], '')
                        continue

                    # if this state is a jump state then all actions jump into destination cell.
                    # so expected reward comes from destination cell with probability of 1.0 .
                    if self._jump_cells.get(state):
                        state_actions = ['J']  # jump
                        destination = self._jump_cells[state]
                        one_step_ahead_utilities = gamma * \
                            current_utilities[destination]
                    else:
                        # in case of hitting the wall intentionally is not an option,
                        # get possible actions for this state:
                        # state_actions = self._get_possible_actions(state)

                        state_actions = ['N', 'S', 'W', 'E']
                        # calculate state expected utility for each action:
                        one_step_ahead_utilities = np.zeros(len(state_actions))
                        for idx, ac in enumerate(state_actions):
                            one_step_ahead_utilities[idx] = self._get_expected_utility(
                                state, ac, gamma, current_utilities
                            )

                    # get max over actions and calc discounted reward.
                    discounted_rewards[state] = np.max(one_step_ahead_utilities)

                    # get best action for this state.
                    best_action = state_actions[np.argmax(one_step_ahead_utilities)]
                    policy[state] = self.policy_symbols.get(best_action, best_action)

                    if normal:
                        # normalizing discounted rewards:
                        _mean = np.mean(discounted_rewards, keepdims=True)
                        _std = np.std(discounted_rewards, keepdims=True) + 1e-10
                        discounted_rewards = (discounted_rewards - _mean) / _std

                    # finally update utility of this state:
                    updating_utilities[state] = self.rewards.get(
                        self.plan[state], 0.) + discounted_rewards[state]
                    delta = np.maximum(delta, np.abs(
                        current_utilities[state] - updating_utilities[state]))

            # update current utilities with new ones:
            current_utilities = updating_utilities.copy()

            print('  Utilities:')
            self._print_table(current_utilities)
            self._plot_heatmap(current_utilities, policy)
            if self.save_fig:
                self._fig.savefig(self.fig_output.joinpath(f'{it: 05}.png'))

            # check for convergence:
            if delta < max_error * err_coeff:
                break
            it += 1
        # end of iterations

        info = 'movement: {0},  hit-wall: {1},  action prob.: {2},  gamma: {3},  #iterations: {4}'.format(
            self.rewards.get('movement', 0), self.rewards.get('hit_wall', 0), self.action_prob, gamma, it)
        self._ax.annotate(
            info, xy=(105, 7), xytext=(0, 0),
            xycoords=('figure pixels', 'figure pixels'),
            textcoords='offset pixels', size=12, ha='left', va='bottom'
        )
        plt.pause(0.1)
        if self.save_fig:
            self._fig.savefig(self.fig_output.joinpath(f'{it: 05}.png'))

        print('\nPolicy:')
        self._print_table(policy)
        input('\n\nPress Enter to end...')

        return current_utilities, policy

    def _get_utilities_at_0(self) -> np.ndarray:
        utilities = np.zeros(self.shape)

        # fill rewards for states with A-Z symbols:
        symbols = list(string.ascii_uppercase)
        # gives a tuple(x,y) of numpy array of indexes.
        AZ: tuple = np.where(np.isin(self.plan, symbols))
        if AZ[0].size > 0:
            for idx, symbol in enumerate(self.plan[AZ]):
                utilities[AZ[0][idx], AZ[1][idx]] = self.rewards.get(symbol, 0.)

        # goal states
        goals = np.where(self.plan == '*')
        if goals[0].size > 0:
            utilities[goals] = self.rewards.get('*', 0.)

        # pitfall states
        pits = np.where(self.plan == '!')
        if pits[0].size > 0:
            utilities[pits] = self.rewards.get('!', 0.)

        return utilities

    def _get_possible_actions(self, state) -> List[str]:
        _actions: List[str] = []
        for action, delta in self.actions.items():
            if self._is_inside_maze(state + delta):
                _actions += action

        return _actions

    def _is_inside_maze(self, state: tuple) -> bool:
        return (0 <= state[0] < self.shape[0]) and (0 <= state[1] < self.shape[1])

    def _get_expected_utility(self, state: tuple, action: str, gamma: float, current_utilities: np.ndarray) -> float:
        """
        Calculate expected utility of given state and action(one-step-ahead).
            U_i+1(s) = Σ_s' P(s'|s,a) (R + γ.U_i(s'))
        """
        state_expected_utility = 0.

        # original action plus noisy ones.
        if action in 'NS':
            actions = [action, 'E', 'W']
        else:
            actions = [action, 'N', 'S']

        actions_probs = [self.action_prob, self.action_noise_prob, self.action_noise_prob]

        for idx, ac in enumerate(actions):
            next_state = tuple(state + self.actions[ac])

            # cost of movement
            R = self.rewards.get('movement', 0.)

            # if hit the wall or obstacles
            if not self._is_inside_maze(next_state) or self.plan[next_state] == '#':
                R += self.rewards.get('hit_wall', 0.)
                next_state = state  # can not move.

            state_expected_utility += actions_probs[idx] * \
                (R + gamma * current_utilities[next_state])

        return state_expected_utility

    def _set_agent_origin(self):
        # set starting state to '@' position in plan.
        at: tuple = np.where(self.plan == '@')  # gives a tuple of numpy array.
        if at[0].size > 0:
            self.current_state = self.plan[at][0]

    def _print_table(self, data: np.ndarray):
        num_cols = data.shape[1]
        table = Texttable(max_width=110)
        table.set_precision(5)
        table.set_cols_align(['c' for i in range(num_cols)])
        table.add_rows(data, header=False)
        print(table.draw())

    def _create_plot(self):
        self._fig = plt.figure(figsize=(8, 7))
        self._fig.canvas.set_window_title("Maze Utility and Policy Plot")
        self._ax: plt.Axes = self._fig.add_subplot(1, 1, 1)
        self._fig.tight_layout(h_pad=0.0, pad=0.0, rect=[0., 0.04, 1., 0.95])

        plt.show(block=False)

    def _plot_heatmap(self, utilities: np.ndarray, policy: np.ndarray):
        self._ax.clear()
        im = self._ax.imshow(utilities, cmap='YlOrBr_r')

        # We want to show all ticks...
        self._ax.set_xticks(np.arange(utilities.shape[1]))
        self._ax.set_yticks(np.arange(utilities.shape[0]))

        # Let the horizontal axes labeling appear on top.
        self._ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        self._ax.tick_params(which="minor", bottom=False, left=False)

        self._ax.set_xticks(np.arange(utilities.shape[1] + 1) - .5, minor=True)
        self._ax.set_yticks(np.arange(utilities.shape[0] + 1) - .5, minor=True)
        self._ax.grid(which="minor", color="black", linestyle='-', linewidth=2)

        self._annotate_heatmap(im, utilities, policy)

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def _annotate_heatmap(self, im, utilities, policy):
        threshold = (im.norm(utilities.max()) / 2) - 0.07
        colors = ['white', 'black']
        font_size = 16 - 10 * (np.max(self.shape) / 14)
        for i in range(utilities.shape[0]):
            for j in range(utilities.shape[1]):
                text = im.axes.text(j, i, '{0:.5f}'.format(utilities[i, j]),
                                    horizontalalignment='center', verticalalignment='center',
                                    position=(j, i - 0.2),
                                    color=colors[im.norm(
                                        utilities[i, j]) > threshold],
                                    fontsize=font_size, fontfamily='monospace', fontweight=700)

                symbol = self.plan[i, j] if self.plan[i, j] in string.ascii_uppercase else ''

                text = im.axes.text(j, i, '{0}{1}'.format(symbol, policy[i, j]),
                                    horizontalalignment='center', verticalalignment='center',
                                    position=(j, i + 0.22),
                                    color='white', fontsize=font_size * 2, fontfamily='sans',
                                    bbox=dict(facecolor='grey', alpha=0.55, boxstyle='round, pad=0.17'))
