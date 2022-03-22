import numpy as np

from simple_maze import SimpleMaze


# define maze plan:
# Maze Symbols:
#   #  : wall or obstacle (the maze is surrounded by walls as default)
#   .  : empty grid cell
#   *  : general goal
#   A-Z: special goal/state
#   !  : pitfall
#   @  : agent origin(starting state)

def main():
    # run with given plan:
    # run_1()
    # run_2()
    run_3()


def run_1():
    plan = [
        '.A.B.',
        '.....',
        '.....',
        '.....',
        '.....'
    ]

    rewards = {
        'A': 10.0,
        'B': 5.0,
        'hit_wall': -1.0,
        'movement': 0.0
    }

    maze = SimpleMaze(plan=plan, rewards=rewards, action_prob=0.6, save_fig=False)
    maze.set_jump_cell(origin=(0, 1), to=(4, 1))
    maze.set_jump_cell(origin=(0, 3), to=(2, 3))

    maze.run_value_iteration(
        gamma=0.9, max_error=0.00001,
        max_iterations=1000, normal=False
    )


def run_2():
    plan = [
        '.A.B......',
        '..........',
        '..........',
        '..........',
        '..........',
        '..........',
        '..........',
        '..........',
        '..........',
        '..........'
    ]

    rewards = {
        'A': 10.0,
        'B': 5.0,
        'hit_wall': -1.0,
        'movement': 0.0
    }

    maze = SimpleMaze(plan=plan, rewards=rewards, action_prob=0.8)
    maze.set_jump_cell(origin=(0, 1), to=(4, 1))
    maze.set_jump_cell(origin=(0, 3), to=(2, 3))

    maze.run_value_iteration(
        gamma=0.9, max_error=0.00001,
        max_iterations=1000, normal=False
    )


def run_3():
    plan = [
        '...*...',
        '..!.#..',
        '..#.##.',
        '.##.A..',
        '.......',
    ]

    rewards = {
        '*': 15.0,
        '!': -3.0,
        'A': 2.0,
        'hit_wall': -1.0,
        'movement': 0.0
    }

    maze = SimpleMaze(plan=plan, rewards=rewards, action_prob=0.8, save_fig=False)
    maze.run_value_iteration(
        gamma=0.9, max_error=0.00001,
        max_iterations=1000, normal=False
    )






if __name__ == '__main__':
    main()
