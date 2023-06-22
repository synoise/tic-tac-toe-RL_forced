from typing import List

import matplotlib
from cycler import cycler
from matplotlib import pyplot as plt

from game.agents import DQNAgent, DQNAgentForced, RandomAgent, MinMaxAgent
from game.tic_tac_toe import TicTacToeGame
from game.utils import play_games

dqn_first = DQNAgent(0, pre_training_games = 50)
dqn_second = DQNAgentForced(1, pre_training_games = 50)
# agents = [dqn_first, xxx]

# results = play_games(lambda: TicTacToeGame(),  [dqn_first, RandomAgent(1)], 600, plot=True)
# dqn_first.is_learning = False

print("Against dqn_second agent:")
results2 = play_games(lambda: TicTacToeGame(), [dqn_second, RandomAgent(0)], 3000, debug=True, plot=True)

print("Against dqn_second agent:")
results1 = play_games(lambda: TicTacToeGame(), [dqn_first, RandomAgent(1)], 3000, debug=True, plot=True)




def plot_game_results(results1: List[int],results2: List[int], num_agents: int, window: int = 100):
    game_number = range(window, len(results1) + 1)
    draws = moving_count(results1, -1, window)
    winners = [moving_count(results1, i, window) for i in range(num_agents)]
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b','purple','orange','yellow'])))
    plt.plot(game_number, draws, label='Draw')
    for i, winner in enumerate(winners, start=1):
        plt.plot(game_number, winner, label='Player  ' + str(i) + ' wins')

    winners2 = [moving_count(results2, i, window) for i in range(num_agents)]
    # plt.rc('axes', prop_cycle=(cycler('color', ['y', 'y', 'y'])))
    plt.plot(game_number, draws, label='Draw Forced')
    for i, winnerx in enumerate(winners2, start=1):
        plt.plot(game_number, winnerx, label='Player Forced' + str(i) + ' wins')


    plt.ylabel(f'Rate over trailing window of {window} games')
    plt.xlabel('Game number')
    plt.xlim([0, len(results1)])
    plt.ylim([0, 1])
    ax = plt.gca()
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend(loc='best')
    plt.show()


def moving_count(items: List[int], value: int, window: int) -> List[int]:
    count = 0
    results = []
    for i in range(len(items)):
        count += -1 if i - window >= 0 and items[i - window] == value else 0
        count += 1 if items[i] == value else 0
        if i >= window - 1:
            results.append(count / window)
    return results


plot_game_results(results1,results2, 2)


# play_games(lambda: TicTacToeGame(),  [dqn_first, RandomAgent(1)], 600, plot=True)
# # dqn_first.is_learning = False