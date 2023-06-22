import random
from copy import deepcopy
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow import keras
# from tensorflow.keras import layers

from game.agents import DQNAgent
from game.agents import MinMaxAgent
from ..game import Agent
from ..tic_tac_toe import TicTacToeGame, TicTacToeAction, GamePlayer, BOARD_SIZE, BOARD_DIM


def lerp(v, d):
    return v[0] * (1 - d) + v[1] * d


class DQNAgentForcedTEST(DQNAgent):
    def __init__(self, i_agent: int, is_learning: bool = True,
                 learning_rate=1e-3, gamma: float = 0.95,
                 epsilon: float = 0.5, epsilon_end: float = 0.001, epsilon_decay_linear: float = 1 / 2000,
                 pre_training_games: int = 500, experience_replay_batch_size=128, memory_size=10000,
                 reward_draw: float = 5., reward_win: float = 10., reward_loss: float = -10.,
                 double_dqn=True, double_dqn_n_games=1,
                 dueling_dqn=True,
                 seed=42):
        super().__init__(i_agent, is_learning, learning_rate, gamma, epsilon, epsilon_end, epsilon_decay_linear,
                         pre_training_games, experience_replay_batch_size, memory_size, reward_draw, reward_win,
                         reward_loss, double_dqn, double_dqn_n_games, dueling_dqn, seed)
        self._id = abs(1 - self.i_agent)
        self._alpha = 0.01
        self._beta = 0.99
        self._counter = 0
        # act = 0
        self.batch_limit = pre_training_games
        self._minMaxAgent = MinMaxAgent(0)
        self._reward = 0

    def get_reward(self, game: TicTacToeGame):
        if game.is_game_over():
            winners = game.get_winners()
            if len(winners) > 1:
                return self.reward_draw  # nagrodą za remis  +3
            elif winners[0] == self.i_agent:
                return self.reward_win  # nagrodą za wygrana    +10
            else:
                return self.reward_loss  # nagrodą za przegrana - kara   -10
        else:
            return 0       # nagroda cząstkowa   0

    def evirometFunction(self, game):   # wynik z sieci

        game_state = self.get_model_inputs(game)
        # Predict action based on current game state.
        q_values = self.model.predict(np.array([game_state]))[0]
        assert q_values.shape == (self.n_actions,)
        # Filter invalid actions
        illegal_value = np.min(q_values) - 1
        legal_actions = self.get_legal_actions(game_state)
        return self.get_action(np.argmax(legal_actions * q_values - (legal_actions - 1) * illegal_value))

    def next(self, game: TicTacToeGame) -> bool:
        self._reward = self.commit_log(game, False)

        if self.is_learning and (                                                                                   #  epsilon jest zmniejszane by zmniejszyć szanse wykonywania przypadkowej akcji. użyto funkcji lerp() - interpolacji liniowej
                self.num_games < self.pre_training_games or
                random.uniform(0, 1) < lerp([self.epsilon, self.epsilon_end], max(0, self.num_games - self.pre_training_games) * self.epsilon_decay_linear)
        ):
            actionMove = random.choice(game.get_legal_actions(self.i_agent))                                        # ruch losowy.

        elif random.uniform(0, 1) < self._alpha:
            actionMove = self.evirometFunction(game)                                                                # Agent wykonuje akcje w oparciu o wyjście sieci neuronowej

        else:
            actionMove = self.evirometFunction(game)                                                                # Agent wykonuje akcje w oparciu o wyjście sieci neuronowej
            teacherActionMove = self._minMaxAgent.get_max_action(game, self._id)                                    # sugerowany ruch nauczyciela MinMax

            if actionMove.position == teacherActionMove[0].position:                                                # warunek ( porównanie ruchu nauczyciela MinMax i akcji NN.
                self._alpha = 1 - (1 - self._alpha) * self._beta                                                    # alpha jest zwiększane szanse by zwiększyć szanse wykonywania wyuczonej przez siec neuronowa akcji

        self.prepare_log(game, actionMove)
        return game.next(actionMove)
