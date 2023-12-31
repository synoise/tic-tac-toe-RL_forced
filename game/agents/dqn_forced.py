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


class DQNAgentForced(DQNAgent):


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
        self._alpha = 0
        self._beta = 0.99
        self._counter = 0
        # act = 0
        self.batch_limit = pre_training_games
        self._minMaxAgent = MinMaxAgent(0)
        self._reward = 0

    def agentFunvtion(self):
        w = 1

    def get_reward(self, game: TicTacToeGame):
        if game.is_game_over():
            winners = game.get_winners()
            if len(winners) > 1:
                return self.reward_draw
            elif winners[0] == self.i_agent:
                return self.reward_win
            else:
                return self.reward_loss
        else:
            return 0


    def evirometFunvtion(self, act, game):
        # if act == 2:
            game_state = self.get_model_inputs(game)
            # Predict action based on current game state.
            q_values = self.model.predict(np.array([game_state]))[0]
            assert q_values.shape == (self.n_actions,)
            # Filter invalid actions
            illegal_value = np.min(q_values) - 1
            legal_actions = self.get_legal_actions(game_state)
            return self.get_action(np.argmax(legal_actions * q_values - (legal_actions - 1) * illegal_value))
        # else:
        #     return 2

    def next(self, game: TicTacToeGame) -> bool:
        # Store previous action in action log.
        # We have to wait for the opponent to do its move before storing any rewards
        # that is why we commit here and in end_game().
        self._reward = self.commit_log(game, False)

        if self.is_learning and (
                self.num_games < self.pre_training_games or
                random.uniform(0, 1) < lerp([self.epsilon, self.epsilon_end], max(0,
                                                                                  self.num_games - self.pre_training_games) * self.epsilon_decay_linear)
        ):
            action = 1
            actionMove = random.choice(game.get_legal_actions(self.i_agent))
        elif random.uniform(0, 1) < self._alpha:
            action = 2
            actionMove = self.evirometFunvtion(self, game)
        else:
            action = 3
            actionMove = self.evirometFunvtion(self, game)
            teacherActionMove = self._minMaxAgent.get_max_action(game, self._id )

        if action == 3 or self._reward > 0:
            self._counter += 1
            # self.prepare_log(game, actionMove)
            if self._counter == self.batch_limit:
                # Train the neural network
                self.counter = 0

            if action == 3 and (actionMove.position == teacherActionMove[0].position):
                alpha = 1 - (1 - self._alpha) * self._beta

            if self._reward:
                self.epsilon *=  self.gamma

        self.prepare_log(game, actionMove)
        return game.next(actionMove)
