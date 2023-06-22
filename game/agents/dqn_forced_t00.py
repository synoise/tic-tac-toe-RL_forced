# import random
# from copy import deepcopy
# import numpy as np
# import tensorflow as tf
# from collections import deque
# from tensorflow import keras
#
# from game.agents import DQNAgent
#
#
# # from tensorflow.keras import layers
# # import random
# # import numpy as np
# # from tic_tac_toe import TicTacToeGame
# # from dqn_agent import DQNAgent
#
#
# def lerp(v, d):
#     return v[0] * (1 - d) + v[1] * d
#
#
# class DQNAgentForced(DQNAgent):
#     def __init__(self):
#         super().__init__()
#         self.epsilon = 0.1
#         self.alpha = 0
#         self.batch_limit = 100
#         self.counter = 0
#
#     def agentFunction(self):
#         # Agent function logic goes here
#         pass
#
#     def environmentFunction(self, game):
#             game_state = self.get_model_inputs(game)
#             # Predict action based on current game state.
#             q_values = self.model.predict(np.array([game_state]))[0]
#             assert q_values.shape == (self.n_actions,)
#             # Filter invalid actions
#             illegal_value = np.min(q_values) - 1
#             legal_actions = self.get_legal_actions(game_state)
#             return self.get_action(np.argmax(legal_actions * q_values - (legal_actions - 1) * illegal_value))
#
#     def next(self, game: TicTacToeGame) -> bool:
#         # Store previous action in action log.
#         # We have to wait for the opponent to do its move before storing any rewards
#         # that is why we commit here and in end_game().
#         self.commit_log(game, False)
#
#         if self.is_learning and (
#                 self.num_games < self.pre_training_games or
#                 random.uniform(0, 1) < lerp([self.epsilon, self.epsilon_end], max(0,
#                                                                                   self.num_games - self.pre_training_games) * self.epsilon_decay_linear)
#         ):
#             #action = 1
#             actionMove = random.choice(game.get_legal_actions(self.i_agent))
#         elif random.uniform(0, 1) < self.alpha:
#             #action = 2
#             actionMove = self.environmentFunction(game)
#         else:
#             action = 3
#
#             actionMove = self.environmentFunction(game)
#
#         # Environment logic goes here
#
#         if action == 3 or reward:
#             self.counter += 1
#             trening_data = [(input, action)]
#             if self.counter == self.batch_limit:
#                 # Train the neural network
#                 self.counter = 0
#
#             if action == 3 and (odpowiedz_sieci_neuronowej == wejście_od_nauczyciela):
#                 self.alpha = 1 - (1 - self.alpha) * beta
#
#             if reward:
#                 self.epsilon *= gamma
#
#         self.prepare_log(game, actionMove)
#         return game.next(actionMove)
#
#
#
# #
# # from . import DQNAgent
# # from ..game import Agent
# # from ..tic_tac_toe import TicTacToeGame, TicTacToeAction, GamePlayer, BOARD_SIZE, BOARD_DIM
# #
# #
# # def lerp(v, d):
# #     return v[0] * (1 - d) + v[1] * d
# #
# #
# # # action=1 akcja losowa
# # # action=2 akcja wspomagana funkcja wartości
# # # action=3 akcja wspomagana nauczycielem
# #
# # counter = 0;
# # act = 0;
# # alpha = 0
# #
# #
# # class DQNAgentForced(DQNAgent):
# #
# #     def agentFunvtion(self):
# #         w = 1
# #
# #     def evirometFunvtion(self, act, game):
# #         if act == 2:
# #             game_state = self.get_model_inputs(game)
# #             # Predict action based on current game state.
# #             q_values = self.model.predict(np.array([game_state]))[0]
# #             assert q_values.shape == (self.n_actions,)
# #             # Filter invalid actions
# #             illegal_value = np.min(q_values) - 1
# #             legal_actions = self.get_legal_actions(game_state)
# #             return self.get_action(np.argmax(legal_actions * q_values - (legal_actions - 1) * illegal_value))
# #         else:
# #             return 2
# #
# #     def next(self, game: TicTacToeGame) -> bool:
# #         # Store previous action in action log.
# #         # We have to wait for the opponent to do its move before storing any rewards
# #         # that is why we commit here and in end_game().
# #         self.commit_log(game, False)
# #
# #         if self.is_learning and (
# #                 self.num_games < self.pre_training_games or
# #                 random.uniform(0, 1) < lerp([self.epsilon, self.epsilon_end], max(0,
# #                                                                                   self.num_games - self.pre_training_games) * self.epsilon_decay_linear)
# #         ):
# #             action = 1
# #             actionMove = random.choice(game.get_legal_actions(self.i_agent))
# #         elif random.uniform(0, 1) < alpha:
# #             action = 2
# #             actionMove = self.evirometFunvtion(self, action, game)
# #         else:
# #             action = 3
# #             actionMove = self.evirometFunvtion(self, action, game)
# #
# #
# #
# #         self.prepare_log(game, actionMove)
# #         return game.next(actionMove)
