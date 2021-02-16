import networkx as nx
from matplotlib import pyplot as plt
import copy
from environment.triangle_board import TriangleBoard
from environment.diamond_board import DiamondBoard
# from diamond_board import DiamondBoard
# rom triangle_board import TriangleBoard


class Environment:
    """
    Environment that allows several plays of a board game and keeps track of reward parameters
    and can vary certain other parameters between games if desired
    """

    def __init__(self, step_reward, final_reward, loser_penalty, boardsize, open_cells, board_type="Diamond"):
        """
        Initializes a gaming environment.
        :param step_reward: reward for making a single move in the game
        :param final_reward: Treated as a multiple of total accumulatable step_reward if step_reward > 0 otherwise a single number
        :param loser_penalty: Sets penalty (negative reward) for ending up in final state without winning
        :param boardsize: Decides size of board to be played inside environment
        :param open_cells: A list of cells to initialize as empty in the boards
        :param track_history: Whether to track gameplay within boards for visualization later
        """

        self.step_reward = step_reward

        self.Board_class = DiamondBoard if board_type == "Diamond" else TriangleBoard
        self.board = self.Board_class(
            boardsize, open_cells, track_history=False)
        self.final_reward = final_reward * step_reward * \
            self.board.count_pieces() if step_reward > 0 else final_reward

        self.loser_penalty = loser_penalty

    def get_state(self):
        """
        Returns current state of board game as list of lists of int
        """
        return self.board.get_state()

    def get_actions(self):
        """
        Returns possible actions from current state of board game
        """
        actions = self.board.get_legal_actions()
        return actions if len(actions) > 0 else None

    def perform_action(self, action):
        """
        Performs an action on the current board game and returns the reward it yielded
        """
        self.board.update(action)
        if self.board.is_lost():
            return self.loser_penalty
        if self.board.is_won():
            return self.final_reward
        else:
            return self.step_reward

    def game_is_finished(self):
        """
        Checks whether current boardgame is in a final state (either won or lost)
        """
        return self.board.is_finished()

    def new_game(self, track_history=False):
        """
        Resets boardgame and starts a new one
        :param track_history: Allows for replay of the game (for visualization)
        """
        self.board = self.Board_class(
            self.board.size, self.board.open_cells, track_history=track_history)
