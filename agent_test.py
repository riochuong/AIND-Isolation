"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

import isolation
import game_agent
import timeit
from importlib import reload


def open_move_score(game, player):
    """The basic evaluation function described in lecture that outputs a score
    equal to the number of moves open for your computer player on the board.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return float(len(game.get_legal_moves(player)))


class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.player1 = game_agent.MinimaxPlayer(score_fn=open_move_score, search_depth=3)
        self.player2 = game_agent.MinimaxPlayer(score_fn=open_move_score, search_depth=3)
        self.game = isolation.Board(self.player1, self.player2,9,9)

    def test_minimax(self):
        print("Minimax test")
      

        self.game.apply_move((2,5))
        self.game.apply_move((6,8))

        print(self.game.to_string())

        self.game.play()
        

if __name__ == '__main__':
    unittest.main()
