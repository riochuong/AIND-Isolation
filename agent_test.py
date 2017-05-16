"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

import isolation
import game_agent
import random
import timeit
from importlib import reload
from sample_players import HumanPlayer
from sample_players import center_score


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


def improved_score(game, player):
    """The "Improved" evaluation function discussed in lecture that outputs a
    score equal to the difference in the number of moves available to the
    two players.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Retu        rns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)


class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)

    # def test_minimax(self):
    #     self.player1 = game_agent.MinimaxPlayer(
    #         score_fn=open_move_score, search_depth=3)
    #     self.player2 = game_agent.MinimaxPlayer(
    #         score_fn=open_move_score, search_depth=3)
    #     self.game = isolation.Board(self.player1, self.player2, 9, 9)
    #     print("Minimax test")
    #     print(self.game.to_string())
    #     self.game.play()

    def test_me_player_1(self):
        win_count = 0
        print("==============================================================")
        print ("Me as player 1")
        for i in range(10):
            self.player1 = game_agent.AlphaBetaPlayer(
                score_fn=open_move_score)
            self.player2 = game_agent.AlphaBetaPlayer(
                score_fn=game_agent.custom_score_2)
            # self.player2 = HumanPlayer()
    
            self.game = isolation.Board(self.player2, self.player1,7,7)

            for _ in range(2):
                move = random.choices(self.game.get_legal_moves())
                print("random move ",move)
                self.game.apply_move(move[0])

            # self.game.apply_move((2,3))
            # self.game.apply_move((1,3))

            #print("Minimax test")
            #print(self.game.to_string())
            print("Board start !")
            print(self.game.print_board())
            winner,__,outcome = self.game.play()
            #print("player 1",self.player1)
            #print("player 2",self.player2)
            #print ("Outcome", outcome)
            if (winner == self.player2):
                win_count += 1
                print("You win")
            else:
                print("You lose")

            print("ratio: ", (1 - len(self.game.get_blank_spaces()) / 81) * 100)
            print ("reason: ",outcome)

        # play me as player 2  
        print("============================================================")
        print(" Me as player 2")  
        for i in range(10):
            self.player1 = game_agent.AlphaBetaPlayer(
                score_fn=open_move_score)
            self.player2 = game_agent.AlphaBetaPlayer(
                score_fn=game_agent.custom_score_2)
            # self.player2 = HumanPlayer()
    
            self.game = isolation.Board(self.player1, self.player2, 7, 7)
            for _ in range(2):
                move = random.choices(self.game.get_legal_moves())
                print("random move ",move)
                self.game.apply_move(move[0])
            #print("Minimax test")
            #print(self.game.to_string())
            print("Board start !")
            print(self.game.print_board())
            winner,__,outcome = self.game.play()
            #print("player 1",self.player1)
            #print("player 2",self.player2)
            #print ("Outcome", outcome)
            if (winner == self.player2):
                win_count += 1
                print("You win")
            else:
                print("You lose")

            print("ratio: ", (1 - len(self.game.get_blank_spaces()) / 81) * 100)
            print ("reason: ",outcome)


        print ("===== Total Win : ",win_count," ===========")

if __name__ == '__main__':
    unittest.main()
