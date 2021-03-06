"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import numpy as np

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    raise NotImplementedError


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    raise NotImplementedError


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    raise NotImplementedError


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            move = self.minimax(game, self.search_depth)
            #print("move from mini max:", game.get_legal_moves()[move])
            return move

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        global count
        count = 0
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        def min_value(game, depth):
            #print("min value")
            global count
            if self.time_left() < self.TIMER_THRESHOLD:
                print("min timeout")
                raise SearchTimeout()
            # check if we reached the end game
            utility = game.utility(game.inactive_player)
            if (utility):
                return utility # game is over just return 
            # check if we exhaust the depth level then we have to return the score
            if (not depth):
                count += 1
                #print("len legal moves: ",len(game.get_legal_moves()))
                #print(game.print_board())
                return self.score(game, game.inactive_player)
            val = float("inf")
            legal_moves = game.get_legal_moves(player=game.active_player)
            print("   min total legal moves", legal_moves)
            for a in legal_moves:
                #count += 1
                print("     Min node move:",a)
                val = min(val, max_value(game.forecast_move(a), depth - 1))
            return val


        def max_value(game, depth):
            #print("max")
            global count
            if self.time_left() < self.TIMER_THRESHOLD:
                print("max timeout")
                raise SearchTimeout()
            # check if we reached the end game
            utility = game.utility(game.active_player)
            if (utility):
                return utility # game is over just return 
            # check if we exhaust the depth level then we have to return the score
            if (not depth):
                count += 1
                #print("len legal moves: ",len(game.get_legal_moves()))
                #print(game.print_board())
                return self.score(game, game.active_player)
            val = float("-inf")
            #otherwise we go through all the actions to get best score
            #print("depth",depth,"legal moves", len(game.get_legal_moves(player=game.active_player)))
            legal_moves = game.get_legal_moves(player=game.active_player)
            print("   max total legal moves", legal_moves)
            for a in legal_moves:
                #count += 1
                print("     Max node move:",a)
                val = max(val, min_value(game.forecast_move(a), depth - 1))
            return val

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # search through all legal moves
        legal_moves = game.get_legal_moves(game.active_player)
        print("Player", game.active_player)
        print("Legal Moves Count",len(legal_moves))
        
        # no legal moves return (-1,-1)
        if (not len(legal_moves)):
            return (-1,-1)
        print("check each move score")
        move_scores = []
        # recursively find out what is the best moves  
        # player must be inactive player who will try to minimize the active player score  
        for move in legal_moves:
            print("Branch at move: ",move)
            move_scores.append(min_value(game.forecast_move(move), depth - 1))
        #move_scores = [min_value(game.forecast_move(move), depth - 1) for move in legal_moves]

        print("Nodes Count: ",count)
        print("Move scores", move_scores)
        print("Moves", legal_moves)
        print("Moves Selected",legal_moves[np.argmax(move_scores)])
        return legal_moves[np.argmax(move_scores)]


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            depth = 0
            #timeleft = self.time_left()
            blank_spaces = len(game.get_blank_spaces())
            print("Num blank spaces ",blank_spaces)

            # if no move left
            if (not len(game.get_legal_moves(self))):
                return best_move

            # if only one move needed 
            legal_moves = game.get_legal_moves(self)
            if (len(legal_moves) == 1):
                return legal_moves[0]

            while(depth <= (blank_spaces)):
                print("start searching at depth ",depth," time left: ", self.time_left())
                potential_move = self.alphabeta(game, depth)
                # check if potential move is a good move 
                if (potential_move != (-1,-1)):
                    best_move = potential_move
                else:
                    return best_move
                depth += 1
                # raise timeout
                if (self.time_left() < (self.TIMER_THRESHOLD )):
                    raise SearchTimeout()

        except SearchTimeout:
            print('raised search timeout')
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        print("returned best possible move", best_move)
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        def min_value(game, alpha, beta, depth):
            if self.time_left() < (self.TIMER_THRESHOLD):
                raise SearchTimeout()
            # check if the game is over
            utility = game.utility(game.inactive_player)
            if (utility):
                return utility
            #check if we reach depth limit
            if (depth <= 0):
                return self.score(game, game.inactive_player)    
            #now we need to chek all legal moves
            legal_moves = game.get_legal_moves(player=game.active_player)
            v = float("inf")
            if (len(legal_moves) == 0):
                return v
            for move in legal_moves:
                v = min(v, max_value(game.forecast_move(move),alpha,beta,depth -1))        
                # check if we can prune any branch here
                # the rest of the value must be smaller than v  
                # but v must be at least alpha to be considered in range
                # otherwise we can prune the remainings
                if (v <= alpha):
                    return v
                # update beta value 
                beta = min(v, beta)
            return v


        def max_value(game, alpha, beta, depth):
            if self.time_left() < (self.TIMER_THRESHOLD):
                raise SearchTimeout()
            # check if the game is over
            utility = game.utility(game.active_player)
            if (utility):
                return utility
            #check if we reach depth limit
            if (depth <= 0):
                return self.score(game, game.active_player)    
            # now we 
            v = float("-inf")
            legal_moves = game.get_legal_moves(player=game.active_player)
            # cut off early here 
            if (not len(legal_moves)):
                return v
            for move in legal_moves:
                v = max(v, min_value(game.forecast_move(move),alpha,beta,depth -1))
                # the remaining value of v must be at least current v
                # however, in order for v to be in range v must be smaller 
                # than beta otherwise we can prune the rest
                if (v >= beta):
                    return v
                # update alpha 
                alpha = max(v, alpha)
            return v

        # alpha-beta search body
        #alpha = float("-inf")
        #beta = float("inf")
        best_move = (-1,-1)
        # check to raise timeout
        if self.time_left() < (self.TIMER_THRESHOLD):
            raise SearchTimeout()

        # get all legal moves
        legal_moves = game.get_legal_moves(player=game.active_player)
        print("number of legal moves", len(legal_moves))
        # return bad moves 
        if (not legal_moves):
            return (-1,-1)
        # if only one moves possible 
        if (len(legal_moves) == 1):
            return legal_moves[0]
        # go through each legal moves and see if we can prune it
        for each_move in legal_moves:
            v = min_value(game.forecast_move(each_move),alpha,beta,depth - 1)
            if (v > alpha):
                alpha = v
                best_move = each_move
            if self.time_left() < (self.TIMER_THRESHOLD):
                raise SearchTimeout()
        # now we can return best move
        print("best move", best_move)
        return best_move
        
    
