import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(board, side, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(board, side, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (moveList, moveTree, value)
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
      value (int or float): value of the board after making the chosen move
    Input:
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      side (boolean): True if player1 (Min) plays next, otherwise False
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(board, side, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return ([ move ], { encode(*move): {} }, value)
    else:
        return ([], {}, evaluate(board))

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.
def minimax(board, side, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (moveList, moveTree, value)
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
      value (float): value of the final board in the minimax-optimal move sequence
    Input:
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      side (boolean): True if player1 (Min) plays next, otherwise False
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    if depth == 0 or chess.lib.isEnd(side, board, flags):
        return ([], {}, evaluate(board))

    if side == False:  # Maximizing player
        best_value = float('-inf')
    else:  # Minimizing player
        best_value = float('inf')

    best_sequence = None
    move_tree = {}

    for move in generateMoves(board, side, flags):
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        sequence, subtree, value = minimax(newboard, newside, newflags, depth - 1)

        move_encoded = encode(*move)
        move_tree[move_encoded] = subtree

        if side == False and value > best_value:  # Maximizing player
            best_value = value
            best_sequence = [move] + sequence
        elif side and value < best_value:  # Minimizing player
            best_value = value
            best_sequence = [move] + sequence

    if best_sequence is None:
        return ([], {}, evaluate(board))
    else:
        # The best_sequence should include both this player's move and the opponent's move(s).
        return (best_sequence, move_tree, best_value)

def alphabeta(board, side, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (moveList, moveTree, value)
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
      value (float): value of the final board in the minimax-optimal move sequence
    Input:
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      side (boolean): True if player1 (Min) plays next, otherwise False
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    if depth == 0 or chess.lib.isEnd(side, board, flags):
        return ([], {}, evaluate(board))

    best_sequence = None
    move_tree = {}

    if not side:  # Maximizing player
        best_value = float('-inf')
        for move in generateMoves(board, side, flags):
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            sequence, subtree, value = alphabeta(newboard, newside, newflags, depth - 1, alpha, beta)

            move_encoded = encode(*move)
            move_tree[move_encoded] = subtree

            if value > best_value:
                best_value = value
                best_sequence = [move] + sequence
            alpha = max(alpha, value)
            if beta <= alpha:
                break  # Beta cut-off
    else:  # Minimizing player
        best_value = float('inf')
        for move in generateMoves(board, side, flags):
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            sequence, subtree, value = alphabeta(newboard, newside, newflags, depth - 1, alpha, beta)

            move_encoded = encode(*move)
            move_tree[move_encoded] = subtree

            if value < best_value:
                best_value = value
                best_sequence = [move] + sequence
            beta = min(beta, value)
            if beta <= alpha:
                break  # Alpha cut-off

    if best_sequence is None:
        return ([], {}, evaluate(board))
    else:
        return (best_sequence, move_tree, best_value)
    
    

def stochastic(board, side, flags, depth, breadth, chooser):
    '''
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (moveList, moveTree, value)
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
      value (float): average board value of the paths for the best-scoring move
    Input:
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      side (boolean): True if player1 (Min) plays next, otherwise False
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths 
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    raise NotImplementedError("you need to write this!")
