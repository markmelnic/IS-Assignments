import sys
import datetime
import numpy as np
from numpy import random
from collections import deque


class GameState:
    """Every node in the search tree corresponds to a game state. Every edge in the search tree, represent a card played.
    In every game state there is
    - the cards from the left hand (numpy array filled with integers that represent cards)
    - the cards from the right hand (numpy array filled with integers that represent cards)
    - the cards that have been played (stack data structure filled with integers that represent cards).
    Consider the stack as a literal pile of cards,
    the card on top of the stack is the one for which the card from either the left or right hand must match in either suit or value!"""
    def __init__(self, leftHand, rightHand, playLeft=True, initialStateTest=False):
        self.leftHand = leftHand #left hand is a list
        self.rightHand = rightHand #right hand is a list
        self.playLeft = playLeft #playleft is a boolean value to (help) determine from which hand we must lay a card (More specifically, it shows wether we play from the left hand). When left has laid a card, this value will be switched to False, when right has laid a card this value will be switched to True
        self.initialStateTest = initialStateTest #this is implemented to make it easy to check if the state is the initial state (where every move is a valid move). It is used in class 'Problem', at 'actions'.


    def printState(self):
        print("------")
        print("GameState: Printing state: ")
        print("Left hand: {}".format(self.leftHand))
        print("Right hand: {}".format(self.rightHand))
        print("Do we play from left hand to get to next state? {}".format(self.playLeft))
        print("------")

representation = """
        h  d  s  c
ace  [[ 0  1  2  3]  h = hearts
ten   [ 4  5  6  7]  d = diamonds
king  [ 8  9 10 11]  s = spades
queen [12 13 14 15]  c = clubs
jack  [16 17 18 19]]

For example: '10' is 'king of spades'
"""

class Problem:
    """The class for a formal problem."""

    def __init__(self, initial):
        """The constructor specifies the initial state"""
        self.initial = initial

    def actions(self, state, solution):
        """Return a list of actions that can be executed in the given state.
        In this case, the list of cards for which there is a legal move, is returned"""

        possibleMoves = []
        if state.initialStateTest: #By default it is set to false, therefore it is only True if we explicitly set it to be (when defining initial state). At the initial state every move is a valid move.
            return state.leftHand #All cards in the left hand are returned as valid moves (without checking, because there is nothing to check against)
        elif state.playLeft: #If it is not the initial state, we can play the game as we normally would (with checking for valid moves!). Here we check if we have to play from the left hand.
            for i in state.leftHand: #iterate through every card in left hand
                if valid_move(solution[-1], i): #check if there is a valid move for that card (i)
                    possibleMoves.append(i) #if the check passes, add the card to the list of possibly played cards.
            return possibleMoves #return the list of possible moves from left hand
        else: #If it is not the initial state, we can play the game as we normally would (with checking for valid moves!). Here we check if we have to play from the right hand.
            for i in state.rightHand: #iterate through every card in your hand
                if valid_move(solution[-1], i): #state.playedCards[-1] gets you the item on top of the stack without popping. (we want to compare the card to the last card played)
                    possibleMoves.append(i) #this adds the card for which there is a valid move to the list of cards that can be played. 'extend' is a way of appending to a list
            return possibleMoves #return the list of possible moves from right hand


    def result(self, state, action, solution):
        #in result, the state is returned that results from executing (one of the) moves from Actions (above).
        """Return the state that results from executing the given action in the given state. The 'action' is one of the cards from the list implemented above"""
        #this is the state now (originalstate), we get an action (cardPlayed), return the state how we want it to be after that action.
        #cardPlayed = 'card played in original state'

        originalState = state
        cardPlayed = action  #cardPlayed = 'card played in original state'
        newHand = [] #make a new list

        if (originalState.playLeft): #if the card played in original state was from the left hand, we know the card should be removed from that hand and placed on top of the stack of played cards.
            for i in originalState.leftHand:
                if i == cardPlayed: #if the card we're looking at is the one that should be removed
                    pass       # then don't add it to the new hand for the new state, but add it to the top of the played cards stack.
                else:
                    newHand.append(i) #add every other card (that has not been played) to the new hand for the new state
            return GameState(newHand, originalState.rightHand, False, False) #this returns a gamestate object with as parameters (in order): new left hand, original right hand, next move will NOT be from left hand, we are NOT in initial state, the last played card(s))


        else: #when it is not the initial state, and it is not left hand turn
            for i in originalState.rightHand:
                if i == cardPlayed:
                    pass
                else:
                    newHand.append(i)

            return GameState(originalState.leftHand, newHand, True, False) #this returns a gamestate object with as parameters (in order): new left hand, original right hand, next move WILL be from left hand, we are NOT in initial state, the last played card(s)


    def goal_test(self, state):
        """Return True if the state is a goal. Namely, when the left and right hand are empty"""
        #The game is defined to have left play first. If left plays first, right lays down the last card
        if (len(state.rightHand)==0):
            return True

        #when both hands are empty the game is done.

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1
        #For now all paths have an equal cost

    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError
        #We don't have to worry about heuristic value (yet)


# ______________________________________________________________________________

class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value;"""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        self.nodeName = np.array_str(np.array(self.state.leftHand)) + np.array_str(np.array(self.state.rightHand))
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state, self.solution())]


    def child_node(self, problem, action):
        next_state = problem.result(self.state, action, self.solution())
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        #for node in self.path()[1:]:
           #print("Solution: node action is {}".format(node.action))
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)


# ______________________________________________________________________________

def valid_move(cardA, cardB):
    #print("validMove: comparing " + str(cardA) + " to " + str(cardB)) UNCOMMENT THIS TO SEE WHICH CARDS ARE BEING COMPARED
    g = np.arange(20).reshape(5, 4) #this produces the same grid as the representation, for the purpose of checking moves
    if check_value(cardA, cardB, g):
        return True
    elif check_suit(cardA, cardB, g):
        return True
    else:
        #print("validMove: No move found")
        return False

def check_suit(cardA, cardB, grid):
    r, c = grid.shape
    for i in range(c):
        if np.any(grid[:, i] == cardA) and np.any(grid[:, i] == cardB):
            return True

def check_value(cardA, cardB, grid):
    r, c = grid.shape
    if (cardA == 99) or (cardB == 99):
        return True
    for i in range(r):
        if np.any(grid[i] == cardA) and np.any(grid[i] == cardB):
            return True

def pick_cards(seed, size):
    random.seed(seed)
    cards = np.random.choice(20, (size*2), replace = False)
    leftHand = cards[:size]
    rightHand = cards[size:]
    return (leftHand, rightHand)

def exportToText(STUDENT_NUMBER="MISSING STUDENT NUMBER", noLeft="NO VALUE", noRight="NO VALUE", nodeCount="NO VALUE", myReport="NO VALUE", DFSL="NO VALUE", DFSR="NO VALUE", BFSL="NO VALUE", BFSR="NO VALUE"):
    fileName = str(STUDENT_NUMBER)+".txt"
    f = open(fileName, "w+")
    f.write("Intelligent Systems Practical 1\n")
    f.write("Student number: {}\n".format(STUDENT_NUMBER))
    f.write("Left hand without solution: {}\n".format(noLeft))
    f.write("Right hand without solution: {}\n".format(noRight))
    f.write("node Count: {}\n".format(nodeCount))
    f.write("REPORT: {}\n".format(myReport))
    f.write("Fast for DFS Left: {}\n".format(DFSL))
    f.write("Fast for DFS Right: {}\n".format(DFSR))
    f.write("Fast for BFS Left: {}\n".format(BFSL))
    f.write("Fast for DFS Right: {}\n".format(BFSR))
    f.close()
