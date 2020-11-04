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
    def __init__(self, leftHand, rightHand, playLeft, playedCards=[]):
        self.leftHand = np.array(leftHand)
        self.rightHand = np.array(rightHand)
        self.playedCards = playedCards
        if len(self.playedCards) == 0:
            self.playedCards.append(98)
            self.playedCards.append(99)
        self.playLeft = playLeft

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

    def actions(self, state, message):
        """Return a list of actions that can be executed in the given state.
        In this case, the list of cards for which there is a legal move, is returned"""

        moves = []
        if state.playLeft == True:
            #state.playLeft lets us know if we want to play a card from the left hand (or from right if it's false)
            for i in range(len(state.leftHand)):
                #print("This action is caused by the {}".format(message))
                if valid_move(state.playedCards[len(state.playedCards)-1], state.leftHand[i]):
                    moves.append((state.leftHand[i], i, "L"))
                #else:
                    #print("Actions: No valid moves for left found!")
            return moves

        if state.playLeft == False:
            #if stack of played cards is odd, right plays
            for i in range(len(state.rightHand)):
                if valid_move(state.playedCards[len(state.playedCards)-1], state.rightHand[i]):
                    moves.append((state.rightHand[i], i, "R"))
                #else:
                    #print("Actiona: No valid moves for right found!")
            return moves

    def result(self, state, action):
        #in result, the state is returned that results from executing (one of the) moves from Actions (above).
        """Return the state that results from executing the given
        action in the given state. The 'action' is one of the cards from the list implemented above"""

        # the triple (n-tuple as per the documentation) passed in via 'action' contains
        # as first element the card(number)
        # as second element the card index
        # as third element a flag on whether the card belongs to left or right hand!
        #print("Result: Received action containing: {} meaning we can play this card".format(action[0]))
        #print("Result: At index {} so we need to delete the element at that index".format(action[1]))
        #print("Result: From the {} hand".format(action[2]))
        if state.playLeft:
        #if action[2] == "L":
            #print("Result-L: The left hand first looks like this: {}".format(state.leftHand))
            cardInPlay = action[0]
            indexOfCardInPlay = action[1]
            #print("Result-L: The card in play is: {}".format(cardInPlay))
            #print("Result-L: We delete it from the array")
            newHand = np.delete(state.leftHand, indexOfCardInPlay)
            #print("Result-L: Hand is now {}, we add the deleted number to stack".format(newHand))
            state.playedCards.append(cardInPlay)
            #print("Result-L: Printing stack: ")
            #for i in state.playedCards:
                #print("--")
                #print(i)
            flag = False
            return GameState(newHand, state.rightHand, flag, state.playedCards)

        else:
            #print("Result-R: The right hand first looks like this: {}".format(state.rightHand))
            cardInPlay = action[0]
            indexOfCardInPlay = action[1]
            #print("Result-R: The card in play is: {}".format(cardInPlay))
            #print("Result-R: The index is: {}".format(indexOfCardInPlay) )
            #print("Result-R: We delete it from the array")
            newHand = np.delete(state.rightHand, indexOfCardInPlay)
            #print("Result-R: Hand is now {}, we add the deleted number to stack".format(newHand))
            state.playedCards.append(cardInPlay)
            #print("Result-R: Printing stack: ")
            #for i in state.playedCards:
            #    print("--")
            #    print(i)
            flag = True
            return GameState(state.leftHand, newHand, flag, state.playedCards)

    def goal_test(self, state):
        """Return True if the state is a goal. Namely, when the left and right hand are empty"""
        #print("GT: performing goal test")
        #print("GT: hands at goal test: ")
        #print("GT: {} <-- Left".format(state.leftHand))
        #print("GT: {} <-- Right".format(state.rightHand))
        if ((len(state.leftHand) == len(state.rightHand)) and len(state.leftHand)==0):
            #print("###########")
            #print(state.playedCards)
            #print("###########")
            return True


        #by definition the game is over when all cards can be matched, so when 10 (2 times 5 cards per hand) cards
        # are on the 'cards played' stack. In this function we check if the playedCards stack has 10 items in it,
        # if it does we have reached a/the goal state!

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
        self.nodeName = np.array_str(self.state.leftHand) + np.array_str(self.state.rightHand)
        if parent:
            self.depth = parent.depth + 1
        #print("Node: The depth of the new node is: {}".format(self.depth))
        #print("Node: The state represented by this node is: ")
        #self.state.printState()

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        #for action in problem.actions(self.state, "Print for loop"):
        #    print("Expand: Action is: {}".format(action))

        return [self.child_node(problem, action)
                for action in problem.actions(self.state, "Return statement")]


    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        #print("ChildNode: Next_state is {}".format(next_state))
        #print("ChildNode: Next_node is {}".format(next_node))
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
    #print("validMove: comparing " + str(cardA) + " to " + str(cardB))
    g = np.arange(20).reshape(5, 4)
    if check_value(cardA, cardB, g):
        #print("validMove-Value: valid move found")
        return True
    elif check_suit(cardA, cardB, g):
        #print("validMove-Suit: valid move found")
        return True
    else:
        #print("validMove: No move found")
        return False

def check_suit(cardA, cardB, grid):
    r, c = grid.shape
    if (cardA == 99) or (cardB == 99):
        return True
    for i in range(c):
        if np.any(grid[:,i] == cardA) and np.any(grid[:,i] == cardB):
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

def breadth_first_tree_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument fringe should be an empty queue.
    Repeats infinitely in case of loops.
    """

    fringe = deque([Node(problem.initial)])  # FIFO queue

    while fringe:
        node = fringe.popleft()
        #print("BFS: This node is now being considered (popped from fringe): {}".format(node.state.printState()))
        if problem.goal_test(node.state):
            print("###########")
            print("success!")
            return node
        fringe.extend(node.expand(problem))
    print("###########")
    print("unfortunately no solution has been found!")
    return None

def depth_first_tree_search(problem):
    """
    [Figure 3.7]
    Search the deepest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument fringe should be an empty queue.
    Repeats infinitely in case of loops.
    """

    fringe = [Node(problem.initial)]  # Stack

    while fringe:
        node = fringe.pop()
        #print("DFS: This node is now being considered (popped form fringe) {}".format(node.state.printState()))
        if problem.goal_test(node.state):
            print("###########")
            print("succes!")
            return node
        fringe.extend(node.expand(problem))
    print("###########")
    print("unfortunately no solution has been found!")
    return None

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
