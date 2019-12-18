# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def genericSearch(problem, stack):
    visited = set([problem.getStartState()])
    path = []

    for n in problem.getSuccessors(problem.getStartState()):
        pathtoStart = []
        pathtoStart.append(n[1])
        stack.push( (n, pathtoStart) )

    while not stack.isEmpty() :
        state = stack.pop() #( [(x,y), Direction, Cost] , path )

        path = state[1]


        if (problem.isGoalState(state[0][0])):
            break

        elif state[0][0] not in visited:
            #mark visited
            visited.add(state[0][0])

            #get children
            for n in problem.getSuccessors(state[0][0]): # n = [(x,y), Direction, Cost]
                if (n[0][0] not in visited):
                    pathtoChild = path + [n[1]]
                    stack.push( (n, pathtoChild) )

    return path

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    #FINAL SOLUTION
    dfs_stack = util.Stack()

    return genericSearch(problem, dfs_stack)

    #INITAL SOLUTION 
    '''
    visited = set([problem.getStartState()])
    path = []

    dfs_stack = util.Stack()

    for n in problem.getSuccessors(problem.getStartState()):
        pathtoStart = []
        pathtoStart.append(n[1])
        dfs_stack.push( (n, pathtoStart) )

    while not dfs_stack.isEmpty() :
        state = dfs_stack.pop() #( [(x,y), Direction, Cost] , path )

        path = state[1]

        #print("AT STATE:", state)

        if (problem.isGoalState(state[0][0])):
            #print("GOAL:", state[0])
            break
        elif state[0][0] not in visited:
            #mark visited
            visited.add(state[0][0])

            #get children
            for n in problem.getSuccessors(state[0][0]): # n = [(x,y), Direction, Cost]
                if (n[0][0] not in visited):
                    pathtoChild = path + [n[1]]
                    dfs_stack.push( (n, pathtoChild) )

    print("PATH:", path)
    return path
    '''

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    bfs_queue = util.Queue()

    return genericSearch(problem, bfs_queue)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    stack = util.PriorityQueue()

    visited = set([problem.getStartState()])
    path = []
    cost = 0

    for n in problem.getSuccessors(problem.getStartState()): # n = [(x,y), Direction, Cost]
        pathtoStart = []
        pathtoStart.append(n[1])
        currentCost = n[2]
        stack.push((n, pathtoStart, currentCost), currentCost)

    while not stack.isEmpty() :
        state = stack.pop() #( [(x,y), Direction, Cost] , path , cost)

        path = state[1]
        cost = state[2]

        if (problem.isGoalState(state[0][0])):
            break

        elif state[0][0] not in visited:
            #mark visited
            visited.add(state[0][0])

            #get children
            for n in problem.getSuccessors(state[0][0]): # n = [(x,y), Direction, Cost]
                if (n[0][0] not in visited):
                    pathtoChild = path + [n[1]]
                    currentCost = cost + n[2]
                    stack.push((n, pathtoChild, currentCost), currentCost)

    return path

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    stack = util.PriorityQueue()

    visited = set([problem.getStartState()])
    path = []
    cost = 0

    for n in problem.getSuccessors(problem.getStartState()): # n = [(x,y), Direction, Cost]
        pathtoStart = []
        pathtoStart.append(n[1])
        uniformCost = cost + n[2]
        heuristicCost = uniformCost + heuristic(n[0], problem)

        stack.push((n, pathtoStart, uniformCost), heuristicCost)

    while not stack.isEmpty() :
        state = stack.pop() #( [(x,y), Direction, Cost] , path , cost)

        path = state[1]
        cost = state[2]

        if (problem.isGoalState(state[0][0])):
            break

        elif state[0][0] not in visited:
            #mark visited
            visited.add(state[0][0])

            children = problem.getSuccessors(state[0][0])
            #find optimal children
            for n in children: # n = [(x,y), Direction, Cost]
                if (n[0][0] not in visited):
                    pathtoChild = path + [n[1]]
                    uniformCost = cost + n[2]
                    heuristicCost = uniformCost + heuristic(n[0], problem)

                    stack.push((n, pathtoChild, uniformCost), heuristicCost)

    return path
    


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
