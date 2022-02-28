# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
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
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    from util import Stack

    start_state = problem.getStartState()
    if problem.isGoalState(start_state):
        return []  # no actions needed
    stack = Stack()  
    stack.push((start_state,[])) # push the tuple (start_state,[]) to the stack
    visited = []  # keep track of visited states/nodes
    
    while not stack.isEmpty():
        state, action = stack.pop()
        if problem.isGoalState(state):
            return action
        if state not in visited:
            visited.append(state)
            successors = problem.getSuccessors(state)
            for successor, direction, cost in successors:
                if successor in visited:
                    continue
                stack.push((successor, action + [direction])) 
        else:
            continue
    return []  # no solution
#   util.raiseNotDefined()

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    "*** YOUR CODE HERE ***"
    from util import Queue

    start_state = problem.getStartState()
    if problem.isGoalState(start_state):
        return []  # no actions needed
    queue = Queue()  
    queue.push((start_state,[])) 
    visited = []  # keep track of visited states/nodes
    
    while not queue.isEmpty():
        state, action = queue.pop()
        if problem.isGoalState(state):
            return action
        if state not in visited:
            visited.append(state)
            successors = problem.getSuccessors(state)
            for successor, direction, cost in successors:
                if successor in visited:
                    continue
                queue.push((successor, action + [direction]))
        else:
            continue
    return []
#   util.raiseNotDefined()

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue  # min-first priority queue   

    start_state = problem.getStartState()
    if problem.isGoalState(start_state):
        return []  # no actions needed
    pq = PriorityQueue()  
    pq.push((start_state,[], 0),0) 
    visited = []  # keep track of visited states/nodes

    while not pq.isEmpty():
        loc, action, current_cost = pq.pop()
        if problem.isGoalState(loc):
            return action
        if loc not in visited:
            visited.append(loc)
            successors = problem.getSuccessors(loc)
            for successor, direction, child_cost in successors:
                if successor in visited:
                    continue
                else:
                    cost = current_cost + child_cost
                    pq.push((successor, action + [direction], cost), cost) # use cost as priority 
        else:
            continue
    return []
#   util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue  # min-first priority queue   

    start_state = problem.getStartState()
    if problem.isGoalState(start_state):
        return []  # no actions needed
    pq = PriorityQueue()  
    pq.push((start_state, [], 0), 0) # initialize root node
    visited = []  # keep track of visited states/nodes

    while not pq.isEmpty():
        loc, action, current_cost = pq.pop()
        if problem.isGoalState(loc):
            return action
        if loc not in visited:
            visited.append(loc)
            successors = problem.getSuccessors(loc)
            for successor, direction, child_cost in successors:
                if successor in visited:
                    continue
                else:
                    cost = current_cost + child_cost
                    f = cost + heuristic(loc, problem)
                    pq.push((successor, action + [direction], cost), f) # use f as priority 
        else:
            continue
    return []
#   util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
