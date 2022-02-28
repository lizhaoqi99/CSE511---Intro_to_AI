# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """

    def getAction(self, gameState):
        """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        current_food = currentGameState.getFood().asList()
        successor_score = successorGameState.getScore()
        score = 0

        if action == "Stop":
            return -1000

        ghostDist = []
        for ghost in newGhostStates:
            ghostDist += [manhattanDistance(ghost.getPosition(), newPos)]
        min_ghostDist = min(ghostDist)

        food = newFood.asList()
        foodDist = []
        for pos in food:
            foodDist += [manhattanDistance(pos, newPos)]

        inverse_foodDist = 0
        if foodDist and min(foodDist) > 0:
            inverse_foodDist = 1.0 / min(foodDist)

        score = score + min_ghostDist * inverse_foodDist ** 4 + successor_score

        for time in newScaredTimes:
            score += time

        if newPos in current_food:
            score *= 1.3

        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
  """

    def getAction(self, gameState):
        """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
        "*** YOUR CODE HERE ***"

        # MIN: ghost
        def min_value(gameState, depth, ghost_index):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            v = float('inf')
            legal_actions = gameState.getLegalActions(ghost_index)
            for action in legal_actions:
                if ghost_index == gameState.getNumAgents() - 1:  # no more ghost left
                    v = min(v, max_value(gameState.generateSuccessor(ghost_index, action), depth))
                else:  # continue for the next ghost (ghosts collectively want the lowest combined action score)
                    v = min(v, min_value(gameState.generateSuccessor(ghost_index, action), depth, ghost_index + 1))
            return v

        # MAX: pacman
        def max_value(gameState, depth):
            depth = depth + 1
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            v = float('-inf')
            legal_actions = gameState.getLegalActions(0)  # pacman always at index 0
            for action in legal_actions:
                v = max(v, min_value(gameState.generateSuccessor(0, action), depth, 1))
            return v

        # MAX at root node:
        legal_actions = gameState.getLegalActions(0)  # pacman always at index 0
        current_val = float('-inf')
        result_action = ''
        for action in legal_actions:
            depth = 0
            val = min_value(gameState.generateSuccessor(0, action), depth,
                            1)  # next level is MIN(ghosts) -> call min_value
            if val > current_val:
                current_val = val  # pacman chooses the action with the highest value
                result_action = action

        return result_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
  """

    def getAction(self, gameState):
        """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
        "*** YOUR CODE HERE ***"

        def min_value(gameState, alpha, beta, depth, ghost_index):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            v = float('inf')
            legal_actions = gameState.getLegalActions(ghost_index)
            for action in legal_actions:
                if ghost_index == gameState.getNumAgents() - 1:  # no more ghost left
                    v = min(v, max_value(gameState.generateSuccessor(ghost_index, action), alpha, beta, depth))
                else:  # continue for the next ghost (ghosts collectively want the lowest combined action score)
                    v = min(v, min_value(gameState.generateSuccessor(ghost_index, action), alpha, beta, depth,
                                         ghost_index + 1))
                beta = min(beta, v)
                if alpha > beta:
                    return v
            return v

        def max_value(gameState, alpha, beta, depth):
            depth = depth + 1
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            v = float('-inf')
            legal_actions = gameState.getLegalActions(0)  # pacman always at index 0
            for action in legal_actions:
                v = max(v, min_value(gameState.generateSuccessor(0, action), alpha, beta, depth, 1))
                alpha = max(alpha, v)
                if alpha > beta:
                    return v
            return v

        legal_actions = gameState.getLegalActions(0)
        current_val = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        result_action = ''
        for action in legal_actions:
            depth = 0
            val = min_value(gameState.generateSuccessor(0, action), alpha, beta, depth, 1)
            if val > current_val:
                current_val = val
                result_action = action
            alpha = max(alpha, val)
            if alpha > beta:
                return result_action

        return result_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
  """

    def getAction(self, gameState):
        """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
        "*** YOUR CODE HERE ***"

        def chance_value(gameState, depth, ghost_index):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            legal_actions = gameState.getLegalActions(ghost_index)
            chance = len(legal_actions)
            expectation = float(0)
            sum_expectation = float(0)
            for action in legal_actions:
                if ghost_index == gameState.getNumAgents() - 1:  # no more ghost left
                    expectation =  max_value(gameState.generateSuccessor(ghost_index, action), depth)
                else:  # recursion to the next ghost and calculate the weighted value of all child nodes
                    expectation =  chance_value(gameState.generateSuccessor(ghost_index, action), depth, ghost_index + 1)
                sum_expectation = sum_expectation + expectation
            if chance == 0:
              return 0
            return sum_expectation/chance

        # MAX: pacman
        def max_value(gameState, depth):
            depth = depth + 1
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            v = float('-inf')
            legal_actions = gameState.getLegalActions(0)  # pacman always at index 0
            for action in legal_actions:
                v = max(v, chance_value(gameState.generateSuccessor(0, action), depth, 1))
            return v

        # MAX at root node:
        legal_actions = gameState.getLegalActions(0)  # pacman always at index 0
        current_val = float('-inf')
        result_action = ''
        for action in legal_actions:
            depth = 0
            val = chance_value(gameState.generateSuccessor(0, action), depth,
                            1)  # next level is CHANCE(ghosts) -> call chance_value
            if val > current_val:
                current_val = val  # pacman chooses the action with the highest value
                result_action = action

        return result_action


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
    "*** YOUR CODE HERE ***"
    currentPos = currentGameState.getPacmanPosition()
    currentFood=currentGameState.getFood()
    currentFoodList=currentFood.asList()
    GhostStates=currentGameState.getGhostStates()
    currentCapsules = currentGameState.getCapsules()
    scaredTimes=[ghostState.scaredTimer for ghostState in GhostStates]
    score=0

    ghostDist=[]
    for ghost in GhostStates:
        ghostDist+=[manhattanDistance(ghost.getPosition(), currentPos)]
    min_ghostDist =  min(ghostDist)
    is_scared = False
    for time in scaredTimes:
        if time != 0:
            is_scared = True
        else:
            is_scared = False
            break
    
    foodDist=[]
    for foodpos in currentFoodList:
        foodDist+=[manhattanDistance(currentPos, foodpos)]

    inverse_foodDist=0
    if foodDist and min(foodDist) > 0:
        inverse_foodDist=1.0 / min(foodDist)
 
    # more food eaten -> higher score
    score = score + currentGameState.getScore() + len(currentFood.asList(False))

    if is_scared and min_ghostDist!=0:
        min_ghostDist = min_ghostDist * 0.8
    if min(ghostDist) == 0:
        score += inverse_foodDist + len(currentCapsules)
    else:
        score += sum(scaredTimes) + min_ghostDist * (float(inverse_foodDist))
    return score
    

# Abbreviation
better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest
  """

    def getAction(self, gameState):
        """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

