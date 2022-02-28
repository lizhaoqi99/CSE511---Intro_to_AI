# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
     
    "*** YOUR CODE HERE ***"
    for ite in range(self.iterations):
      values_dict = util.Counter()  # initialize the empty dictionary for every iteration 
                                    # because we need to update all the V-values for every iteration
      states = self.mdp.getStates()
      for state in states:
        actions = self.mdp.getPossibleActions(state)
        q_values = []
        for action in actions:
          # compute Q*(s,a)
          q_value = self.getQValue(state, action)
          q_values.append(q_value)
          # compute V*(s) 
          values_dict[state] = max(q_values)
      self.values = values_dict


  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    q_value = 0
    transition_list = self.mdp.getTransitionStatesAndProbs(state, action) # returns a list of (nextState, prob) pair
    for transition in transition_list:
      successor_state, prob = transition 
      reward = self.mdp.getReward(state, action, successor_state)
      v_value_successor_state = self.getValue(successor_state)
      q_value += prob * (reward + self.discount * v_value_successor_state)
    return q_value

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    if self.mdp.isTerminal(state):
      return None

    actions = self.mdp.getPossibleActions(state)
    q_values_dict = {}
    for action in actions:
      q_value = self.getQValue(state, action)
      q_values_dict.update({q_value: action})
      best_action = q_values_dict[max(q_values_dict)]
    return best_action


  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
