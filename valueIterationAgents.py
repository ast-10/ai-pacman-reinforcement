# valueIterationAgents.py
# -----------------------
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
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        allStates = self.mdp.getStates()

        for i in range(0, iterations):
            val = util.Counter()
            # val = self.values.copy()
            # calculating the maximum val of every state in a particular iteration
            for state in allStates:
                # check only if not a terminal state
                if not self.mdp.isTerminal(state):
                    # calculating Vmax = max Q(s,a), 
                    # where vmax is the max qvalue that can be obtained form all possible actions in the current state
                    vMax = max(self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state))
                    val[state] = vMax
                else:
                    continue
            self.values = val


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # start with a random value
        qVal = 0
        transitionFunc = self.mdp.getTransitionStatesAndProbs(state, action) # returns the next state and probabilty
        # of reaching the next state from current state given an action

        for i in transitionFunc:
            nextState = i[0]
            probabilty = i[1]
            reward = self.mdp.getReward(state, action, nextState) # R(s)
            discount = self.discount # gamma or the discounting factor
            qVal += probabilty * (reward + discount * self.values[nextState])
        
        return qVal

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        allPossibleActions = self.mdp.getPossibleActions(state)
        
        # check if current state is terminal state
        if self.mdp.isTerminal(state):
            return None

        # if no possible actions
        if not allPossibleActions:
            return None

        # else, return the action with the max Q-value
        return max(allPossibleActions, key=lambda action: self.computeQValueFromValues(state, action))

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
