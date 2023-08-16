from __future__ import absolute_import, division, print_function
from math import sqrt, log
from game import Game, WHITE, BLACK, EMPTY
import copy
import time
import random

class Node:
    # NOTE: modifying this block is not recommended
    def __init__(self, state, actions, parent=None):
        self.state = (state[0], copy.deepcopy(state[1]))
        self.num_wins = 0 #number of wins at the node
        self.num_visits = 0 #number of visits of the node
        self.parent = parent #parent node of the current node
        self.children = [] #store actions and children nodes in the tree as (action, node) tuples
        self.untried_actions = copy.deepcopy(actions) #store actions that have not been tried
        simulator = Game(*state)
        self.is_terminal = simulator.game_over

# NOTE: deterministic_test() requires BUDGET = 1000
# You can try higher or lower values to see how the AI's strength changes
BUDGET = 1000

class AI:
    # NOTE: modifying this block is not recommended because it affects the random number sequences
    def __init__(self, state):
        self.simulator = Game()
        self.simulator.reset(*state) #using * to unpack the state tuple
        self.root = Node(state, self.simulator.get_actions())


    # function MCTS(Self.root)
    #    while within computational budget do
    #        v ← TreePolicy(root)
    #        winner ← DefaultPolicy(v)
    #        Backup(v, winner)
    #    end while
    #    return BestChild(root, 0)
    # end function



    def mcts_search(self):
        iters = 0
        action_win_rates = {} #store the table of actions and their ucb values
        while(iters < BUDGET):
            if ((iters + 1) % 100 == 0):
                print("\riters/budget: {}/{}".format(iters + 1, BUDGET), end="")

            node = self.select(self.root)
            winner = self.rollout(node)
            self.backpropagate(node, winner)

            iters += 1
        print()

        _, action, action_win_rates = self.best_child(self.root, 0)
        return action, action_win_rates


    #   function TreePolicy(node)
    #       while node is non-terminal do
    #           if node is not fully expanded then
    #               return Expand(node)
    #           else
    #               node ← BestChild(node, c)  
    #           end if
    #       end while
    #       return node
    #   end function

    def select(self, node):
        while (not node.is_terminal):
            if len(node.untried_actions) > 0:
                return self.expand(node)
            else:
                node = self.best_child(node)[0]
        return node
    

    #   function Expand(node)
    #       action ← an action from UntriedActions(node)
    #       child ← a new child node with state = DoAction(state, action)
    #       Add child to node.children
    #       Remove action from node.untriedActions
    #       return child
    #   end function


    def expand(self, node):

        # TODO: add a new child node from an untried action and return this new node
        child_node = None #choose a child node to grow the search tree

        # NOTE: passing the deterministic_test() requires popping an action like this
        self.simulator.reset(*node.state) 
        action = node.untried_actions.pop(0)
        self.simulator.place(*action)

        # NOTE: Make sure to add the new node to node.children
        # NOTE: You may find the following methods useful:
        #   self.simulator.state()
        #   self.simulator.get_actions()

        new_state = self.simulator.state()
        new_actions = self.simulator.get_actions()
        child_node = Node(new_state, new_actions, parent=node)
        node.children.append((action, child_node))

        return child_node
    

    #   function BestChild(node, c)
    #       bestChild ← null
    #       bestValue ← −∞
    #       for each child of node do
    #           nodeValue ←
    #               child.wins/child.visits + c * √(2 * ln(node.visits)/child.visits)
    #           if nodeValue > bestValue then
    #               bestValue ← nodeValue
    #               bestChild ← child
    #           end if
    #       end for
    #       return bestChild
    #   end function

    def best_child(self, node, c=1): 

        # TODO: determine the best child and action by applying the UCB formula

        best_child_node = None # to store the child node with best UCB
        best_action = None # to store the action that leads to the best child
        action_ucb_table = {} # to store the UCB values of each child node (for testing)

        # NOTE: deterministic_test() requires iterating in this order
        for child in node.children:
            # NOTE: deterministic_test() requires, in the case of a tie, choosing the FIRST action with 
            # the maximum upper confidence bound 
            action, child_node = child
            exploitation = child_node.num_wins / child_node.num_visits
            exploration = sqrt((2*log(node.num_visits)) / (child_node.num_visits))
            ucb_value = exploitation + c * exploration
            action_ucb_table[action] = ucb_value

            if best_child_node is None or ucb_value > action_ucb_table[best_action]:
                best_child_node = child_node
                best_action = action
            

        return best_child_node, best_action, action_ucb_table
    



    def backpropagate(self, node, result):

        while (node is not None):
            # TODO: backpropagate the information about winner
            # IMPORTANT: each node should store the number of wins for the player of its **parent** node
            if node.parent is None:
                node.num_visits += 1
                break

            parent_node = node.parent
            current_player = parent_node.state[0]
            node.num_wins += result[current_player]
            node.num_visits += 1
            node = parent_node


    #   function DefaultPolicy(node)
    #       while node is non-terminal do
    #           node ← RandomChild(node)
    #       end while
    #       return Result(node)
    #   end function`

    def rollout(self, node):
        self.simulator.reset(*node.state)
        
        while not self.simulator.game_over:
            action = self.simulator.rand_move()
            self.simulator.place(*action)

        # Determine reward indicator from result of rollout
        reward = {}
        if self.simulator.winner == BLACK:
            reward[BLACK] = 1
            reward[WHITE] = 0
        elif self.simulator.winner == WHITE:
            reward[BLACK] = 0
            reward[WHITE] = 1
        return reward
