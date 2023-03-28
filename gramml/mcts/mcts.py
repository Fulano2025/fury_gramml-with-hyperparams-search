import math
from time import time

from itertools import count

from typing import Optional, Dict, List, Any, Union, Tuple

import numpy as np

from gramml.mcts.rollout import default_policy, validation_score
from gramml.mcts.tree import TreeNode

from gramml.mcts.selection import SelectionPolicy
from gramml.mcts.state import State



class MCTS():
    """
    A class that implements the Monte Carlo Tree Search (MCTS) algorithm for AutoML.
    """

    def __init__(self, initial_state: State, time_limit: Optional[float] = None, iteration_limit: Optional[int] = None,
                 rollout_policy: Any = default_policy, rollout_reward: Any = validation_score,
                 selection_policy: SelectionPolicy = None, selection_policy_params: Dict[str, Any] = None) -> None:
        """
        Initialize an instance of the MCTS algorithm.

        Args:
            initial_state: The initial state.
            time_limit: The time limit to spend on searching (in seconds).
            iteration_limit: The number of iterations to perform the search.
            rollout_policy: A function that generates a random rollout or simulation.
            rollout_reward: A function that returns the reward for a given simulated state and sample data.
            selection_policy: A class that implements the selection policy for traversing the tree.
            selection_policy_params: The parameters for the selection policy.
        """
        
        if all(limit is None for limit in [time_limit, iteration_limit]):
            raise ValueError("Must have either a time limit or an iteration limit")
            
        if any(param is None for param in [selection_policy, selection_policy_params]):
            raise ValueError("You need a selection policy and its params.")
            
        if time_limit is not None:         
            self.time_limit = time_limit
            self.limit_type = 'time'
            self.max_round_time = 60
            
        elif iteration_limit > 0:                
            self.search_limit = iteration_limit
            self.limit_type = 'iterations'
                    
        self.rollout = rollout_policy
        self.get_reward = rollout_reward
        self.reward_db = {}

        self.selection_db = {}
        self.selection_list = []
        
        self.debug_stats = {
            'terminal_count':0,
            'rollout_count':0,
            'best_guess_count':0,
            'rollout_repeated_count':0,
            'terminal_repeated_count':0,
            'action_count':0,
            'time_registry':[],
            'action_registry':[],
        }
                 
        self.selection_policy = selection_policy(selection_policy_params)
        
        self.root = TreeNode(initial_state, None)
        self.selection_policy.init_statistics(self.root)
        

    def search(self, X_sample: np.ndarray, y_sample: np.ndarray, cv: int = 3) -> Dict[State, float]:
        """
        Search for the best pipelines using the MCTS algorithm.

        Args:
            X_sample: The input sample data.
            y_sample: The target sample data.
            cv: The number of cross-validation folds.

        Returns:
            The dictionary of states and their corresponding rewards.
        """
        
        self.X_sample = X_sample
        self.y_sample = y_sample
        
        self.cv = cv
        
        if self.limit_type == 'time':
            
            time_limit = time() + self.time_limit
            while time() < time_limit:
                try:
                    t1 = time()
                    last_action_count = self.debug_stats['action_count']

                    terminal_state = self.execute_round(time() + self.max_round_time)
                    terminal_state.picked_time = time()
                    self.selection_list.append(terminal_state)
                    
                    t2 = time()
                    self.debug_stats['time_registry'].append(t2-t1)            
                    self.debug_stats['action_registry'].append(self.debug_stats['action_count'] - last_action_count)
                except Exception as e:
                    #print('Execute Round Exception:',e)
                    self.print_stats()
                    raise e
        else:
            for i in range(self.search_limit):
                print("Episode", i)
                try:
                    t1 = time()
                    last_action_count = self.debug_stats['action_count']
                    
                    terminal_state = self.execute_round()
                    terminal_state.picked_time = time()
                    self.selection_list.append(terminal_state)
                    
                    t2 = time()
                    self.debug_stats['time_registry'].append(t2-t1)            
                    self.debug_stats['action_registry'].append(self.debug_stats['action_count'] - last_action_count)
                except Exception as e:
                    print('Execute Round Exception:',e)
                    pass
                
        self.print_stats()

        return self.selection_db
    
    def print_stats(self):
        print(f"Average Round Time: {np.mean(self.debug_stats['time_registry'])}")
        print(f"Total Rounds: {len(self.debug_stats['time_registry'])}")
        print(f"Terminal count: {self.debug_stats['terminal_count']}")
        print(f"Terminal Repeated count: {self.debug_stats['terminal_repeated_count']}")
        print(f"Best Guess count: {self.debug_stats['best_guess_count']}")
        print(f"Rollout count: {self.debug_stats['rollout_count']}")
        print(f"Rollout Repeated count: {self.debug_stats['rollout_repeated_count']}")
        print(f"Average Action by Minute: {np.mean(self.debug_stats['action_registry'])}")  
            
    def execute_round(self, round_time_limit: float = float('inf')) -> TreeNode:
        """
        Executes a round of the Monte Carlo Tree Search algorithm.

        Args:
            round_time_limit (float): The maximum time in seconds to run the round.

        Returns:
            TreeNode: The node at the end of the round.
        """
        node = self.root

        while not node.is_terminal and time() < round_time_limit:
            node = self.select_and_expand(self.root)
            #print(f"name:{node.name}, alpha:{node.alpha}, beta:{node.beta}")
            self.debug_stats['action_count'] += 1
            reward = self.simulation(node)
            self.debug_stats['rollout_count'] += 1
            self.selection_policy.backpropagate(node, reward)        

        if node.is_terminal:
            self.debug_stats['terminal_count'] += 1
            if node.state not in self.selection_db:
                self.selection_db[node.state] = reward
            else:
                self.debug_stats['terminal_repeated_count'] += 1
            self.prune_tree(node)
            return node.state
        else:
            self.debug_stats['terminal_count'] += 1
            self.debug_stats['best_guess_count'] += 1
            best_simulated = {k:v for k, v in sorted(self.reward_db.items(), key=lambda item: item[1], reverse=True)}
            for state,reward in best_simulated.items():
                if state not in self.selection_db:
                    self.selection_db[state]=reward
                    return state


    def simulation(self, node: TreeNode) -> float:
        """
        Runs a simulation from the given node.

        Args:
            node (TreeNode): The node to run the simulation from.

        Returns:
            float: The reward obtained from the simulation.
        """
        state = self.rollout(node.state)
        
        if state in self.reward_db:
            self.debug_stats['rollout_repeated_count'] += 1
            reward = self.reward_db[state]
        else:
            reward = self.get_reward(state, self.X_sample, self.y_sample, self.cv)
            self.reward_db[state] = reward

        return reward

    def prune_tree(self, leaf: TreeNode) -> None:
        """
        Prunes the game tree starting from the given leaf.

        Args:
            leaf (TreeNode): The leaf to start pruning from.
        """
        node = leaf
        while node is not None:
            if node.is_fully_expanded and all([child.pruned for child in node.children.values()]):
                node.pruned = True
                node = node.parent
            else:
                break

    def select_and_expand(self, node: TreeNode) -> TreeNode:
        """Select a child node to expand or expand current node and return it.

        Args:
            node (TreeNode): Node to select or expand from.

        Returns:
            TreeNode: Child node to be expanded or the expanded node.
        """
        while not node.is_terminal:
            if node.is_fully_expanded:
                node = self.selection_policy.select_best_child(node)
            else:
                return self.expand(node)
        return node

    def best_leaf(self, node: TreeNode) -> TreeNode:
        """Select the best leaf node for expanding.

        Args:
            node (TreeNode): Node to select the best leaf node from.

        Returns:
            TreeNode: The best leaf node for expanding.
        """
        while not node.is_terminal:
            if len(node.children) > 0:
                node = self.selection_policy.select_best_child(node)
            else:
                node = self.expand(node)
        return node
    
    def expand(self, node: TreeNode) -> TreeNode:
        """Expand a node by creating a new child node.

        Args:
            node (TreeNode): Node to be expanded.

        Returns:
            TreeNode: Newly created child node.

        Raises:
            Exception: If this code is reached, there is a problem with the expansion process.
        """
        actions = node.state.get_possible_actions()
        for action in actions:
            if str(action) not in node.children:
                new_node = TreeNode(node.state.take_action(action), node, name=str(action))
                self.selection_policy.init_statistics(new_node)
                node.children[str(action)] = new_node
                if len(actions) == len(node.children):
                    node.is_fully_expanded = True
                return new_node

        raise Exception("Should never reach here")



    def explore_tree_with_BFS(initial_state: State, max_leaf: int = 10, step: bool = False) -> List[Any]:
        """
        Performs a breadth-first search on the tree, starting from the initial state and exploring nodes up to a maximum
        number of leaves.

        Args:
            initial_state: The initial state of the search.
            max_leaf: The maximum number of leaves to explore.
            step: Whether to return a list of step-by-step states or not.

        Returns:
            A list of leaves (either states or step-by-step states).

        Raises:
            None.
        """
        root = TreeNode(initial_state, None)
        queue = [root]

        leafs = []
        while len(queue) > 0 and len(leafs) < max_leaf:
            node = queue.pop(0)
            if node.is_terminal:
                if step:
                    leafs.append(node.state.step)
                else:
                    leafs.append(node.state)
            else:
                actions = node.state.getPossibleActions()
                for action in actions:
                    queue.append(TreeNode(node.state.takeAction(action), node, name=str(action)))

        return leafs
