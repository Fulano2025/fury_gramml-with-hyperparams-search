import math
import numpy as np

from typing import Any, Dict, List, Optional, Union, Callable
from gramml.mcts.tree import TreeNode


class SelectionPolicy:    
    """
    Base class for selection policies.

    Args:
        kwargs (Dict[str, Any]): Key-value pairs to be set as instance attributes.
    """

    def __init__(self, kwargs: Dict[str, Any]) -> None:
        """
        Initializes instance attributes based on the key-value pairs passed in `kwargs`.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def backpropagate(self, node: TreeNode, reward: float) -> None:
        """
        Backpropagates the reward through the tree by updating the statistics of each node.

        Args:
            node (TreeNode): The current node being updated.
            reward (float): The reward to backpropagate.
        """
        while node is not None:
            self.update_statistics(node, reward)
            node = node.parent
    
    def select_child_by_value(self, node: TreeNode, get_value: Callable) -> TreeNode:
        """
        Selects a child node from the given `node` based on the value of each child.

        Args:
            node (TreeNode): The node to select a child from.
            get_value (Callable): A function that returns the value of a given child.

        Returns:
            TreeNode: The child node with the best value.
        """
            
        selectable_children = [child for child in node.children.values() if not child.pruned]
        children_values = []
        
        for child in selectable_children:
            
            child_value = get_value(child)
            
            children_values.append(child_value)
             
        ix = self.get_best_index(children_values)
            
        return selectable_children[ix]
    
    def select_best_child(self, node: TreeNode) -> int:
        """
        Selects the best child node from the given `node` based on its value.

        Args:
            node (TreeNode): The node to select the best child from.

        Returns:
            int: The child node with the best value.
        """
        return self.select_child_by_value(node, self.get_value)
    
    def get_value(self, child: TreeNode) -> float:
        """
        Abstract method to get the value of a given child node.

        Args:
            child (TreeNode): The child node to get the value of.

        Raises:
            NotImplementedError: This method must be implemented in the subclass.

        Returns:
            float: The value of the given child node.
        """
        raise NotImplementedError('Subclasses should to implement this.')
    
    def get_best_index(children_values: List[float]) -> int:
        """
        Returns the index of the maximum value in a list of values.

        Args:
            children_values: A list of values.
            
        Raises:
            NotImplementedError: This method must be implemented in the subclass.

        Returns:
            The index of the maximum value in the list.

        """
        raise NotImplementedError('Subclasses should to implement this.')
    
    def update_statistics(node: TreeNode, reward: float) -> None:
        """
        Updates the statistics of a node based on a reward.

        Args:
            node: The node to update.
            reward: The reward to use for the update.
            
        Raises:
            NotImplementedError: This method must be implemented in the subclass.

        Returns:
            None.

        """
        raise NotImplementedError('Subclasses should to implement this.')
    

class DiscretePolicy(SelectionPolicy):
    """
    A selection policy where the best index is chosen based on the highest value.
    """
    def get_best_index(self, children_values: List[float]) -> int:
        """
        Returns the index with the highest value.

        Args:
            children_values: A list of values.

        Returns:
            The index with the highest value.
        """
        arr = np.array(children_values)
        return np.random.choice(np.where(arr == np.amax(arr))[0])
    

class StochasticPolicy(SelectionPolicy):
    """
    A selection policy where the best index is chosen based on a probability distribution
    that is proportional to the values.
    """
    def get_best_index(self, children_values: List[float]) -> int:
        """
        Returns an index sampled from a probability distribution that is proportional
        to the children values.

        Args:
            children_values: A list of values.

        Returns:
            An index sampled from a probability distribution.
        """
        total = sum(children_values)
        return np.random.choice(len(children_values), p=np.array(children_values) / total if total > 0 else None)

    
class UpperConfidenceBound(DiscretePolicy):
    """
    A selection policy that uses the Upper Confidence Bound algorithm to choose the
    best child node to explore next.
    """
    def init_statistics(self, node: TreeNode) -> None:
        """
        Initializes statistics needed for selection policy.
        """
        node.num_visits = 0
        node.total_reward = 0        
        
    def update_statistics(self, node: TreeNode, reward: float) -> None:
        """
        Updates the statistics of a node after a visit.

        Args:
            node: The node that was visited.
            reward: The reward obtained from the visit.

        Returns:
            None.
        """
        node.num_visits += 1
        node.total_reward += reward

    def get_value(self, child: TreeNode) -> float:
        """
        Computes the value of a child node based on the Upper Confidence Bound formula.

        Args:
            child: The child node.

        Returns:
            The value of the child node.
        """    
        return child.total_reward / child.num_visits + self.C * math.sqrt(
            2 * math.log(child.parent.num_visits) / child.num_visits)
    
    
class BooststrapThompsonSampling(StochasticPolicy):
    """
    A selection policy that uses the Booststrap Thompson Sampling (with Normal Prior) algorithm to choose the
    best child node to explore next.
    """
    def init_statistics(self, node: TreeNode) -> None:
        """
        Initializes statistics needed for selection policy.
        """
        node.alpha = 0  # np.zeros(J)
        node.beta = 0  # np.zeros(J)
        
    def update_statistics(self, node: TreeNode, reward: float) -> None:
        """
        Updates the statistics of a node after a visit.

        Args:
            node: The node that was visited.
            reward: The reward obtained from the visit.

        Returns:
            None.
        """
        flips = np.random.randint(2, size=self.J-1)
        flips = np.insert(flips, 0, 1)
        node.alpha += (flips * reward).sum()
        node.beta  += flips.sum()

    def get_value(self, child: TreeNode) -> float:
        """
        Computes the value of a child node based on the Upper Confidence Bound formula.

        Args:
            child: The child node.

        Returns:
            The value of the child node.
        """
        return child.alpha/child.beta
    
    
class TreeParzenEstimator(StochasticPolicy):
    """
    A selection policy that uses the TreeParzenEstimator algorithm to choose the
    best child node to explore next.
    """
    def init_statistics(self, node: TreeNode) -> None:
        """
        Initializes statistics needed for selection policy.
        """
        node.rewards = np.array([])
        
    def update_statistics(self, node: TreeNode, reward: float) -> None:
        """
        Updates the statistics of a node after a visit.

        Args:
            node: The node that was visited.
            reward: The reward obtained from the visit.

        Returns:
            None.
        """
        node.rewards = np.append(node.rewards, reward)

    def get_value(self, child: TreeNode) -> float:
        """
        Computes the value of a child node based on the Upper Confidence Bound formula.

        Args:
            child: The child node.

        Returns:
            The value of the child node.
        """
        y_star = np.quantile(child.root.rewards, self.gamma)
        g = len(child.rewards[child.rewards >= y_star])
        l = len(child.rewards[child.rewards <  y_star])
        return g/l if l > 0.0 else 1.0


class BooststrapThompsonSamplingWithBetaPrior(StochasticPolicy):
    """
    A selection policy that uses the Booststrap Thompson Sampling (with Beta Prior) algorithm to choose the
    best child node to explore next.
    """
    def init_statistics(self, node: TreeNode) -> None:
        """
        Initializes statistics needed for selection policy.
        """
        node.alpha = 0  # np.zeros(J)
        node.beta = 0  # np.zeros(J)
        
    def update_statistics(self, node: TreeNode, reward: float) -> None:
        """
        Updates the statistics of a node after a visit.

        Args:
            node: The node that was visited.
            reward: The reward obtained from the visit.

        Returns:
            None.
        """
        flips = np.random.randint(2, size=self.J-1)
        flips = np.insert(flips, 0, 1)
        node.alpha += (flips * reward).sum()
        node.beta  += (flips * (1 - reward)).sum()

    def get_value(self, child: TreeNode) -> float:
        """
        Computes the value of a child node based on the Upper Confidence Bound formula.

        Args:
            child: The child node.

        Returns:
            The value of the child node.
        """
        return child.alpha/(child.alpha + child.beta)