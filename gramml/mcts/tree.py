import numpy as np


class TreeNode: 
    """
    A class representing a tree node.
    """
    
    def __init__(self, state, parent, name: str = 'ROOT'):
        """
        Initializes a new tree node.

        Args:
            state: The state represented by this node.
            parent: The parent node of this node.
            name: The name of this node (default 'ROOT').
        """
        self.state = state
        self.is_terminal = state.is_terminal()
        self.is_fully_expanded = self.is_terminal
        self.parent = parent
        self.children = {}
        self.name = name
        self.pruned = False
        self.root = parent.root if parent is not None else self

    def __str__(self) -> str:
        """
        Returns a string representation of this node.

        Returns:
            A string representation of this node.
        """
        s = []
        s.append(f"name: {self.name}")
        s.append(f"is_terminal: {self.is_terminal}")
        s.append(f"possible_actions: {list(self.children.keys())}")
        return f"{self.__class__.__name__}: {{{', '.join(s)}}}"

    def print_tree(self, level: str = '') -> None:
        """
        Prints a visual representation of the tree rooted at this node.

        Args:
            level: The current indentation level (default '').
        """
        if self:
            for child in self.children.values():
                child.print_tree(level + '-')
