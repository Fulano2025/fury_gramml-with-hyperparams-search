import numpy as np
from time import time
from typing import List, Dict, Any
from gramml.mcts.state import State



def default_policy(state: State) -> State:
    """Performs the default policy for a given state.

    Args:
        state (State): The initial state.

    Returns:
        State: The final state after performing the default policy.
    """
    while not state.is_terminal():
        try:
            actions = state.get_possible_actions()
            possible_states = [state.take_action(action) for action in actions]
            new_state = np.random.choice(possible_states)
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = new_state
    state.elapsed_time = time()
    return state


def validation_score(state: State, X_sample: np.ndarray, y_sample: np.ndarray,
                     cv: int) -> float:
    """Calculates the validation score for a given state.

    Args:
        state (State): The state to calculate the validation score for.
        X_sample (np.ndarray): The input features.
        y_sample (np.ndarray): The target values.
        cv (int): The number of cross-validation folds.

    Returns:
        float: The validation score for the given state.
    """
    return state.get_reward(X_sample, y_sample, cv=cv)
