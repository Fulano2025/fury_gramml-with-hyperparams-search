import pynisher
import pandas as pd
import numpy as np
import sklearn

from gramml.dispatcher.initializers import PipelineInitializer
from sklearn.ensemble import RandomForestRegressor
from typing import List, Union, Dict, Any
from time import time


class State:
    """
    A class representing a general state
    """
    def clone(self) -> 'State':
        """
        Clones the current state and returns a new instance.
        """
        raise NotImplementedError('Subclasses should to implement this.')
    
    def get_possible_actions(self) -> List[str]:
        """
        Returns the list of possible actions for the current state.
        """
        raise NotImplementedError('Subclasses should to implement this.')
    
    def take_action(self, action: Union[str, List[str]]) -> 'State':
        """
        Updates the current state by taking a given action.
        """
        raise NotImplementedError('Subclasses should to implement this.')
    
    def is_terminal(self) -> bool:
        """
        Checks whether the current state is a terminal state.
        """
        raise NotImplementedError('Subclasses should to implement this.')

    def get_reward(self, X_sample: np.ndarray, y_sample: np.ndarray, cv: Union[None, int]) -> float:
        """
        Computes the reward for the current state.
        """
        raise NotImplementedError('Subclasses should to implement this.')


class GrammarState(State):
    """
    A class representing a single state in a grammar-based search for a machine learning pipeline.
    """

    def __init__(self, initial_symbol: str, grammar: Dict[str, List[List[str]]], blacklist: List[List[str]],
                 eval_time_budget: Union[None, float] = None, seed: Union[None, int] = None) -> None:
        """
        Initializes a new GrammarState instance.

        Args:
            initial_symbol: The initial symbol of the grammar.
            grammar: The grammar represented as a dictionary.
            blacklist: The blacklist of invalid pipelines.
            eval_time_budget: The time budget for evaluating a pipeline configuration.
            seed: The random seed for the evaluation of pipelines.

        Returns:
            None.
        """
        self.symbol = initial_symbol
        self.grammar = grammar
        self.blacklist = blacklist

        self.eval_time_budget = eval_time_budget
        self.seed = seed

        self.stack = [initial_symbol]
        self.step = ""

        self.PENALTY: float = 0.0

        self.running_time = None
        self.elapsed_time = None

    def clone(self) -> 'GrammarState':
        """
        Clones the current state and returns a new instance.

        Returns:
            A new instance of GrammarState.
        """
        copy_object = GrammarState(
            self.symbol,
            self.grammar,
            self.blacklist,
            eval_time_budget=self.eval_time_budget,
            seed=self.seed)

        copy_object.stack = self.stack.copy()
        copy_object.step = self.step

        return copy_object

    def get_possible_actions(self) -> List[str]:
        """
        Returns the list of possible actions for the current state.

        Returns:
            A list of possible actions.
        """
        top_symbol = self.stack[-1]
        if top_symbol.isupper():
            return self.grammar.get(top_symbol)
        else:
            return [top_symbol]

    def take_action(self, next_symbol: Union[str, List[str]]) -> 'GrammarState':
        """
        Updates the current state by taking a given action.

        Args:
            next_symbol: The next symbol to be taken.

        Returns:
            A new instance of GrammarState.
        """
        new_state = self.clone()

        new_state.blacklist = self.blacklist

        new_state.symbol = new_state.stack.pop()

        if isinstance(next_symbol, list):
            new_state.stack.extend(reversed(next_symbol))
        else:
            new_state.step += next_symbol

        return new_state

    def is_terminal(self) -> bool:
        """
        Checks whether the current state is a terminal state.

        Returns:
            True if the state is terminal, False otherwise.
        """
        return len(self.stack) == 0

    def get_reward(self, X_sample: np.ndarray, y_sample: np.ndarray, cv: Union[None, int]) -> float:
        """
        Computes the reward for the current pipeline configuration.

        Args:
            X_sample: The training input data.
            y_sample: The training target data.
            cv: The number of cross-validation folds.

        Returns:
            The reward for the pipeline configuration.
        
        TODO: Se ejecuta local. Para automeli esto hay que abstraerlo 
              con una intefaz que permita la ejecución local o remota. 
        """
        
        self.running_time = 0
        t_start = time()

        conf = eval(self.step) 
        
        class_list = self.get_class_list(conf, [])
        
        if self.in_blacklist(class_list):
            return self.PENALTY
        
        try:
         
            pipeline = PipelineInitializer().get_pipeline(conf)[1]
            
            def partial_eval_func(X_sample, y_sample, pipeline, cv):
                if cv is not None:
                    score = sklearn.model_selection.cross_val_score(pipeline,
                                                                    X_sample, y_sample,
                                                                    scoring='balanced_accuracy',
                                                                    #n_jobs=-1,
                                                                    cv=sklearn.model_selection.StratifiedKFold(cv))
                    return score.mean() #1-((1-score.mean())+(score.std())) 
                else:
                    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                        X_sample, y_sample, test_size=0.33, random_state=self.seed)

                    pipeline.fit(X_train, y_train)
                    return sklearn.metrics.balanced_accuracy_score(y_test, pipeline.predict(X_test))
            
            
            if self.eval_time_budget is not None:
                reward_func = pynisher.enforce_limits(
                    cpu_time_in_s=self.eval_time_budget)(partial_eval_func) 
                try:
                    val_score = reward_func(X_sample, y_sample, pipeline, cv)
                except:
                    print('TIMEOUT REWARD')
                    val_score = self.PENALTY
            else:
                val_score = partial_eval_func(X_sample, y_sample, pipeline, cv)
                        
            sc_mean = val_score
            
            self.running_time = time() - t_start
            
            if pd.isnull(sc_mean):
                return self.PENALTY

            return sc_mean
        
        except Exception as e:
            print('CROSS VAL Fail')
            print(conf)
            print(e)
            self.running_time = time() - t_start
            return self.PENALTY


    def __eq__(self, other: Any) -> bool:
        """
        Returns True if the step of this instance is equal to the step of the other instance.
        """
        return self.step == other.step

    def __hash__(self) -> int:
        """
        Returns the hash of the step of this instance.
        """
        return hash(self.step)

    def in_blacklist(self, class_list: List[str]) -> bool:
        """
        Returns True if any of the pipelines in the blacklist matches with the given class list.
        """
        pipeline_list = '>>'.join(class_list)
        return any('>>'.join(pipeline) in pipeline_list for pipeline in self.blacklist) 

    def get_class_list(self, data: Dict[str, Any], class_list: List[str]) -> List[str]:
        """
        Returns a list of the classes in the given data dictionary.
        """
        allowed_types = (str, list, int, float, type(None))
        for key, value in data.items():
            if key == 'class':
                class_list.append(value)
            if isinstance(value, dict):
                self.get_class_list(value, class_list)
            elif isinstance(value, list):
                for val in value:
                    if isinstance(val, allowed_types):
                        pass
                    else:
                        self.get_class_list(val, class_list)
        return class_list

    
 