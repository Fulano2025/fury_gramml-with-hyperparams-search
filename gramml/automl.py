import numpy as np
import random as random
import sklearn


from gramml.mcts.mcts import MCTS

from gramml.mcts.state import GrammarState

from gramml.grammar.utils import (augment_grammar, restrict_grammar,
                                   components_to_symbols, get_minimal_grammar,
                                   get_number_of_paths)

from gramml.columns.column_selectors import FullColumnSelector

from gramml.dispatcher.initializers import PipelineInitializer




class GramML:
    
    def __init__(self, grammar, components, blacklist):
        
        self.grammar = grammar
        self.components = components
        self.blacklist = blacklist
        
        
    def generate_search_space(self, grammar, components, feat_type, seed, default_hyperparams):

        for k, v in components.items():
            if "autosklearn" in components[k]["class"]:
                components[k]["hyperparams"].append({"name": "random_state", "value": seed})
        
        column_selector = FullColumnSelector(feat_type)
        
        COMPONENT_SPACE = components_to_symbols(components, default_hyperparams=default_hyperparams)

        grammar['SYMBOLS'].update(COMPONENT_SPACE)
        
        AUGMENTED_GRAMMAR = augment_grammar(
            restrict_grammar(
                grammar['SYMBOLS'],[
                    col_type.split('_')[0] for col_type in feat_type.keys()],
                grammar['COLUMN_TYPE_TRANSFORMER_NODE']
            ), column_selector
        )
        
        return get_minimal_grammar(AUGMENTED_GRAMMAR)
        
                
    def fit(self, X_train, y_train, feat_type, time_budget, eval_time_budget, cv,
            iterations=None, seed=1, default_hyperparams=True,
            selection_policy=None, selection_policy_params=None):
        
        random.seed(seed)
        np.random.seed(seed)
        
        """
        Pipeline search
        """    
        
        self.SEARCH_SPACE = self.generate_search_space(self.grammar,
                                                       self.components,
                                                       feat_type, seed, default_hyperparams)
        
        print("NUMBER OF POSSIBLE PIPELINES",
              get_number_of_paths(self.SEARCH_SPACE, self.grammar['INITIAL_SYMBOL']))
                        
        self.initial_state = GrammarState(self.grammar['INITIAL_SYMBOL'], self.SEARCH_SPACE,
                                               self.blacklist, eval_time_budget=eval_time_budget,
                                               seed=seed)
        
        time_limit = time_budget if time_budget > 0 else None 

        self.searcher = MCTS(self.initial_state, time_limit=time_limit, selection_policy=selection_policy,
                             selection_policy_params=selection_policy_params)
        
        self.candidates = self.searcher.search(X_train, y_train, cv=cv)
        self.selection_list = self.searcher.selection_list
        self.reward_db = self.searcher.reward_db
    
    def get_brute_space(self, X_train, y_train, feat_type, time_budget, eval_time_budget, cv,
                    iterations=None, seed=1, default_hyperparams=True):
        
        random.seed(seed)
        np.random.seed(seed)
 
        self.SEARCH_SPACE = self.generate_search_space(self.grammar,
                                                       self.components,
                                                       feat_type, seed, default_hyperparams)
    
        print("NUMBER OF POSSIBLE PIPELINES",
              get_number_of_paths(self.SEARCH_SPACE, self.grammar['INITIAL_SYMBOL']))    
                
        self.initial_state = GrammarState(self.grammar['INITIAL_SYMBOL'], self.SEARCH_SPACE,
                                               self.blacklist, eval_time_budget=eval_time_budget,
                                               seed=seed)

        time_limit = time_budget if time_budget > 0 else None 

        self.searcher = MCTS(self.initial_state, time_limit=time_limit)
        
        self.start_exploration = time.perf_counter()
        
        config_space = self.searcher.explore_tree_with_BFS(
            self.initial_state, max_leaf = float('inf'), verbose=0, step=False)
        
        return config_space
 

    def score(self, X_train, y_train, X_test, y_test, step):
        
        pipeline = PipelineInitializer().get_pipeline(step)[1]
        
        try:
            pipeline.fit(X_train, y_train)
            test_score = sklearn.metrics.balanced_accuracy_score(y_test, pipeline.predict(X_test))
        except:
            test_score = float("-inf")
            
        return test_score