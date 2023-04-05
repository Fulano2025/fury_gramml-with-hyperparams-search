# GramML

This code repository serves as a general example to accompany the following papers:

General approach:

Vazquez, H. C., Sánchez, J., & Carrascosa, R. GramML: Exploring Context-Free Grammars with Model-Free Reinforcement Learning. In Sixth Workshop on Meta-Learning at the Conference on Neural Information Processing Systems.

Extended approach with hyperparameter search:
Coming soon...


## Load Dataset
Load the Iris dataset from sklearn datasets


```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Gramml Config
indicate types of columns


```python
FEATURES = {
    'NUMERICAL_COLUMNS': [0, 1, 2, 3]
}
```

Parameters of the experiment


```python
from gramml.mcts.selection import UpperConfidenceBound, BooststrapThompsonSampling

args = {
    'overall_time_budget':120, # budget total, 60 seconds
    'eval_time_budget': 40, # max fitting time, 40 seconds
    'seed':42, # seed for reproducibility
    'default_hyperparams':False, # If True it uses default hp, else it perform hp search 
    'cv':None,
    'selection_policy': UpperConfidenceBound,
    'selection_policy_params': {
        'C': 0.7
    }
}
```

Search space


```python
# Autosklearn Grammar
from gramml.autosk_grammar import GRAMMAR

# Autosklearn Components with 3 sample hp from Autosklearn hp space
from gramml.autosk_components_3hp import COMPONENTS

# Autosklearn Blacklist
from gramml.autosk_blacklist import BLACKLIST
```

## Run Gramml


```python
# Cargamos las configuraciones

from gramml.automl import GramML


automl = GramML(GRAMMAR, COMPONENTS, BLACKLIST)

results = automl.fit(
    X_train, y_train, FEATURES,
    args['overall_time_budget'], args['eval_time_budget'],
    cv=None,
    iterations=None,
    seed=args['seed'],
    default_hyperparams=bool(args['default_hyperparams']),
    selection_policy=args['selection_policy'],
    selection_policy_params=args['selection_policy_params']
)
```

    NUMBER OF POSSIBLE PIPELINES 160304184384
    
    2023-03-28T13:48:16-0400 - INFO: Average Round Time: 60.58624744415283  [level: INFO]
    2023-03-28T13:48:16-0400 - INFO: Total Rounds: 2  [level: INFO]
    2023-03-28T13:48:16-0400 - INFO: Terminal count: 2  [level: INFO]
    2023-03-28T13:48:16-0400 - INFO: Terminal Repeated count: 0  [level: INFO]
    2023-03-28T13:48:16-0400 - INFO: Best Guess count: 2  [level: INFO]
    2023-03-28T13:48:16-0400 - INFO: Rollout count: 557  [level: INFO]
    2023-03-28T13:48:16-0400 - INFO: Rollout Repeated count: 0  [level: INFO]
    2023-03-28T13:48:16-0400 - INFO: Average Action by Minute: 278.5  [level: INFO]


## Evaluate Results


```python
RANKING_SIZE = 5
```


```python
ranked_candidates = [k for k,v in sorted(automl.candidates.items(), key=lambda x: x[1], reverse=True)]
print("ranked candidates score", [v for k,v in sorted(automl.candidates.items(), key=lambda x: x[1], reverse=True)])
```

    ranked candidates score [1.0, 1.0]



```python
best_candidates = ranked_candidates[:RANKING_SIZE]
```


```python
test_scores = []
for candidate in best_candidates:
    test_scores.append(automl.score(X_train, y_train, X_test, y_test, eval(candidate.step)))
```



```python
print("best candidates score", test_scores)
```

    best candidates score [1.0, 1.0]



```python
print("candidates history", [v for k,v in automl.candidates.items()])
```

    candidates history [1.0, 1.0]


### Exploration db

You can also access to the exploration history, e.g. the simulation rounds


```python
exploration_db = automl.reward_db
```


```python
print("first 10 simulation rounds", [v for k,v in exploration_db.items()][:10])
```

    first 10 simulation rounds [0.9607843137254902, 0.0, 0.9444444444444445, 0.0, 0.3333333333333333, 0.0, 0.9411764705882352, 0.9607843137254902, 0.533868092691622, 0.0]


## Notice

This project uses autosklearn version "0.7.0". During benchmarking, we found that this version was compatible when using [mosaic](https://github.com/herilalaina/mosaic_ml). However, some components have been modified to avoid errors with gramml. It is important to note that this version may be outdated and may not receive security updates, so it is recommended to use the latest stable version of [autosklearn](https://github.com/automl/auto-sklearn). Nevertheless, we have verified that this modified version works correctly in the context of this project for experimental purposes only. The developers of this project are not responsible for any potential problems or vulnerabilities that may arise when using this code.

## License

This project is licensed under the terms of the MIT License. See `LICENSE`.

## Citation

Below is the BibTeX text, if you would like to cite our works.

General approach: 
```
@inproceedings{vazquezgramml,
  title={GramML: Exploring Context-Free Grammars with Model-Free Reinforcement Learning},
  author={Vazquez, Hernan Ceferino and S{\'a}nchez, Jorge and Carrascosa, Rafael},
  booktitle={Sixth Workshop on Meta-Learning at the Conference on Neural Information Processing Systems}
}
```

Extended approach with hyperparameter search:
Comming soon...
