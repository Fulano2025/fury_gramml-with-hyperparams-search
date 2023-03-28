IMPORTS = ["import sklearn as sklearn",
           "from sklearn import *",
           "from sklearn.experimental import enable_hist_gradient_boosting",
           "import autosklearn as autosklearn",
           "from autosklearn.pipeline.components import classification",
           "from autosklearn.pipeline.components import data_preprocessing",
           "from autosklearn.pipeline.components import feature_preprocessing",
           "from autosklearn.pipeline.components import regression",
           "from autosklearn.pipeline.components.data_preprocessing import balancing",
           "from autosklearn.pipeline.components.data_preprocessing.balancing import balancing",
           "from autosklearn.pipeline.components.data_preprocessing import categorical_encoding",
           "from autosklearn.pipeline.components.data_preprocessing import category_shift",
           "from autosklearn.pipeline.components.data_preprocessing.category_shift import category_shift",
           "from autosklearn.pipeline.components.data_preprocessing import imputation",
           "from autosklearn.pipeline.components.data_preprocessing.imputation import categorical_imputation",
           "from autosklearn.pipeline.components.data_preprocessing.imputation import numerical_imputation",
           "from autosklearn.pipeline.components.data_preprocessing import minority_coalescense",
           "from autosklearn.pipeline.components.data_preprocessing import rescaling",
           "from autosklearn.pipeline.components.data_preprocessing import variance_threshold",
           "from autosklearn.pipeline.components.data_preprocessing.variance_threshold import variance_threshold"]


NODE_NAME = "name"
NODE_CLASS = "class"

NODE_STEPS = "steps"
                
NODE_COL_INDEX = "col_index"
NODE_HYPERPARAMS = "hyperparams"

HP_FIXED_PARAMS = "fixed"
HP_CONF_PARAMS = "configurable"

HP_TYPE = 'hptype'

HP_TYPE_FLO = 'float'
HP_TYPE_INT = 'integer'
HP_TYPE_CAT = 'categorical'

HP_NAME = 'name'
HP_VALUE = 'value'

HP_VALUES = 'values'

HP_CHOICES = 'choices'

HP_MIN = 'min'
HP_MAX = 'max'