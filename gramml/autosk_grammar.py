# ToDo definir simbolos como constantes

CATEGORICAL_COLUMNS = 'categorical_columns'
    
NUMERICAL_COLUMNS   = 'numerical_columns'
    
GRAMMAR = {

    'INITIAL_SYMBOL' : 'ROOT_NODE',

    'COLUMN_TYPE_TRANSFORMER_NODE' : 'COLUMN_PIPELINE_NODE',


    'SYMBOLS' : {

        ##########################
        ## NON_TERMINAL SYMBOLS ##
        ##########################

        # NODES: elements that are surrounded by curly braces {}, like JSON objects or Python dictionaries.
        #        the compiler automatically add curly braces {} to this elements
        #        to generate this behaviour de symbol need to ends with "_NODE"

        'ROOT_NODE': [
            ['PIPE', 'ROOT_STEPS']
        ],

        # STEPS: elements that are surrounded by braces [], like JSON arrays or Python lists.
        #        the compiler automatically add braces [] to this elements
        #        to generate this behaviour de symbol need to ends with "_STEPS"

        'ROOT_STEPS': [
            ['DATA_PREPROCESS_NODE', 'FEATURE_PREPROCESS_NODE', 'ESTIMATOR_NODE']
        ],


        'DATA_PREPROCESS_NODE': [
            ['FEATURE_UNION', 'DATA_PREPROCESS_STEPS']
        ],


        'FEATURE_PREPROCESS_NODE': [
            ['PIPE', 'FEATURE_PREPROCESS_STEPS']
        ],


        'ESTIMATOR_NODE': [
            ['ESTIMATOR']
        ],


        'ESTIMATOR': [
            ['CLASSIFIER'],
            #['REGRESSOR']
        ],

        # DATA_PREPROCESS_STEPS

        'DATA_PREPROCESS_STEPS': [
            ['COLUMN_PIPELINES']
        ],

        'COLUMN_PIPELINES': [
            ['COLUMN_PIPELINE_NODE'],
            ['COLUMN_PIPELINE_NODE', 'COLUMN_PIPELINE_NODE']
        ],    


        'COLUMN_PIPELINE_NODE':[
            ['PIPE', 'CATEGORICAL_COLUMN_PIPELINE_STEPS'],
            ['PIPE', 'NUMERICAL_COLUMN_PIPELINE_STEPS'  ],
        ],


        'CATEGORICAL_COLUMN_PIPELINE_STEPS':[
            ['CATEGORICAL_TRANSFORMER_NODE', 'CATEGORICAL_IMPUTATION_NODE', 'CATEGORICAL_COALESCENSE_NODE'],
        ],

        'NUMERICAL_COLUMN_PIPELINE_STEPS':[
            ['NUMERICAL_TRANSFORMER_NODE', 'VARIANCE_THRESHOLD_NODE', 'RESCALING_NODE'],
        ],    


        'CATEGORICAL_TRANSFORMER_NODE': [
            ['CATEGORICAL_SHIFT', 'CATEGORICAL_COLUMNS']
        ],

        'NUMERICAL_TRANSFORMER_NODE': [
            ['NUMERICAL_IMPUTATION', 'NUMERICAL_COLUMNS']
        ],


        'CATEGORICAL_IMPUTATION_NODE': [
            ['CATEGORICAL_IMPUTATION']
        ],

        'CATEGORICAL_COALESCENSE_NODE': [
            ['CATEGORICAL_COALESCENSE']
        ],

        'VARIANCE_THRESHOLD_NODE': [
            ['VARIANCE_THRESHOLD']
        ],

        'RESCALING_NODE': [
            ['RESCALING']
        ],

         # FEATURE_PREPROCESS_STEPS

        'FEATURE_PREPROCESS_STEPS':[
            ['FEATURE_SELECTION_NODE'],
        ],

        'BALANCING_NODE': [
            ['BALANCING']
        ],

        'FEATURE_SELECTION_NODE': [
            ['FEATURE_SELECTION']
        ],

        #######################
        ## TERMINAL SYMBOLS ##
        ######################

        # JSON Elements

        'SEP': [
            [',']
        ],

        'CLASS':[
            ['"class":']
        ],

        'STEPS_START': [
            ['"steps":[']
        ],
        
        'STEPS_END': [
            [']']
        ],

        'NODE_START': [
            ['{']
        ],
        
        'NODE_END': [
            ['}']
        ],
        
        'HP_START': [
            ['"hyperparams":[']
        ],
        
        'HP_END': [
            [']']
        ],


        # COLUMNS

        'CATEGORICAL_COLUMNS': [
            [f'"col_index": .{CATEGORICAL_COLUMNS}.columns']
        ],

        'NUMERICAL_COLUMNS': [
            [f'"col_index": .{NUMERICAL_COLUMNS}.columns']
        ],

        # COMPONENTS

        'PIPE':[
            ['Pipeline.component']
        ],

        'FEATURE_UNION': [
            ['FeatureUnion.component']
        ],

        'CLASSIFIER': [
            ['AdaboostClassifier.component'],
            ['BernoulliNBClassifier.component'],
            ['DecisionTreeClassifier.component'],
            ['ExtraTreesClassifier.component'],
            ['GaussianNBClassifier.component'],
            ['GradientBoostingClassifier.component'],
            ['KNearestNeighborsClassifier.component'],
            ['LDAClassifier.component'],
            ['LibLinear_SVCClassifier.component'],
            ['LibSVM_SVCClassifier.component'],
            ['MultinomialNBClassifier.component'],
            ['PassiveAggressiveClassifier.component'],
            ['QDAClassifier.component'],
            ['RandomForestClassifier.component'],
            ['SGDClassifier.component'],
        ],
        'FEATURE_SELECTION': [
            ['Densifier.component'],
            ['ExtraTreesPreprocessorClassification.component'],
            ['FastICA.component'],
            ['FeatureAgglomeration.component'],
            ['KernelPCA.component'],
            ['RandomKitchenSinks.component'],
            ['LibLinear_Preprocessor.component'],
            ['NoPreprocessing.component'],
            ['Nystroem.component'],
            ['PCA.component'],
            ['PolynomialFeatures.component'],
            ['RandomTreesEmbedding.component'],
            ['SelectPercentileClassification.component'],
            ['SelectRates.component'],
            ['TruncatedSVD.component']
        ],

        'BALANCING': [
            ['Balancing.component']
        ],

        'CATEGORICAL_ENCODING': [
            ['NoEncoding.component'],
            ['OneHotEncoder.component']
        ],

        'CATEGORICAL_SHIFT': [
            ['CategoryShift.component']
        ],
        'CATEGORICAL_COALESCENSE': [
            ['MinorityCoalescer.component'],
            ['NoCoalescence.component']
        ],

        'RESCALING': [
            ['MinMaxScalerComponent.component'],
            ['NoRescalingComponent.component'],
            ['NormalizerComponent.component'],
            ['QuantileTransformerComponent.component'],
            ['RobustScalerComponent.component'],
            ['StandardScalerComponent.component']
        ],

        'VARIANCE_THRESHOLD': [
            ['VarianceThreshold.component']
        ],

        'CATEGORICAL_IMPUTATION': [
            ['CategoricalImputation.component']
        ],

        'NUMERICAL_IMPUTATION': [
            ['NumericalImputation.component']
        ]
    }
    
}