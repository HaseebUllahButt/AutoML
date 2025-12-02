"""
AutoML System Configuration
Handles all system-wide settings and thresholds for edge case handling
"""

class AutoMLConfig:
    """Central configuration for the AutoML system"""
    
    # ==================== FILE VALIDATION ====================
    MAX_FILE_SIZE_MB = 500
    ALLOWED_EXTENSIONS = ['.csv', '.txt', '.tsv']
    ENCODING_ATTEMPTS = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16', 'utf-16-le', 'utf-16-be']
    DELIMITER_ATTEMPTS = [',', ';', '|', '\t', '^', '~', ':']
    
    # ==================== MISSING VALUE REPRESENTATIONS ====================
    MISSING_INDICATORS = [
        '', ' ', '  ', '   ',
        'NA', 'N/A', 'na', 'n/a', 'N.A.', 'n.a.',
        'NaN', 'nan', 'NAN',
        'null', 'NULL', 'Null',
        'None', 'none', 'NONE',
        'NIL', 'nil', 'Nil',
        '?', '??', '???',
        '--', '—', '___', '_',
        '.', '..', '...',
        'Missing', 'MISSING', 'missing',
        'Unknown', 'UNKNOWN', 'unknown',
        'N', 'n',
    ]
    
    # ==================== DATA QUALITY THRESHOLDS ====================
    MAX_MISSING_RATIO = 0.95  # Drop columns with >95% missing
    MIN_VARIANCE_THRESHOLD = 1e-10  # Drop zero-variance columns
    MAX_CARDINALITY_RATIO = 0.95  # Drop if unique_count/total_rows > 0.95 (likely ID)
    MIN_ROWS_REQUIRED = 10  # Minimum rows needed for training
    MIN_SAMPLES_PER_CLASS = 2  # Minimum samples per class for stratification
    MAX_CATEGORIES_OHE = 50  # Switch to hashing/target encoding above this
    RARE_CATEGORY_THRESHOLD = 0.01  # Collapse categories with <1% frequency
    
    # ==================== OUTLIER DETECTION ====================
    OUTLIER_Z_THRESHOLD = 5.0  # Z-score threshold
    OUTLIER_IQR_MULTIPLIER = 3.0  # IQR multiplier
    MAX_OUTLIER_REMOVAL_RATIO = 0.1  # Don't remove more than 10% of data as outliers
    
    # ==================== MODEL TRAINING ====================
    CV_FOLDS = 5
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    MAX_TRAINING_TIME_SECONDS = 300  # 5 minutes per model
    EARLY_STOPPING_PATIENCE = 10
    
    # ==================== MEMORY MANAGEMENT ====================
    CHUNK_SIZE = 10000  # For large file processing
    MAX_CORRELATION_FEATURES = 500  # Don't compute corr matrix above this
    MAX_PAIRPLOT_FEATURES = 10  # Don't create pairplots above this
    MEMORY_WARNING_THRESHOLD_MB = 1000  # Warn if DataFrame > 1GB
    
    # ==================== MODEL REGISTRY ====================
    CLASSIFICATION_MODELS = {
        'dummy': {'enabled': True, 'fast': True},
        'logistic_regression': {'enabled': True, 'fast': True},
        'knn': {'enabled': True, 'fast': False},
        'naive_bayes': {'enabled': True, 'fast': True},
        'decision_tree': {'enabled': True, 'fast': True},
        'random_forest': {'enabled': True, 'fast': False},
        'gradient_boosting': {'enabled': True, 'fast': False},
        'xgboost': {'enabled': True, 'fast': False},
        'lightgbm': {'enabled': True, 'fast': True},
        'svm': {'enabled': False, 'fast': False},  # Disabled by default (too slow)
    }
    
    REGRESSION_MODELS = {
        'dummy': {'enabled': True, 'fast': True},
        'linear_regression': {'enabled': True, 'fast': True},
        'ridge': {'enabled': True, 'fast': True},
        'lasso': {'enabled': True, 'fast': True},
        'elastic_net': {'enabled': True, 'fast': True},
        'knn': {'enabled': True, 'fast': False},
        'decision_tree': {'enabled': True, 'fast': True},
        'random_forest': {'enabled': True, 'fast': False},
        'gradient_boosting': {'enabled': True, 'fast': False},
        'xgboost': {'enabled': True, 'fast': False},
        'lightgbm': {'enabled': True, 'fast': True},
        'svm': {'enabled': False, 'fast': False},  # Disabled by default (too slow)
    }
    
    # ==================== HYPERPARAMETER SEARCH ====================
    HP_SEARCH_METHOD = 'random'  # 'grid', 'random', or 'bayesian'
    HP_SEARCH_ITERATIONS = 20  # For random/bayesian search
    HP_SEARCH_CV = 3  # Cross-validation folds during HP search
    
    # ==================== FEATURE ENGINEERING ====================
    ENABLE_AUTO_FEATURE_ENGINEERING = True
    MAX_POLYNOMIAL_DEGREE = 2
    MAX_INTERACTION_FEATURES = 50
    
    # ==================== OUTPUT ====================
    SAVE_PREPROCESSING_PIPELINE = True
    SAVE_TRAINED_MODEL = True
    GENERATE_HTML_REPORT = True
    GENERATE_CODE_STUB = True
    
    # ==================== STREAMLIT UI ====================
    PAGE_TITLE = "AutoML System"
    PAGE_ICON = "🤖"
    LAYOUT = "wide"
    
    @classmethod
    def get_enabled_models(cls, task_type: str):
        """Get list of enabled models for a task"""
        if task_type == 'classification':
            models = cls.CLASSIFICATION_MODELS
        elif task_type == 'regression':
            models = cls.REGRESSION_MODELS
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return [name for name, config in models.items() if config['enabled']]
    
    @classmethod
    def get_fast_models(cls, task_type: str):
        """Get list of fast models for quick testing"""
        if task_type == 'classification':
            models = cls.CLASSIFICATION_MODELS
        elif task_type == 'regression':
            models = cls.REGRESSION_MODELS
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return [name for name, config in models.items() if config['enabled'] and config['fast']]
