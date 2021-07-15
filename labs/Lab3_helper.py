from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_absolute_error, r2_score

# source: https://stackoverflow.com/questions/59254662/sklearn-columntransformer-with-multilabelbinarizer
class MultiHotEncoder(BaseEstimator, TransformerMixin):
    """Wraps `MultiLabelBinarizer` in a form that can work with `ColumnTransformer`. Note
    that input X has to be a `pandas.DataFrame`.
    """
    def __init__(self):
        self.mlbs = list()
        self.n_columns = 0
        self.categories_ = self.classes_ = list()

    def fit(self, X:pd.DataFrame, y=None):
        # Your solution here
        pass

    def transform(self, X:pd.DataFrame):
        if self.n_columns == 0:
            raise ValueError('Please fit the transformer first.')
        if self.n_columns != X.shape[1]:
            raise ValueError(f'The fit transformer deals with {self.n_columns} columns '
                             f'while the input has {X.shape[1]}.'
                            )
        # Your solution here
        
def exercise_2(one_hot_columns,multi_label_columns,standard_scaler_columns):
    ct = make_column_transformer(
        #(??, standard_scaler_columns),
        #(??, one_hot_columns),
        #(??, multi_label_columns),
        # Your solution here
    )
    
    return ct


class StandardScalerImproved(StandardScaler):
    def fit(self, X:pd.DataFrame, y=None):
        super().fit(X,y)
        # Your solution here
        #self.feature_names_ = ??? # Make sure this is a np.array to match convention
        return self
        
    def get_feature_names(self):
        return self.feature_names_
    

# source: https://stackoverflow.com/questions/59254662/sklearn-columntransformer-with-multilabelbinarizer
class MultiHotEncoderImproved(BaseEstimator, TransformerMixin):
    """Wraps `MultiLabelBinarizer` in a form that can work with `ColumnTransformer`. Note
    that input X has to be a `pandas.DataFrame`.
    """
    def __init__(self):
        self.mlbs = list()
        self.n_columns = 0
        self.categories_ = self.classes_ = list()
        self.feature_names_ = list()

    def fit(self, X:pd.DataFrame, y=None):
        # Your solution here
        pass

    def transform(self, X:pd.DataFrame):
        if self.n_columns == 0:
            raise ValueError('Please fit the transformer first.')
        if self.n_columns != X.shape[1]:
            raise ValueError(f'The fit transformer deals with {self.n_columns} columns '
                             f'while the input has {X.shape[1]}.'
                            )
        # Your solution here
        
    def get_feature_names(self):
        return self.feature_names_
    
        
def exercise_5(one_hot_columns,multi_label_columns,standard_scaler_columns):
    ct = make_column_transformer(
        #(??, standard_scaler_columns),
        #(??, one_hot_columns),
        #(??, multi_label_columns),
        # Make sure to include handle_unknown='ignore',drop='if_binary' in OneHotEncoder
    )
    
    return ct

def exercise_6(ct,X,y):
    pipeline = make_pipeline(
        # Your solution here
    )
    
    return pipeline.fit(X,y)

def exercise_7(ct,X,y):
    pipeline = make_pipeline(
        # Your solution here
    )
    
    return pipeline.fit(X,y)

def exercise_8(pipeline,X,y):
    best_params = None
    # Your code here
    # Try n_neighbors from 1 to 50
    # Using scoring r2
    # Set the cross-validation folds to 10
    
    return best_params

def exercise_9(X,y):
    r2_knn = None
    r2_linear = None
    one_hot_columns = ['zipcode', 'room_type', 'property_type', 'bed_type', 'instant_book', 'superhost']
    multi_label_columns = ['amenities_processed']
    standard_scaler_columns = ['years_host', 'host_response_rate','accommodates', 'bathrooms', 'bedrooms', 
                               'beds', 'price', 'weekly_price', 'monthly_price', 'security_deposit', 
                               'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights', 
                               'maximum_nights', 'number_of_reviews',  'reviews_per_month']
    ct_knn = exercise_5(one_hot_columns,multi_label_columns,standard_scaler_columns)
    ct_linear = exercise_5(one_hot_columns,multi_label_columns,standard_scaler_columns)

    pipeline_linear = make_pipeline(
        # Your solution here
    )
    pipeline_knn = make_pipeline(
        # Your solution here
    )
    
    # Your code here
    # Try n_neighbors from 1 to 50
    # Use scoring r2
    # Set the cross-validation folds to 10
    
    return r2_knn,r2_linear