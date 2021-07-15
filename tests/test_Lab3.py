import sys
import os
sys.path.append(".")

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pathlib
DIR=pathlib.Path(__file__).parent.absolute()

import joblib 
answers = joblib.load(str(DIR)+"/answers_Lab3.joblib")

# Import the student solutions
import Lab3_helper

import pandas as pd

df_airbnb = pd.read_csv(f"{DIR}/../data/airbnb.csv", engine='python')
df_airbnb = df_airbnb[df_airbnb['review_scores_rating']>85].copy()
df_airbnb['review_scores_rating'] = df_airbnb['review_scores_rating'] - df_airbnb['review_scores_rating'].mean()
df_airbnb['amenities_processed'] = df_airbnb['amenities'].apply(lambda e: set([v.replace('"',"").replace("{","").replace("}","").strip() for v in e.split(",")]).difference(set([''])))

from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal

def test_exercise_1():
    mhe = Lab3_helper.MultiHotEncoder()
    mhe.fit(df_airbnb[['amenities_processed']])
    encoded = pd.DataFrame(mhe.transform(df_airbnb[['amenities_processed']]),columns=mhe.categories_)
    assert_frame_equal(answers['exercise_1'], encoded, check_names=False)
    #assert answers['exercise_1'].equals(helper.exercise_1())

def test_exercise_2():
    one_hot_columns = ['zipcode', 'room_type', 'property_type', 'bed_type', 'instant_book', 'superhost']
    multi_label_columns = ['amenities_processed']
    standard_scaler_columns = ['years_host', 'host_response_rate', 'host_listings_count', 
                               'host_total_listings_count', 'accommodates', 'bathrooms', 'bedrooms', 
                               'beds', 'price', 'weekly_price', 'monthly_price', 'security_deposit', 
                               'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights', 
                               'maximum_nights', 'number_of_reviews',  'reviews_per_month']
    ct = Lab3_helper.exercise_2(one_hot_columns,multi_label_columns,standard_scaler_columns)
    index_values = df_airbnb[standard_scaler_columns].dropna().index
    X = df_airbnb.loc[index_values].drop('review_scores_rating',axis=1)
    y = df_airbnb.loc[index_values]['review_scores_rating']

    encoded = ct.fit_transform(X)
    assert_frame_equal(answers['exercise_2'], encoded, check_names=False)    
    
def test_exercise_3():
    standard_scaler_columns = ['years_host', 'host_response_rate', 'host_listings_count', 
                               'host_total_listings_count', 'accommodates', 'bathrooms', 'bedrooms', 
                               'beds', 'price', 'weekly_price', 'monthly_price', 'security_deposit', 
                               'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights', 
                               'maximum_nights', 'number_of_reviews',  'reviews_per_month']
    scaler = Lab3_helper.StandardScalerImproved()
    scaler.fit(df_airbnb[standard_scaler_columns])
    encoded = pd.DataFrame(scaler.transform(df_airbnb[standard_scaler_columns]),columns=scaler.get_feature_names())
    assert_frame_equal(pd.DataFrame(answers['exercise_3']), pd.DataFrame(encoded), check_names=False)    

def test_exercise_4():
    multi_label_columns = ['amenities_processed']
    mhei = Lab3_helper.MultiHotEncoderImproved()
    mhei.fit(df_airbnb[multi_label_columns])
    encoded = pd.DataFrame(mhei.transform(df_airbnb[multi_label_columns]),columns=mhei.get_feature_names())
    assert_frame_equal(answers['exercise_4'], encoded, check_names=False)    


