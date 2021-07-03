import sys
import os
sys.path.append(".")

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pathlib
DIR=pathlib.Path(__file__).parent.absolute()

import joblib 
answers = joblib.load(str(DIR)+"/answers_Lab2.joblib")

# Import the student solutions
import Lab2_helper

import pandas as pd

df_airbnb = pd.read_csv(f"{DIR}/../data/airbnb.csv")

from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal

def test_exercise_1():
    assert_series_equal(answers['exercise_1'], Lab2_helper.exercise_1(df_airbnb['number_of_reviews'],l=0.2), check_names=False)
    #assert answers['exercise_1'].equals(helper.exercise_1())

def test_exercise_2():
    Lab2_helper.exercise_2(df_airbnb,'number_of_reviews',percentiles=[0.25,0.5,0.75])
    assert_frame_equal(answers['exercise_2'], df_airbnb[['number_of_reviews<=0.25','number_of_reviews<=0.5','number_of_reviews<=0.75']], check_names=False)
    #assert answers['exercise_2'].equals(helper.exercise_2())

def test_exercise_3():
    Lab2_helper.exercise_2(df_airbnb,'number_of_reviews',percentiles=[0.25,0.5,0.75])
    Lab2_helper.exercise_2(df_airbnb,'review_scores_rating',percentiles=[0.25,0.5,0.75])
    assert_frame_equal(answers['exercise_3'], Lab2_helper.exercise_3(df_airbnb['number_of_reviews<=0.5'],df_airbnb['review_scores_rating<=0.5']), check_names=False)
    #assert answers['exercise_3'].equals(helper.exercise_3(titanic_df))

def test_exercise_4():
    Lab2_helper.exercise_2(df_airbnb,'number_of_reviews',percentiles=[0.25,0.5,0.75])
    Lab2_helper.exercise_2(df_airbnb,'review_scores_rating',percentiles=[0.25,0.5,0.75])
    jd_num_reviews_rating_0p5 = Lab2_helper.exercise_3(df_airbnb['number_of_reviews<=0.5'],df_airbnb['review_scores_rating<=0.5'])
    assert_frame_equal(answers['exercise_4'], Lab2_helper.exercise_4(jd_num_reviews_rating_0p5), check_names=False)
    #assert answers['exercise_4'].equals(helper.exercise_4(titanic_df))
