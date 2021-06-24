import sys
import os
sys.path.append(".")

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pathlib
DIR=pathlib.Path(__file__).parent.absolute()

import joblib 
answers = joblib.load(str(DIR)+"/answers_Lab1.joblib")

# Import the student solutions
import Lab1_helper as helper

import pandas as pd
titanic_df = pd.read_csv("https://dlsun.github.io/pods/data/titanic.csv")

credit = pd.read_csv(f"{DIR}/../data/credit.csv",index_col=0)

def test_exercise_1():
    assert answers['exercise_1'].equals(helper.exercise_1())

def test_exercise_2():
    assert answers['exercise_2'].equals(helper.exercise_2())

def test_exercise_3():
    assert answers['exercise_3'].equals(helper.exercise_3(titanic_df))

def test_exercise_4():
    assert answers['exercise_4'].equals(helper.exercise_4(titanic_df))
    
def test_exercise_5():
    titanic_df_copy = titanic_df.set_index('name')
    helper.exercise_5(titanic_df_copy)
    assert answers['exercise_5'].equals(titanic_df_copy)
    
def test_exercise_6():
    X,y = answers['exercise_6']
    Xs,ys = helper.exercise_6(credit)
    assert X.equals(Xs) and y.equals(ys)