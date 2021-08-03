import sys
import os
sys.path.append(".")

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pathlib
DIR=pathlib.Path(__file__).parent.absolute()

import joblib 
answers = joblib.load(str(DIR)+"/answers_Lab4.joblib")

# Import the student solutions
import Lab4_helper

import pandas as pd
import numpy as np

import pandas as pd
import glob
dfs = {}
for file in glob.glob(f'{DIR}/../data/pesticide/*.csv'):
    name = file.split("/")[-1].replace(".csv","")
    dfs[name] = pd.read_csv(file)

data = {}
for file in glob.glob(f'{DIR}/../data/pesticide/*.csv.gz'):
    name = file.split("/")[-1].replace(".csv.gz","")
    data[name] = pd.read_csv(file)
                      
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal

def test_exercise_1():
    df = Lab4_helper.exercise_1(data['resultsdata13'],dfs['pest_codes'])
    assert_frame_equal(answers['exercise_1'], df, check_names=False)

def test_exercise_2():
    df = Lab4_helper.exercise_2(data['sampledata13'],data['resultsdata13'],dfs['pest_codes'])
    assert_frame_equal(answers['exercise_2'], df, check_names=False)
    
def test_exercise_3():
    df = Lab4_helper.exercise_3(data['sampledata13'],data['resultsdata13'],dfs['pest_codes'],dfs['commodity_codes'])
    assert_frame_equal(answers['exercise_3'], df, check_names=False)
    
def test_exercise_4():
    df = Lab4_helper.exercise_4(data['sampledata13'],data['resultsdata13'],dfs['pest_codes'],dfs['commodity_codes'])
    #assert_frame_equal(answers['exercise_4'], df, check_names=False)
    assert_frame_equal(answers['exercise_4'].reset_index(drop=True), df.reset_index(drop=True), check_names=False)