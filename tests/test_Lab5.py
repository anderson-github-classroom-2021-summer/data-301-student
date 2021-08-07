import sys
import os
sys.path.append(".")

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pathlib
DIR=pathlib.Path(__file__).parent.absolute()

import joblib 
answers = joblib.load(str(DIR)+"/answers_Lab5.joblib")

# Import the student solutions
import Lab4_helper
import Lab5_helper

import pandas as pd
import numpy as np

import json
import requests

response = requests.get("https://corgis-edu.github.io/corgis/datasets/json/cancer/cancer.json")
data = json.loads(response.text)

import glob
dfs = {}
for file in glob.glob(f'{DIR}/../data/pesticide/*.csv'):
    name = file.split("/")[-1].replace(".csv","")
    dfs[name] = pd.read_csv(file)

data2 = {}
for file in glob.glob(f'{DIR}/../data/pesticide/*.csv.gz'):
    name = file.split("/")[-1].replace(".csv.gz","")
    data2[name] = pd.read_csv(file)

from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal

def test_exercise_1():
    df = Lab5_helper.exercise_1(data)
    assert_frame_equal(answers['exercise_1'].reset_index(drop=True), df.reset_index(drop=True), check_names=False)

def test_exercise_2():
    df = Lab5_helper.exercise_2(data)
    assert_frame_equal(answers['exercise_2'].reset_index(drop=True), df.reset_index(drop=True), check_names=False)    
    
def test_exercise_3():
    df = Lab5_helper.exercise_2(data)
    df_scaled = Lab5_helper.exercise_3(df)
    assert_frame_equal(answers['exercise_3'].reset_index(drop=True), df_scaled.reset_index(drop=True), check_names=False)   
    
def test_exercise_4():
    df = Lab5_helper.exercise_2(data)
    df_scaled = Lab5_helper.exercise_3(df)
    r2_knn,r2_linear = Lab5_helper.exercise_4(df_scaled)
    #assert_frame_equal(answers['exercise_4'], df, check_names=False)
    assert abs(r2_knn - answers['exercise_4'][0]) < 0.0001 and abs(r2_linear - answers['exercise_4'][1]) < 0.0001
    
def test_exercise_5():
    df = Lab5_helper.exercise_2(data)
    df_scaled = Lab5_helper.exercise_3(df)    
    df_top_pest = Lab4_helper.exercise_2(data['sampledata13'],data['resultsdata13'],dfs['pest_codes'])
    top_names = df_top_pest['Pesticide Name'].value_counts().index
    df_concentrations = Lab5_helper.exercise_5(top_names,data['sampledata13'],data['resultsdata13'],dfs['pest_codes'])

    #assert_frame_equal(answers['exercise_4'], df, check_names=False)
    assert_frame_equal(answers['exercise_5'].reset_index(drop=True), df_concentrations.reset_index(drop=True), check_names=False)
    
def test_exercise_6():
    df = Lab5_helper.exercise_2(data)
    df_scaled = Lab5_helper.exercise_3(df)    
    df_top_pest = Lab4_helper.exercise_2(data2['sampledata13'],data2['resultsdata13'],dfs['pest_codes'])
    top_names = df_top_pest['Pesticide Name'].value_counts().index
    df_concentrations = Lab5_helper.exercise_5(top_names,data['sampledata13'],data['resultsdata13'],dfs['pest_codes'])
    us_state_abbrev = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO',
    'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
    'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'}
    df_scaled2 = df_scaled.copy()
    df_scaled2.index = df_scaled2.index.map(us_state_abbrev)
    df_scaled2
    df_train = df_scaled2.merge(df_concentrations,left_index=True,right_index=True,how='right')
    df_train_scaled = Lab5_helper.exercise_3(df_train)
    r2_knn,r2_linear = Lab5_helper.exercise_6(df_train_scaled)
    assert abs(r2_knn - answers['exercise_6'][0]) < 0.0001 and abs(r2_linear - answers['exercise_6'][1]) < 0.0001