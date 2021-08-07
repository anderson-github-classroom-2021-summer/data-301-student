---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.8.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Hiearchical Data
* JSON


**Instructions:** This is an individual assignment, but you may discuss your code with your classmates.

**Problem type key and definition:**
* _Exercises_ are autograded on GitHub classroom
* _Problems_ are manually graded and often open ended without a single correct answer.
* _Stop and think_ prompts are not graded, and are provided to guide you.

Please see the README for instructions on how to submit and obtain the lab.

```python
%load_ext autoreload
%autoreload 2


# Put all your solutions into Lab1_helper.py as this script which is autograded
import Lab5_helper 

from pathlib import Path
home = str(Path.home()) # all other paths are relative to this path. 
# This is not relevant to most people because I recommended you use my server, but
# change home to where you are storing everything. Again. Not recommended.
```

## Data
This data comes from the CORGIS Dataset Project: https://corgis-edu.github.io/corgis/json/cancer/

>Information about the rates of cancer deaths in each state is reported. The data shows the total rate as well as rates based on sex, age, and race. Rates are also shown for three specific kinds of cancer: breast cancer, colorectal cancer, and lung cancer.

```python
import json
import requests

response = requests.get("https://corgis-edu.github.io/corgis/datasets/json/cancer/cancer.json")
data = json.loads(response.text)
print("Each record is a state:")
data[0]
```

**Exercise 1:** What is the state with highest rate of cancer deaths? Write a function that sorts the rates in descending order. Make sure you use json_normalize.

```python
df = Lab5_helper.exercise_1(data)
df
```

**Problem 1:** Create a plot (similar to the one below) that plots rate of lung cancer versus rate of breast cancer.

```python
from pandas import json_normalize
# Your solution here
```

**Problem 2:** Create a plot (similar to the one below) that plots rate of lung cancer versus colorectal cancer.

```python
from pandas import json_normalize
# Your solution here
```

**Exercise 2:** Construct a dataframe that contains the rate of lung, breast, and colorectal cancer with the first column being the state.

```python
df = Lab5_helper.exercise_2(data)
df
```

**Exercise 3:** Scale the numeric columns of the dataframe from the previous exercise such that they have a mean of 0 and unit variation. You must use from ``sklearn.preprocessing import StandardScaler``.

```python
df_scaled = Lab5_helper.exercise_3(df)
df_scaled
```

**Exercise 4:** Compare a hypertuned k-nearest neighbor regressor to a linear regressor using cross_val_score and GridSearchCV. See the helper file for more details. The data has already been scaled.

```python
r2_knn,r2_linear = Lab5_helper.exercise_4(df_scaled)
r2_knn,r2_linear
```

### What other data that we have considered in this class might be interesting to consider?

... the pesticide data ...

We'll lean on functions from Lab 4. You'll want to copy Lab4_helper.py into this directory to complete these exercises.

```python
# Copied from lab 4
import Lab4_helper

import pandas as pd
import glob
dfs = {}
for file in glob.glob(f'{home}/data-301-student/data/pesticide/*.csv'):
    name = file.split("/")[-1].replace(".csv","")
    dfs[name] = pd.read_csv(file)
    
data = {}
for file in glob.glob(f'{home}/data-301-student/data/pesticide/*.csv.gz'):
    name = file.split("/")[-1].replace(".csv.gz","")
    data[name] = pd.read_csv(file)
```

From lab 4, we know the top pesticides across states:

```python
df_top_pest = Lab4_helper.exercise_2(data['sampledata13'],data['resultsdata13'],dfs['pest_codes'])
df_top_pest
```

```python
top_names = df_top_pest['Pesticide Name'].value_counts().index
top_names
```

**Exercise 5:** Let's create a dataset that computes the average concentration of these pesticides for each state. Sort this by state.

HINTS: I used groupby and unstack. I also used other tools we've learned and nothing we haven't other than sort_index() which is a function I don't think I've mentioned. It does what you guess it does :)

```python
df_concentrations = Lab5_helper.exercise_5(top_names,data['sampledata13'],data['resultsdata13'],dfs['pest_codes'])
df_concentrations
```

Now... as I'm writing this I have no idea if this will perform any better if we include this information, but there is only way to find out... merging things together.

```python
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
df_train
```

I have to admit I'm excited, but also waiting to be disappointed as is often the case in the real world... will this new information improve performance? More columns does not always mean better (often the reverse). We also need to scale the data again for knn. Rework your previous exercise and find out!

**Exercise 6:** Using df_train, compare a hypertuned k-nearest neighbor regressor to a linear regressor using cross_val_score and GridSearchCV. See the helper file for more details. 

```python
df_train_scaled = Lab5_helper.exercise_3(df_train)
r2_knn,r2_linear = Lab5_helper.exercise_6(df_train_scaled)
r2_knn,r2_linear
```

Oh well... This doesn't mean this information wouldn't be useful for other regressors, but for the two we are most familiar with this did not help. Too many data scientists throw out negative results in the hunt for positive results. This may be ok depending on the industry, but for many applications where inference is key, you must never completely throw away negative results. We need to change the culture around data science such that a well run experiment that produces a result we don't want is valued. Does this mean pesticides can't cause cancer, of course not. This is a very limited dataset. We can only speak and discuss the bias and limits of our analysis. 

I hope you have had a great time in this course!

```python
# Good job!
# Don't forget to push with ./submit.sh
```

```python

```
