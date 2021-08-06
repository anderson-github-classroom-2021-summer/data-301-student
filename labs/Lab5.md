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

**Exercise 1:** What is the state with highest rate of cancer deaths? Write a function to return this 

```python
df = Lab4_helper.exercise_1(data)
df
```

## Do certain states tend to use one particular pesticide type over another?


**Exercise 2:** Return a dataframe that lists the top pesticide by state. Sort by state.

```python
df = Lab4_helper.exercise_2(data['sampledata13'],data['resultsdata13'],dfs['pest_codes'])
df
```

## How consistent are the top pesticides per commodity? How much does this change from state to state?

```python
dfs['commodity_codes'].head()
```

**Exercise 3:** Return a dataframe that lists the top pesticide by commodity. Sort by Commodity Name.

```python
df = Lab4_helper.exercise_3(data['sampledata13'],data['resultsdata13'],dfs['pest_codes'],dfs['commodity_codes'])
df
```

**Exercise 4:** Return a dataframe that lists the top pesticide by commodity and state. Sort by commodity name.

```python
df = Lab4_helper.exercise_4(data['sampledata13'],data['resultsdata13'],dfs['pest_codes'],dfs['commodity_codes'])
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)
```

**Problem 1:** Are there any exceptions to the top pesticide per commodity across states? If so, what are they?

**Your answer here**


**Problem 2:** Come up with three more questions you would want to know from this data? You should be able to answer them from this data. i.e., you must avoid asking questions where more data is needed. 

**Your answer here**

```python
# Good job!
# Don't forget to push with ./submit.sh
```

```python

```
