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

# To join or not to join
* Merging datasets


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
import Lab4_helper 

from pathlib import Path
home = str(Path.home()) # all other paths are relative to this path. 
# This is not relevant to most people because I recommended you use my server, but
# change home to where you are storing everything. Again. Not recommended.
```

## Data
There are a lot of different data files associated with this pesticide data:

>This dataset contains information on pesticide residues in food. The U.S. Department of Agriculture (USDA) Agricultural Marketing Service (AMS) conducts the Pesticide Data Program (PDP) every year to help assure consumers that the food they feed themselves and their families is safe. Ultimately, if EPA determines a pesticide is not safe for human consumption, it is removed from the market.

>The PDP tests a wide variety of domestic and imported foods, with a strong focus on foods
that are consumed by infants and children. EPA relies on PDP data to conduct dietary risk
assessments and to ensure that any pesticide residues in foods remain at safe levels. USDA
uses the data to better understand the relationship of pesticide residues to agricultural practices
and to enhance USDA’s Integrated Pest Management objectives. USDA also works with U.S.
growers to improve agricultural practices.

> While the original 2013 MS Access database can be found here, the data has been transferred to a SQLite database for easier, more open use. The database contains two tables, Sample Data and Results Data. Each sampling includes attributes such as extraction method, the laboratory responsible for the test, and EPA tolerances among others. These attributes are labeled with codes, which can be referenced in PDF format here, or integrated into the database using the included csv files.

Source: https://www.kaggle.com/usdeptofag/pesticide-data-program-2013

We will first examine files that define codes and relationships. They are stored as a dictionary of dataframes with the key being the file name without the .csv.

```python
import pandas as pd
import glob
dfs = {}
for file in glob.glob(f'{home}/data-301-student/data/pesticide/*.csv'):
    name = file.split("/")[-1].replace(".csv","")
    dfs[name] = pd.read_csv(file)
    print(name)
    display(dfs[name].head())
```

We will now read in data itself.

```python
import pandas as pd
import glob
data = {}
for file in glob.glob(f'{home}/data-301-student/data/pesticide/*.csv.gz'):
    name = file.split("/")[-1].replace(".csv.gz","")
    data[name] = pd.read_csv(file)
    print(name)
    display(data[name].head())
```

## What are the most common types of pesticides tested in this study?


**Exercise 1:** The dataframe stored in ``data['resultsdata13']['pestcode']`` has test pesticide test results. Write a function that counts the number of times a pesticide is tested and join these results to ``dfs['pest_codes']``. Report this count with the pestecide name and sort by pestecide name. Here is the example output:

```python
df = Lab4_helper.exercise_1(data['resultsdata13'],dfs['pest_codes'])
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
