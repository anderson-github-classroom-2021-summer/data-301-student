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

# An introduction to Selection Bias in Data


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
import Lab2_helper 

from pathlib import Path
home = str(Path.home()) # all other paths are relative to this path. 
# This is not relevant to most people because I recommended you use my server, but
# change home to where you are storing everything. Again. Not recommended.
```

```python
import pandas as pd
df_airbnb = pd.read_csv(f'{home}/data-301-student/data/airbnb.csv')
df_airbnb.head()
```

Here is a nice way to view the columns in alpabetical order:

```python
pd.Series(df_airbnb.columns).sort_values()
```

**Problem 1:** Visualize the distribution of ``number_of_reviews`` using a histogram with 20 bins.

```python
# Your solution here
```

**Exercise 1:** Apply a ladder transformation to balance the data. Use $\lambda=0.2$. Store this value in a new column called ``number_of_reviews_transformed``.

```python
df_airbnb['number_of_reviews_transformed'] = Lab2_helper.exercise_1(df_airbnb['number_of_reviews'],l=0.2)
df_airbnb['number_of_reviews_transformed'].plot.hist()
```

```python
df_airbnb[['number_of_reviews','number_of_reviews_transformed']].head()
```

**Problem 2:** Write the code to find the 25th, 50th, and 75th percentile of ``number_of_reviews`` using the quantile function. We will use these later.

```python
# Your solution here
```

**Exercise 2:** Write a function that adds three new categorical columns to the dataframe. They should be boolean columns that specify whether the ``column`` is less than or equal to the 25th, 50th, and 75th percentile of the ``column`` specified.

```python
Lab2_helper.exercise_2(df_airbnb,'number_of_reviews',percentiles=[0.25,0.5,0.75])
df_airbnb[['number_of_reviews<=0.25','number_of_reviews<=0.5','number_of_reviews<=0.75']].head()
```

**Problem 3:** Now apply this same transformation but substitute the ``review_scores_rating`` column into your exercise_2 function.

```python
# Your solution here
df_airbnb[['review_scores_rating<=0.25','review_scores_rating<=0.5','review_scores_rating<=0.75']].head()
```

**Exercise 3:** Construct the joint probability distribution of ``review_scores_rating<=0.5`` and ``number_of_reviews<=0.5``.

```python
jd_num_reviews_rating_0p5 = Lab2_helper.exercise_3(df_airbnb['number_of_reviews<=0.5'],df_airbnb['review_scores_rating<=0.5'])
jd_num_reviews_rating_0p5
```

**Exercise 4:** Using the joint probability distribution you found above, write a function that computes the conditional probability P(review_scores_rating<=0.5|number_of_reviews<=0.5). 

```python
jd_num_reviews_rating_0p5.sum(axis=1)
```

```python
rating_given_num_reviews_0p5 = Lab2_helper.exercise_4(jd_num_reviews_rating_0p5)
rating_given_num_reviews_0p5[True] # Only need to look at one side of this
```

```python
rating_given_num_reviews_0p5.loc[True,True] # P(review_scores_rating<=0.5|number_of_reviews<=0.5)
```

```python
rating_given_num_reviews_0p5.loc[False,True] # P(review_scores_rating>0.5|number_of_reviews<=0.5)
```

**Problem 4:** Interpret the value of P(review_scores_rating<=0.5|number_of_reviews<=0.5) compared to P( review_scores_rating>0.5|number_of_reviews<=0.5). What does this seem to indicate?

**Your answer here**


**Problem 5:** Discuss the interpretation and analysis of ``P(review_scores_rating<=0.5|number_of_reviews<=0.5)`` in the context of the **selection bias** seen in the correlation of higher ratings with more reviews. What does this suggest about the customers and whether this will affect where they are inclined to stay.


```python
# Good job!
# Don't forget to push with ./submit.sh
```

```python

```

```python

```

```python

```

```python

```
