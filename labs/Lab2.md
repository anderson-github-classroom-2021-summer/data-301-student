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

# An introduction to Bias in Data


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
```

<!-- #region -->
#### Discuss your previous answer in the context of selection bias, algorithmic bias, human bias, a biased estimator (namely that $E(\hat{\beta}) = \beta$), and regularization bias ($\hat{\beta} = (X'X + \lambda I)^{-1}X'y$) (15). You should cite at atleast two of the articles/podcasts we covered in this module: 

[Fisman and Luca (2016), Fixing Discrimination](https://hbr.org/2016/12/fixing-discrimination-in-online-marketplaces)

[NPR, Bias in the sharing economy](https://www.npr.org/2016/04/26/475623339/-airbnbwhileblack-how-hidden-bias-shapes-the-sharing-economy)

[Benjamin 2019, Automating Bias](https://science.sciencemag.org/content/366/6464/421)

[O'Neil Weapons of Math Destruction](https://www.npr.org/2018/01/26/580617998/cathy-oneil-do-algorithms-perpetuate-human-bias) 

[MIT article, Machine Learning Bias](https://sloanreview.mit.edu/article/the-risk-of-machine-learning-bias-and-how-to-prevent-it/)

[Vox Fight for your Face](https://www.vox.com/today-explained/archives)) 




#your answer here

We can see that how often an airbnb has been reviewed is correlated with higher ratings - suggesting that there is a **selection bias** occuring in terms of users' use of airbnb. Customers are more inclined to stay at AirBNBs that have a lot of ratings (similar to position bias in recommender systems, where users are more likely to click on higher ranked search results) AND review it rather than select AirBNBs with fewer ratings. 

Further, it suggests that the review process is not [incentive compatible from an microeconomic theory perspective](https://www.britannica.com/topic/incentive-compatibility) (Note that Airbnb has since changed the review process.). Namely, customers are more likely to leave a review when it's positive but less so if it's negative. Economists [Fisman and Luca (2016)](https://hbr.org/2016/12/fixing-discrimination-in-online-marketplaces) proposed a series of market design choices that might reduce discrimination in online markets more
generally—such as further automating transactions on platforms. As a result of this work, the company created a task force that weighed the different options, which led to a full-time team of data scientists to explore discrimination on an ongoing basis.


Note that the selection bias above, could be heavily driven by **human bias**. [Recall the Harvard Business School experiment on bias in the sharing economy](https://www.npr.org/2016/04/26/475623339/-airbnbwhileblack-how-hidden-bias-shapes-the-sharing-economy). If customers are less likely to stay at AirBNB's hosted by an African American or Black host, then those listings will progressively lose their ranking in users' search results. 

Finally, the algorithm (**algorithm bias**) exacerbates this selection by then being more likely to show airbnb users the listings that have a lot of reviews. 

Of course, all of the above biases are distinct from the notion of **biased estimator** in statistical inference (**selection bias** is one cause of estimators being biased), which is also different from the bias that **regularization** attempts to mitigate in the bias/variance tradeoff within prediction! But neither ridge nor RF can be used for inference. 
<!-- #endregion -->

## Exercise 0
Please read and reference the following as your progress through this course. 

* [What is the Jupyter Notebook?](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.ipynb#)
* [Notebook Tutorial](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)
* [Notebook Basics](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb)


## Excercises 1-2
We will use Pandas a lot in this course (NumPy some but less so). Please read and study [10 minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html) before proceeding to any of the exercises below.


#### Exercise 1. Make a dataframe called ``a`` of size 6 x 4 where every element is a 2.

```python
a = Lab1_helper.exercise_1()
a
```

#### Exercise 2. Create a dataframe that contains the content from the following table. Set the index of this dataframe to Series.

Notes: All of the columns should be strings. Missing values should be filled with ``np.NaN``.

|             Series             |  Aired  |            Episodes           |
|:------------------------------:|:-------:|:-----------------------------:|
| The Marvel Super Heroes        | 1966    | 65                            |
| Fantastic Four                 | 1967-68 | 20                            |
| Spider-Man                     | 1967-70 | 52                            |
| The New Fantastic Four         | 1978    | 13                            |
| Fred and Barney Meet the Thing | 1979    | 13 (26 segments of The Thing) |
| Spider-Woman                   | 1979-80 |                               |


```python
b = Lab1_helper.exercise_2()
b
```

## Exercises 3-10
Now let's look at a dataset read from a file, and talk about ``.iloc`` and ``.loc``. For this exercise, we will use the popular Titanic dataset from Kaggle. Here is some sample code to read it into a dataframe.

```python
import pandas as pd
titanic_df = pd.read_csv("https://dlsun.github.io/pods/data/titanic.csv")
titanic_df
```

```python
titanic_df.index # default index
```

#### Stop and think: How would you set the index of the data frame to the Sex column and then select only those passengers who are female?

```python
df = titanic_df.set_index('gender').loc['female']
df
```

What is important about the above statement is we have used ``.loc`` when we want to reference the pandas dataframe index. This index can be integers (starting at 0 or 1 or random). Further, it could just be a string like the example above. If you want a traditional index like you would in an array, then use ``iloc``.


#### Exercise 3. Select the ``name`` column without using .iloc?

```python
sel = Lab1_helper.exercise_3(titanic_df)
sel
```

#### Exercise 4. After setting the index to ``gender``, select all passengers that are ``male``?

```python
sel = Lab1_helper.exercise_4(titanic_df)
sel
```

<!-- #region hideOutput=false hideCode=true -->
#### Exercise 5. Reset the index of ``titanic_df_copy`` using ``inplace=True``.
<!-- #endregion -->

```python
titanic_df_copy = titanic_df.set_index('name')
Lab1_helper.exercise_5(titanic_df_copy)
titanic_df_copy
```

# Ethics

We are finally ready to think about data science ethics! 

We have preprocessed a dataset on loan applications to make this example appropriate for linear regression (i.e., y=mx+b). The independent variable data is real and has not been modified apart from being transformed (e.g., Married=Yes => Married=1.). In other words, this is a real dataset with minimal modifcations. 

Our client is a loan company, they would like you to look at this historical data of 296 loans which have been approved for varying amounts and stored in the column LoanAmountApproved. They are interested in extracting which independent variables are the most influential/important when predicting the amount of the approved loan. Upon ethical review, they have determined that ``Gender`` is a protected column and should not be considered in the analysis.

I am doing the majority of the coding for you in this part. I want you to use the ethical frameworks presented in class (see slides and video) to discuss.

```python
import pandas as pd

# Read in the data into a pandas dataframe
credit = pd.read_csv(f"{home}/data-301-student/data/credit.csv",index_col=0)
credit.head()
```

#### Construct a linear model model.

Our model should predict LoanAmountApproved using all of the columns except ``Gender`` which after an ethical review was deemed inappropriate to consider when make a determination on the amount of loan approved for an applicant.

Use ``sklearn.linear_model.LinearRegression`` with the default constructor arguments. We will need to call ``.fit``. The documentation for this is available at:

https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares

#### Exercise 6.

Your exercise is to create a new dataframe called ``X`` that does not have ``Gender`` or ``LoanAmountApproved``. You must also create a series object called ``y``. The order of the columns in ``X`` must be the same as mine.

```python
X,y = Lab1_helper.exercise_6(credit)

X
```

```python
y
```

```python
# Now we can create our model
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X,y)

coef = pd.Series(model.coef_,index=X.columns)
coef.abs().sort_values(ascending=False) # this takes the absolute value and then sorts the values in descending order
```

Now let's write some code that calculates the mean absolute error of our model which is one measure of how good our model is performing. Looks like we are approximately $27K off in our model on average.

```python
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y,model.predict(X))
```

The company asks you for your interpretation of the model. You say that being married is a high indicator of receiving a high amount for a loan. This surprises some of your colleagues, but they think this is reasonable to them. Everyone seems happy with the work. However, an experienced data scientist on your team suggests you run a correlation of the columns used in the regression against the column ``Gender`` since it is considered a protected column. You do so quickly to satisfy this request and get:

```python
# I know this code is beyond the Chapter at this point, so ignore the details.
Xgender = X.copy()
Xgender['Gender'] = credit['Gender']
Xgender.corr().loc['Gender'].abs().sort_values(ascending=False).drop('Gender')
```

**The problems labeled "Problem X" will not be autograded.** You must still complete them.


#### Problem 1: 

What do you think about the results? Specifically, is the fact that Married is correlated with Gender at a correlation of 0.36 concerning from an ethical standpoint? What do you as an individual think? Can you think of any suggestions about what to do? Answers these questions using the ethical frameworks we discussed in the ethics video/slides Utilitarianism, Deontology, or Virtue-ethics. In other words, answer these questions as you explain your answers:

Evaluate the alternate options using the following questions:
* Which option will produce the most good and do the least harm?
* Which option best enables me to fulfill my duties to all who have a stake? 
* Which option leads me to act as the sort of person I want to be?

**Your answer here**


Let's assume your suggestion was to remove it. Let's do so and then compare the accuracy of the model

```python
from sklearn.linear_model import LinearRegression

X2 = X.drop('Married',axis=1)
model2 = LinearRegression().fit(X2, y)
mean_absolute_error(y,model2.predict(X2))
```

#### Problem 2:

What do you think now that you know how a decision changes the mean absolute error, should you drop it? Your prediction is now off by more than $10,000. Is this ok? Again, use the frameworks outlined above and in the ethics slides.

**YOUR SOLUTION HERE**

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
