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

# An introduction to Model Evaluation
* Hyperparameter tuning through model evaluation
* Bias in data
* Custom transformer


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
import Lab3_helper 

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

## Interpretation and bias

<!-- #region -->
**Problem 1:** Visualize the distribution of ``review_scores_rating`` using a boxplot. This is our dependent variable, and it is the average customer rating of the Airbnb listing, on a scale of 0-100. You'll need to use the altair package like we did in lecture. Here is some code to get you started. This code is not meant to run without additional modifications.
```python
alt.Chart(df_airbnb)
```
and
```python
properties(width=200)
```
<!-- #endregion -->

```python
import altair as alt
# Your solution here
```

One of the things that stands out to me is it might be difficult to find a single model that performs well on this entire range of scores. For this lab, let's limit ourselves to properties that are rated greater than 85 on this 100 point scale. This redefines are problem to predicting among properties with a rating of greater than 85, can we predict a more exact rating and determine what is driving this rating.

```python
print("Number of observations before:",len(df_airbnb))
df_airbnb = df_airbnb[df_airbnb['review_scores_rating']>85].copy()
print("Number of observations after:",len(df_airbnb))
```

I am also going to mean center these ratings, so that a 0 rating represents our average score. This will aid in interpretation later.

```python
df_airbnb['review_scores_rating'] = df_airbnb['review_scores_rating'] - df_airbnb['review_scores_rating'].mean()
```

```python
df_airbnb['review_scores_rating'].plot.hist()
```

Let's take a look at the non-numeric columns and see which ones might be categories (i.e., those with a limited number of values).

```python
object_columns = list(df_airbnb.select_dtypes(include=['object']))
df_airbnb[object_columns].nunique()
```

There are some obvious categorical variables which include zipcode, room_type, property_type, and bed_type. There are some that are obviously not categorical variables such as name, description, host_name, and host_about. The column amenities is most likely another text description, but let's see with a little digging:

```python
df_airbnb['amenities'].value_counts()
```

The individual entries are strings, but we want to convert this to a set for processing.

```python
df_airbnb['amenities'].loc[0]
```

```python
df_airbnb['amenities'].loc[1]
```

The code below is beyond the scope of what we can do from scratch yet, but it deals with this nested array that is embedded inside a string. Don't worry. We'll have you writing this type of thing by the end of the class.

```python
df_airbnb['amenities_processed'] = df_airbnb['amenities'].apply(lambda e: set([v.replace('"',"").replace("{","").replace("}","").strip() for v in e.split(",")]).difference(set([''])))
```

Once we fix that column, we can now convert this to binary labels. And there is a sklearn function for that of course.

```python
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
new_df = pd.DataFrame(mlb.fit_transform(df_airbnb['amenities_processed']),index=df_airbnb.index,columns=mlb.classes_)
new_df
```

How do we perform a similar operation with numeric values? For example, we might have numeric variables that are categorical variables. 

```python
number_columns = list(df_airbnb.select_dtypes(include=['number']))
for c in number_columns:
    print(c)
    display(df_airbnb[c].value_counts())
```

We now know which encoder or scaler we need to run on which columns. This is summarized below:
* OneHotEncoder: ``zipcode``, ``room_type``, ``property_type``, ``bed_type``, ``instant_book``, and ``superhost``.
* MultiLabelBinarizer: ``amenities_processed``
* StandardScaler: ``years_host``, ``host_response_rate``, ``host_listings_count``, ``host_total_listings_count``, ``accommodates``, ``bathrooms``, ``bedrooms``, ``beds``, ``price``, ``weekly_price``, ``monthly_price``, ``security_deposit``, ``cleaning_fee``, ``guests_included``, ``extra_people``, ``minimum_nights``, ``maximum_nights``, ``number_of_reviews``,  and ``reviews_per_month``.


One problem though. MultiLabelBinarizer is not compatable with ColumnTransformer. This is mainly because MultiLabelBinarizer works on a single column when we need it work on multiple columns. The great news is that because sklearn is object oriented we can make our own transformer!

<!-- #region -->
**Exercise 1:** Insert the following code into the correct location and complete the missing portions.

Code segment A:
```python
result = list()
for i in range(self.n_columns):
    result.append(self.mlbs[i].transform(X.iloc[:,i]))

result = np.concatenate(result, axis=1)
return result
```

Code segment B:
```python
for i in range(X.shape[1]): # X can be of multiple columns
    mlb = MultiLabelBinarizer()
    mlb.fit(???)
    self.mlbs.append(mlb)
    self.classes_.append(mlb.classes_)
    self.n_columns += 1
return self
```


<!-- #endregion -->

```python
mhe = Lab3_helper.MultiHotEncoder()
mhe.fit(df_airbnb[['amenities_processed']])
encoded = pd.DataFrame(mhe.transform(df_airbnb[['amenities_processed']]),columns=mhe.categories_)
encoded
```

**Exercise 2:** Fill in the correct objects for our column transformer inside Lab3_helper.exercise_2.

```python
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
encoded
```

This output while hopefully useful to predict the review score is not easy to interpret. _Some_ sklearn transformers implement ``get_feature_names()``, but our StandardScaler and MultiOneHot classes do not. We need to fix this problem.

**Exercise 3:** Complete the class that I have provided called ``StandardScalerImproved`` such that it implements ``get_feature_names()``.

```python
scaler = Lab3_helper.StandardScalerImproved()
scaler.fit(df_airbnb[standard_scaler_columns])
encoded = pd.DataFrame(scaler.transform(df_airbnb[standard_scaler_columns]),columns=scaler.get_feature_names())
encoded
```

**Exercise 4:** Our other class, MultiHotEncoder, has the same problem, but it's a little trickier because inside that object is a list of MultiLabelBinarizers. This is a relatively minor adjustments, and it will be our next exercise to implement get_feature_names() for MultiHotEncoder.

```python
mhei = Lab3_helper.MultiHotEncoderImproved()
mhei.fit(df_airbnb[multi_label_columns])
encoded = pd.DataFrame(mhei.transform(df_airbnb[multi_label_columns]),columns=mhei.get_feature_names())
encoded
```

**Exercise 5:** Let's put our new classes to work, and make sure everything works. Fill in the correct objects for our column transformer inside Lab3_helper.exercise_5.

```python
one_hot_columns = ['zipcode', 'room_type', 'property_type', 'bed_type', 'instant_book', 'superhost']
multi_label_columns = ['amenities_processed']
standard_scaler_columns = ['years_host', 'host_response_rate', 'host_listings_count', 
                           'host_total_listings_count', 'accommodates', 'bathrooms', 'bedrooms', 
                           'beds', 'price', 'weekly_price', 'monthly_price', 'security_deposit', 
                           'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights', 
                           'maximum_nights', 'number_of_reviews',  'reviews_per_month']
ct = Lab3_helper.exercise_5(one_hot_columns,multi_label_columns,standard_scaler_columns)
index_values = df_airbnb[standard_scaler_columns].dropna().index
X = df_airbnb.loc[index_values].drop('review_scores_rating',axis=1)
y = df_airbnb.loc[index_values]['review_scores_rating']

encoded = ct.fit_transform(X)
feature_names = ct.get_feature_names()
feature_names
```

We now have everything prepared to ``fit`` our models. We will do this using both linear regression and k-nearest neighbor. We will compare the performance of these models first using all the training data, and then we will use cross-validation to select the ``k`` value hyper-parameter.


**Exercise 6:** Construct a linear regression that predicts ``review_scores_rating`` using the column transformer we constructed above.

```python
from sklearn.metrics import mean_absolute_error, r2_score

one_hot_columns = ['zipcode', 'room_type', 'property_type', 'bed_type', 'instant_book', 'superhost']
multi_label_columns = ['amenities_processed']
standard_scaler_columns = ['years_host', 'host_response_rate', 'host_listings_count', 
                           'host_total_listings_count', 'accommodates', 'bathrooms', 'bedrooms', 
                           'beds', 'price', 'weekly_price', 'monthly_price', 'security_deposit', 
                           'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights', 
                           'maximum_nights', 'number_of_reviews',  'reviews_per_month']
ct = Lab3_helper.exercise_5(one_hot_columns,multi_label_columns,standard_scaler_columns)

# Remove samples that could cause a problem because of missing values
index_values = df_airbnb[standard_scaler_columns].dropna().index
X = df_airbnb.loc[index_values].drop('review_scores_rating',axis=1)
y = df_airbnb.loc[index_values]['review_scores_rating'] # What we are trying to predict

model = Lab3_helper.exercise_6(ct,X,y)
y_ = model.predict(X)
mean_absolute_error(y,y_),r2_score(y,y_)
```

**Exercise 7:** Construct a k-nearest neighbor regressor with ``k=10`` that predicts ``review_scores_rating`` using the column transformer we constructed above.

```python
from sklearn.metrics import mean_absolute_error, r2_score

one_hot_columns = ['zipcode', 'room_type', 'property_type', 'bed_type', 'instant_book', 'superhost']
multi_label_columns = ['amenities_processed']
standard_scaler_columns = ['years_host', 'host_response_rate', 'host_listings_count', 
                           'host_total_listings_count', 'accommodates', 'bathrooms', 'bedrooms', 
                           'beds', 'price', 'weekly_price', 'monthly_price', 'security_deposit', 
                           'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights', 
                           'maximum_nights', 'number_of_reviews',  'reviews_per_month']
ct = Lab3_helper.exercise_5(one_hot_columns,multi_label_columns,standard_scaler_columns)

# Remove samples that could cause a problem because of missing values
index_values = df_airbnb[standard_scaler_columns].dropna().index
X = df_airbnb.loc[index_values].drop('review_scores_rating',axis=1)
y = df_airbnb.loc[index_values]['review_scores_rating'] # What we are trying to predict

model = Lab3_helper.exercise_7(ct,X,y)
y_ = model.predict(X)
mean_absolute_error(y,y_),r2_score(y,y_)
```

**Problem 2:** Compare the results from our work in exercise 6 and 7. Is there a clear "winner" in terms of performance between our k-nearest neighbors approach and the linear regression. If not, which metric indicates that the linear model is better?

**Your answer here**


**Problem 3:** For your linear regression model, create a bar chart showing the top 20 coefficients sorted by the |coefficients|. See if you can get something similar to my graph. I used ``barh`` to get the horizontal bar graph.

HINTS: You might/probably will want to use the following commands:

```python
model.steps[-1]
```

```python
model.steps[-2]
```

```python
model = Lab3_helper.exercise_6(ct,X,y)
# Your solution here
```

So what is listing count? According to some digging, it is not suprisingly the total number of listings for a host. There is a high coefficient in the linear model for this feature when predicting a rating for a property. It seems like hosts with a high total listing count result in higher reviews. This could be because they are better hosts and thus because they have more properties, but it also seems there might be algorithm bias. Hosts with few listings may not get as much attention in search results from Airbnb, and therefore, they are not able to rebound after a few bad reviews. Interpretation is one of the most difficult things in data science, and thus, arguably a more important area to study than other data science topics. In many instances, interpretation requires significant domain expertise, and understanding of the methodology. 


**Problem 4:** Complete the same analysis and produce a similar plot, but remove the columns related to host listing count. I removed them in the variables below for you.

```python
from sklearn.metrics import mean_absolute_error, r2_score

one_hot_columns = ['zipcode', 'room_type', 'property_type', 'bed_type', 'instant_book', 'superhost']
multi_label_columns = ['amenities_processed']
standard_scaler_columns = ['years_host', 'host_response_rate','accommodates', 'bathrooms', 'bedrooms', 
                           'beds', 'price', 'weekly_price', 'monthly_price', 'security_deposit', 
                           'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights', 
                           'maximum_nights', 'number_of_reviews',  'reviews_per_month']
ct = Lab3_helper.exercise_5(one_hot_columns,multi_label_columns,standard_scaler_columns)

# Remove samples that could cause a problem because of missing values
index_values = df_airbnb[standard_scaler_columns].dropna().index
X = df_airbnb.loc[index_values].drop('review_scores_rating',axis=1)
y = df_airbnb.loc[index_values]['review_scores_rating'] # What we are trying to predict

model = Lab3_helper.exercise_6(ct,X,y)
# Your solution here
```

I think we are now getting somewhere! But how, for example, do we determine what x4 and x5 are? They are from our one host columns and they start at x0. So x4 is ``instant_book``, and what do we know :), instant book is important for predicting customer satisfaction. Now that is some insight that a company can use. Good job data scientist! The type of room is also very important (private/entire/shared). The identifier x5 is ``superhost``. 


## Hyperparameter tuning

We want to compare linear regression to k-nearest neighbor, but we arbitrarily chose ``k``. We know how to perform a grid search from the chapter, so let's get to it!

**Exercise 8:** Run grid search using sklearn and find the best parameters for n_neighbors. More details are in the helper file. 

```python
from sklearn.metrics import mean_absolute_error, r2_score

one_hot_columns = ['zipcode', 'room_type', 'property_type', 'bed_type', 'instant_book', 'superhost']
multi_label_columns = ['amenities_processed']
standard_scaler_columns = ['years_host', 'host_response_rate','accommodates', 'bathrooms', 'bedrooms', 
                           'beds', 'price', 'weekly_price', 'monthly_price', 'security_deposit', 
                           'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights', 
                           'maximum_nights', 'number_of_reviews',  'reviews_per_month']
ct = Lab3_helper.exercise_5(one_hot_columns,multi_label_columns,standard_scaler_columns)

index_values = df_airbnb[standard_scaler_columns].dropna().index
X = df_airbnb.loc[index_values].drop('review_scores_rating',axis=1)
y = df_airbnb.loc[index_values]['review_scores_rating'] # What we are trying to predict
pipeline = Lab3_helper.exercise_7(ct,X,y)

best_params = Lab3_helper.exercise_8(pipeline,X,y)
best_params
```

<!-- #region -->
So are we ready to compare k-nearest neighbors to linear regression? ... Well not so fast. We have now run out of data. Whenever, you compare two models you need to treat them both the same. We want to compare the best k-nearest neighbor to linear regression... This is often where things get a little off the rails for some data scientists. What we need to do is be careful about how we are going to nest our validation and hyperparameter tuning. We want to estimate the test set error, so we want to run:
```python
cross_val_score(best_knn_pipeline,...) vs cross_val_score(linear_pipeline,...)
```
The linear pipeline is relatively easy since we are not performing any hyperparameter tuning. The best_knn_pipeline is not necessarily going to be the same for each fold of the cross-validation because the training data is not the same. Luckily for us, sklearn can handle all of this for us if we nest things properly. The key difference is that instead of calling ``.fit`` on GridSearchCV directly, we are going to pass this object directly to cross_val_score. i.e.,
```python
cross_val_score(grid_search_cv_object,...)
```
Let's give it a try in the next exercise!

**Exercise 9:** Compare a hypertuned k-nearest neighbor regressor to a linear regressor using cross_val_score and GridSearchCV. See the helper file for more details.
<!-- #endregion -->

```python
# Remove samples that could cause a problem because of missing values
index_values = df_airbnb[standard_scaler_columns].dropna().index
X = df_airbnb.loc[index_values].drop('review_scores_rating',axis=1)
y = df_airbnb.loc[index_values]['review_scores_rating'] # What we are trying to predict
r2_knn,r2_linear = Lab3_helper.exercise_9(X,y)
r2_knn,r2_linear
```

**Problem 5:** What do these numbers mean? Answer the following. Are they any good? Are they better than guessing the mean rating for all listings? What does the negative number mean?

**Your answer here**

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
