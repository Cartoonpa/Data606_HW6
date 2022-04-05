# Data606_HW6
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from sklearn.linear_model import LogisticRegression
import pymc3 as pm
import statsmodels.api as sm

## The Dataset

You can download the dataset from here: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

## 1 point

## ToDo: read the csv file into a dataframe and show the first 5 rows
diab_df = pd.read_csv('diabetes.csv')
diab_df.head()

## 3 points

## Assign the Outcome variable to y and the rest to X.
## USe LogisticRegression to fit the data and print out the intercept and the coefficients

y = diab_df['Outcome']
x = diab_df.iloc[:, :-1]

logit_model=sm.Logit(y,x)
result=logit_model.fit()
print(result.summary())

## 2 points
 
## Explain what the code is doing:
## What are the prior probabilities of the intercept and coefficients?
# Which likelihood distribution has been used?              
## What does pm.invlogit(linreg) mean?
## What is map_est?

with pm.Model() as logreg_model:
  
    w0 = pm.Normal('w0', mu=0, sd=100)
    w1 = pm.Normal('w1', mu=0, sd=100)
    w2 = pm.Normal('w2', mu=0, sd=100)
    w3 = pm.Normal('w3', mu=0, sd=100)
    w4 = pm.Normal('w4', mu=0, sd=100)
    w5 = pm.Normal('w5', mu=0, sd=100)
    w6 = pm.Normal('w6', mu=0, sd=100)
    w7 = pm.Normal('w7', mu=0, sd=100)
    w8 = pm.Normal('w8', mu=0, sd=100)
   
    
    linreg = w0 * np.ones(diab_df.shape[0]) + w1 * diab_df.Pregnancies.values + w2 * diab_df.Glucose.values \
    + w3 * diab_df.BloodPressure.values + w4 * diab_df.SkinThickness.values + w5 * diab_df.Insulin.values + \
    w6 * diab_df.BMI.values + w7 * diab_df.DiabetesPedigreeFunction.values + w8 * diab_df.Age.values
    p_outcome = pm.invlogit(linreg)

    likelihood = pm.Bernoulli('likelihood', p_outcome, observed=diab_df.Outcome.values)

    
    map_est= pm.find_MAP()
    print(map_est)

This code implements a Bayesian logistic regression. It creates 9 random variables with a normal distribution with a mean of 0 and a standard deviation of 100. Then using Bernoulli to calculate the likelihood, and the find_MAP function for getting the maximum a posteriori. 



## 2 points
with logreg_model:
    ## ToDo: draw 400 samples using pm.Metropolis() and assign to the variable trace

    start_val = pm.find_MAP(model = logreg_model)
    step = pm.Metropolis()
    trace = pm.sample(400, step = step , start = start_val)
## Explain the output of the plot 
az.plot_posterior(trace)

The output above displays the distribution of the parameter using the Metropolis algorithm from pymc3. Also, the output allows us to analyze the posterior. Finally, the graphs show that only taking 400 samples is not enough.

Link to Github: 
