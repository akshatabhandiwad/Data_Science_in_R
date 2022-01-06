# Data Science in R

Following are some of the small data science projects I did, using the R programming language. Each of these folders contains a dataset, a .rmd (R Markdown document) consisting of the code written in R, and .md (a readme file) consisting of the output. 


**1. Spam Classifier** 

This project aims to create a classifier that can separate spam from non-spam email messages. [The dataset](https://archive.ics.uci.edu/ml/datasets/spambase) from UCI Machine Learning Repository is used for this project. 
Before running Linear Discriminant Analysis (LDA), the data is partitioned into training and validation sets  in 80-20 proportion. Also, the data set is standardized using the Caret package.


**2. Hitters Salary Prediction** 

This project aims to predict the salaries of the Baseball players. Hitters data set from the ISLR package is used for this project. Among other information, it contains annual salary of baseball players (in thousands of dollars) on opening day of the season.
Training dataset is created with 80% of the observation. Decision tree (Regression Tree) is used to outline the definition and Ensemble Methods (Bagging and Boosting) to produce the optimal predictive model.

