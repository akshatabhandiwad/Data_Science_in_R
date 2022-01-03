Spam Classifier - Linear Discriminant Analysis (LDA)
================
2022-01-03

##### Loading required packages

``` r
if(!require("pacman")) install.packages("pacman")
pacman::p_load(esquisse, forecast, tidyverse, gplots,data.table, caret, MASS, gains)

search()
```

#### Loading the data

``` r
data <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"),
                 header = FALSE)
dataset <- data.frame(data)
```

#### Renaming the columns

``` r
# For Renaming Columns                    
name <- read.csv(url('https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names'), 
                 header = FALSE, skip = 33)
name$V2 <- gsub("continuous.","", name$V1)
name$V3 <- gsub(":","", name$V2)

col.names <- c(name$V3,'Spam')

# Assigning Column names 
colnames(dataset) <- col.names
names(dataset) <- gsub("\\.","",names(dataset))

# Converting Spam column to factor column
dataset$Spam <- as.factor(dataset$Spam)
```

#### Data Partition and Standardization

``` r
options(scipen = 999)

# Data partition
set.seed(42)
train.index <- createDataPartition(dataset$Spam, p = 0.8, list = FALSE)
spam.train <- dataset[train.index, ]
spam.valid <- dataset[-train.index, ]


# standardize the data
# Estimate pre-processing parameters (excluding spam column)
norm.values  <- preProcess(spam.train[,1:57], method = c("center", "scale"))


# Transforming the data using the estimated parameters
spam.train.norm <- predict(norm.values, spam.train)
spam.valid.norm <- predict(norm.values, spam.valid)
```

## 1. Comparing the ‘Spam’ average and ‘Non-spam’ average and predictor identification

``` r
#Column Mean by spam group 
data.mean <- spam.train.norm %>%
                group_by(Spam) %>% 
                summarise_all(mean) 
  
#Transposing without spam column
mean.data <- as.data.frame(as.matrix(t(data.mean[,-1])))

# adding column names
names(mean.data) <- c("NonSpam","Spam")


# absolute difference of means
mean.data$Difference <- abs(mean.data$NonSpam - mean.data$Spam)
mean.data <- mean.data[order(mean.data$Difference,decreasing=TRUE),]
mean.data$Columns <- rownames(mean.data)


#Choosing Top 10 Columns
selected.columns <- mean.data$Columns[1:10]
data.frame(selected.columns)
```

    ##               selected.columns
    ## 1      word_freq_your         
    ## 2      word_freq_000          
    ## 3      word_freq_remove       
    ## 4      char_freq_$            
    ## 5      word_freq_you          
    ## 6      word_freq_business     
    ## 7      word_freq_hp           
    ## 8      word_freq_free         
    ## 9      char_freq_!            
    ## 10 capital_run_length_total

``` r
#Final dataset with 10 predictors from training data
final.data1 <- spam.train.norm[,c(selected.columns)]
final.data2 <- data.frame (spam.train.norm[,c('Spam')])
colnames(final.data2) <- 'Spam'

final.data.train <-  cbind(final.data1,final.data2)
```

##### The top ten predictors for which difference between the spam-class average and the nonspam class average is the highest are word\_freq\_your, word\_freq\_000, word\_freq\_remove, char\_freq\_$, word\_freq\_you, word\_freq\_business, word\_freq\_hp, word\_freq\_free, char\_freq\_! , capital\_run\_length\_total

 

## 2. Linear discriminant analysis using the training dataset with top 10 predictors

``` r
# LDA on training data
lda <- lda(Spam~., data = final.data.train)
lda
```

    ## Call:
    ## lda(Spam ~ ., data = final.data.train)
    ## 
    ## Prior probabilities of groups:
    ##         0         1 
    ## 0.6059207 0.3940793 
    ## 
    ## Group means:
    ##   `word_freq_your         ` `word_freq_000          ` `word_freq_remove       `
    ## 0                -0.3183569                -0.2723733                -0.2616917
    ## 1                 0.4894929                 0.4187903                 0.4023667
    ##   `char_freq_$            ` `word_freq_you          ` `word_freq_business     `
    ## 0                -0.2467906                -0.2308880                -0.2082196
    ## 1                 0.3794554                 0.3550043                 0.3201502
    ##   `word_freq_hp           ` `word_freq_free         ` `char_freq_!            `
    ## 0                 0.2060657                -0.2027840                -0.2012532
    ## 1                -0.3168384                 0.3117927                 0.3094390
    ##   `capital_run_length_total   `
    ## 0                    -0.2004815
    ## 1                     0.3082524
    ## 
    ## Coefficients of linear discriminants:
    ##                                      LD1
    ## `word_freq_your         `      0.4522563
    ## `word_freq_000          `      0.3628565
    ## `word_freq_remove       `      0.4248817
    ## `char_freq_$            `      0.2463159
    ## `word_freq_you          `      0.1948910
    ## `word_freq_business     `      0.2008115
    ## `word_freq_hp           `     -0.2386762
    ## `word_freq_free         `      0.2954577
    ## `char_freq_!            `      0.2761332
    ## `capital_run_length_total   `  0.3446591

 

## 3. Prior probabilities

``` r
prior.prob <- lda$prior
prior.prob
```

    ##         0         1 
    ## 0.6059207 0.3940793

##### The prior probabilities is the proportion of each Spam class and Non-spam class before performing linear discriminant analysis on the data. The prior probability of Spam class is 39.4% and Non-spam class is 60.6%

 

## 4. Coefficients of linear discriminants

``` r
# Coefficients of LDA
lda$scaling
```

    ##                                      LD1
    ## `word_freq_your         `      0.4522563
    ## `word_freq_000          `      0.3628565
    ## `word_freq_remove       `      0.4248817
    ## `char_freq_$            `      0.2463159
    ## `word_freq_you          `      0.1948910
    ## `word_freq_business     `      0.2008115
    ## `word_freq_hp           `     -0.2386762
    ## `word_freq_free         `      0.2954577
    ## `char_freq_!            `      0.2761332
    ## `capital_run_length_total   `  0.3446591

##### The coefficient of linear discriminants gives us the weightage of each of the variable. The LD’s formed here are used to create a separation line between the classes (Spam/Non-Spam). The coefficients of linear discriminants help us to understand the impact of variables on the separation line. The higher weight variables have more impact. They contribute to the transformation of original data set into a common scale. The coefficient are multipied by the actual variables to get the linear discriminant scores.

##### In this case, the separation line formula would be given by,

##### Y = 0.4522 \* word\_freq\_your + 0.3628 \* word\_freq\_000 + 0.4248 \* word\_freq\_remove + .. … .. + 0.3446 \* capital\_run\_length\_total.

##### Y = Coeff of LD1 \* Variable 1

##### word\_frequency\_your and word\_freq\_remove LD scores have the highest impact on the separation line.

 

## 5. Linear discriminants using our analysis

``` r
# Sample prediction
sample <- predict(lda, spam.valid.norm[1:10,])
sample
```

    ## $class
    ##  [1] 0 1 1 0 0 1 0 1 1 1
    ## Levels: 0 1
    ## 
    ## $posterior
    ##             0          1
    ## 8  0.94426927 0.05573073
    ## 15 0.01167094 0.98832906
    ## 20 0.21210281 0.78789719
    ## 21 0.91587785 0.08412215
    ## 31 0.63480071 0.36519929
    ## 32 0.37583559 0.62416441
    ## 42 0.62107651 0.37892349
    ## 50 0.38424808 0.61575192
    ## 57 0.04864809 0.95135191
    ## 63 0.37925253 0.62074747
    ## 
    ## $x
    ##           LD1
    ## 8  -1.0883269
    ## 15  2.8069856
    ## 20  1.1314459
    ## 21 -0.8513183
    ## 31  0.1319130
    ## 32  0.7000317
    ## 42  0.1633958
    ## 50  0.6808969
    ## 57  2.0215546
    ## 63  0.6922398

##### The linear discriminants that are generated using the analysis for a sample of 10 observations. The default cut-off value is 0.5. It can be seen that for the first observation, posterior probability is higher for the Non-Spam class and LD score is negative. This is classified as 0 (Non Spam class). Similarly for the second observation, the posterior probability is higher for the Spam class and LD score is positive. This is classified as 1 (Spam class). There are some misclassification like in the case of 5th and 7th observations. But, it can be understood that the generated linear discriminant scores are separating the observations into Spam and Non-Spam class. For Spam class mostly the LD values are greater than 0 and for Non-Spam class mostly the LD values are less than 0.

 

## 6. Linear discriminants are in the model

##### There is only 1 linear discriminant (LD1) in the model as as there are only two classes : Spam and Non-Spam (Two- class classification problem). The number of linear discriminants is (n-1), where n is the number of classes.

 

## 7. LDA plots using the training and validation data

``` r
# Predict/plot - using Training data
pred.train <- predict(lda, spam.train.norm)

ldahist(pred.train$x[ ,1], g = spam.train.norm$Spam, 
        xlim = c(-7,7),
        ymax =  0.8,
        col = "olivedrab4")
```

![](Spam_Classifier_files/figure-gfm/LDA%20Plots-1.png)<!-- -->

``` r
# Predict/plot - using Validation data
pred.valid <- predict(lda, spam.valid.norm)
ldahist(pred.valid$x[ ,1], g = spam.valid.norm$Spam, 
        xlim = c(-7,7),
        ymax =  0.8,
        col = "olivedrab2")
```

![](Spam_Classifier_files/figure-gfm/LDA%20Plots-2.png)<!-- -->

##### In both the LDA plots of training and validation data, the score less than 0 are mostly classified as Non-Spam and score greater than 0 are mostly classified as Spam. Though there are some misclassification, scores holds true of majority of the observations. The LDA has created separation among the two classes.

##### As the plots look similar for both training and validation datasets, it can be said the model is doing a good job. So, they are the plots are not different.

 

## 8. Sensitivity and Specificity using confusion matrix

``` r
# Confusion matrix Validation data
confusionMatrix(pred.valid$class, spam.valid.norm$Spam,
                positive = '1')
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 526 115
    ##          1  31 247
    ##                                                
    ##                Accuracy : 0.8411               
    ##                  95% CI : (0.8159, 0.8642)     
    ##     No Information Rate : 0.6061               
    ##     P-Value [Acc > NIR] : < 0.00000000000000022
    ##                                                
    ##                   Kappa : 0.6532               
    ##                                                
    ##  Mcnemar's Test P-Value : 0.00000000000646     
    ##                                                
    ##             Sensitivity : 0.6823               
    ##             Specificity : 0.9443               
    ##          Pos Pred Value : 0.8885               
    ##          Neg Pred Value : 0.8206               
    ##              Prevalence : 0.3939               
    ##          Detection Rate : 0.2688               
    ##    Detection Prevalence : 0.3025               
    ##       Balanced Accuracy : 0.8133               
    ##                                                
    ##        'Positive' Class : 1                    
    ## 

##### For the normalized validation data, Sensitivity is 0.6823 and Specificity is 0.9443 with Accuracy of 0.8411 considering class of interest as ‘Spam’.

 

## 9. Lift and Decile charts for the validation dataset

``` r
# Computing the gains
spam.lvl <- as.numeric(levels(spam.valid.norm$Spam))[spam.valid.norm$Spam]

gains <- gains (spam.lvl, pred.valid$posterior[,2], groups = 10)


#Plotting the Lift Chart
plot(c(0,gains$cume.pct.of.total*sum(spam.lvl)) ~ c(0, gains$cume.obs),
     xlab = '# Cases', ylab = 'Cumulative',
     main = "Lift Chart",
     col = "blue1",
     type = "l")

lines(c(0,sum(spam.lvl)) ~c(0,dim(spam.valid.norm)[1]), col = "red",  lty = 2)
```

![](Spam_Classifier_files/figure-gfm/liftchart-1.png)<!-- -->

``` r
#Plotting the Decile Lift Chart
barplot(gains$mean.resp/mean(spam.lvl),
        names.arg = gains$depth,
        space = 1.3,
        ylim = c(0,2.5),
        col = "blue3",
        xlab = "Percentile",
        ylab = "Mean Response",
        main = "Decile-wise Lift Chart",
        border = NA)
```

![](Spam_Classifier_files/figure-gfm/liftchart-2.png)<!-- -->

##### The lift chart shows that the model has gained a significant lift compared to the base model. In the first 400 cases, the model is able to identify nearly 310 Spam cases out of the total Spam cases where as the benchmark model is able to identify only 160 cases. So here we have a lift of 310/160 = 1.9375

##### The model is likely to identify a spam email in the first 3 deciles (30%) of the decile chart, when the observations are ranked by their propensities. Model is able to perform nearly 2.28X times better on average in the top three deciles than the random classification model. It is seen that the second decile is higher than the first decile. So the model is not performing to it’s best standard. There is a possibility of model to exhibit better staircase decile, if the cutoff is further decreased.

 

##10. Using probability threshold of 0.25

``` r
# Confusion matrix for cutoff = 0.25 to spam class

confusionMatrix(as.factor(ifelse(pred.valid$posterior[,2]>= 0.25, 1, 0)), 
                spam.valid.norm$Spam,
                positive = '1')
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 487  52
    ##          1  70 310
    ##                                              
    ##                Accuracy : 0.8672             
    ##                  95% CI : (0.8436, 0.8885)   
    ##     No Information Rate : 0.6061             
    ##     P-Value [Acc > NIR] : <0.0000000000000002
    ##                                              
    ##                   Kappa : 0.7244             
    ##                                              
    ##  Mcnemar's Test P-Value : 0.1238             
    ##                                              
    ##             Sensitivity : 0.8564             
    ##             Specificity : 0.8743             
    ##          Pos Pred Value : 0.8158             
    ##          Neg Pred Value : 0.9035             
    ##              Prevalence : 0.3939             
    ##          Detection Rate : 0.3373             
    ##    Detection Prevalence : 0.4135             
    ##       Balanced Accuracy : 0.8653             
    ##                                              
    ##        'Positive' Class : 1                  
    ## 

##### In this case, we want to classify the spam emails amongst the non-spam ones. Hence, the threshold of 0.25 is applied to the posterior probability of spam class.

#### After applying threshold of 0.25, the accuracy of the model is increased to 0.8672 from 0.8411 (threshold of 0.5). As seen from the confusion matrix, more number of observations are classifed as spam correctly. There is also increase in the Sensitivity from 68% to 85% and decrease in Specificity from 94% to 87%.
