# FinTech - Predict the defalut rate of loan status
## Project Background
Lending Club is the world leader in p2p lending having issued over $9 billion dollars in loans since they began in 2007. <br>

The borrowers apply for a loan and if they meet certain criteria, their loan is added to Lending Club’s online platform. Investors can browse the loans on the platform and build a portfolio of loan, and they profit from interest payments on loans. <br>

Lending Club categorizes borrowers into seven different loan grades: A through G,and each of them assigned with different interest rates and risks.Where a borrower is graded depends on many factors the most important of which is the data held in the borrower’s credit report. <br>

From the investors' point of view, deciding which loans to invest in, or gauging the ongoing performance of a portfolio requires investors to be able to predict how much a given loan will return before reaching maturity. <br>

The target of this project is to give a model to predict the default rate of loanstatus for newly issued loans.
This model makes the investors choose more easily from thousands of available loans at Lending Club, by using machine learning to calculate which notes are more likely to perform better than others. 


## Required Python Libraries
- numpy, pandas: numeric data and dataframe operation
- matplotlib, seaborn: data visualizatin
- sklearn: sci-kit Learn for model buidling and evaluation
- xgboost: XGBoost algorithm
- bayes_opt: Bayesian Optimization

## 1. Data Preparation and EDA
- Numerical and Categorical Data explore
- Label Target 
### Label Target
The Feature 'loanstatus' is used as target. Among all the 7 statuses, "Fully Paid" and "Charged Off" refer to those loans whose statuses are fixed, while other statuses may still change over time. 
In this project, only the loans with statuses of "Fully Paid" and "Charged Off" are considered. They are labeled as "0" (no default) and "1" (default), respectively.


## 2. Faeture Engineer
| Type	| Feature Name |	Definition	|Operation|
| ------------- | ------------- |------------- | ------------- |
| datetime | earliestcrline |the date the borrower's earliest reported credit line was opened| split to numeric month/year|
| string  | emplength  |employment length in years|replace n/a; work year length to numeric|
| essentially numerical | intrate |interest rate on the loan| % to numerical|
| essentially numerical | revolutil | the credit amount the borrower is using relative to all available revolving credit| % to numerical|
| string | grade |LC assigned loan grade| category feature encoding|
| string | subgrade |LC assigned loan subgrade| category feature encoding|
| string | zip Code |the first 3 numbers of the zip code provided by the borrower| encoding by frequency|
| string | emptitle |job title of the borrower| encoding by frequency|
| string | addr_state |the state of the borrower | encoding by frequency|
| string | homeownership |home ownership status of the borrower: RENT/OWN/MORTGAGE/OTHER| one-hot encoding |
| string | verificationstatus |if the co-borrowers' joint income was verified by LC| one-hot encoding |
| string | purpose |purpose of loan | one-hot encoding |
| string | initialliststatus |initial listing status of the loan | one-hot encoding |

## 3. Train Model
- XGBoost - Preliminary manually parameter tuning based on stratified train-test split
- Model evaluation: ROC Curve
- Prediction
- Feature importance

## 4. Model Tunning
- Hyperparameter Tuning
- Retrain model with tuned parameters

## 5. Model Evaluation
- Feature importance
- Validate on test data
- Model evaluation: ROC Curve
- Thresholds

### Tuned best parameters:
- colsample_bytree'= 0.32221742682392707,
- gamma'= 1.888745445753438,
- max_depth'= 4,
- min_child_weight'= 19.984365517870163,
- subsample'= 0.7112055441024843

- learning_rate = 0.01
- n_estimators = 1500
- seed = 1441
- nthread = -1
- scale_pos_weight = 1
- eval_metric= 'auc'

### Model evaluation:
#### Feature importance
The top 10 important features based on tuned models include Debt to Income Ratio, Installment, Annual Income, Interest Rate, Number of trades in past 24 months and so on.

#### ROC Curve
The predicted default rate on the test set is about 0.1355, very close to the average default rate of around 0.1373. 
The result of model performance is visualized via the ROC curve. The AUC scores on the training and validation sets are about 0.70 and 0.74, and the AUC score on the testing set is about 0.70.

#### Thresholds
The best threshold of predicting default is determined based on Index value(= tpr - fpr), which optimizes the sensitivity and specificity.
The 'TPR-FPR' maximizes at a threshold of 0.15. The maximum f1-score at the threshold of 0.15 is 0.34876, with a recall rate of 0.62, suggesting that 62% of total defalut loans is correctly classified by the model.


## Data Source
LendingClub Statistics historical data 2014 <br>
https://www.lendingclub.com/info/download-data.action


### Reference
[1] https://www.lendacademy.com/lending-club-review/  <br>
[2] http://blog.lendingrobot.com/research/predicting-the-number-of-payments-in-peer-lending/ <br>


