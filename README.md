# FinTech  
## Project Background
Lending Club is the world leader in p2p lending having issued over $9 billion dollars in loans since they began in 2007. <br>

The borrowers apply for a loan and if they meet certain criteria, their loan is added to Lending Club’s online platform. Investors can browse the loans on the platform and build a portfolio of loan, and they profit from interest payments on loans. <br>

Lending Club categorizes borrowers into seven different loan grades: A through G,and each of them assigned with different interest rates and risks.Where a borrower is graded depends on many factors the most important of which is the data held in the borrower’s credit report. <br>

From the investors' point of view, deciding which loans to invest in, or gauging the ongoing performance of a portfolio requires investors to be able to predict how much a given loan will return before reaching maturity. <br>

The target of this project is to build a model that anticipates how future loans will behave and predicts the return, based on information from historic loans. Since the loan amount is known from inception, we ‘only’ need to predict the total amount paid back to investors to calculate the financial return. In other words, estimating the total amount paid back to investor only requires to estimate the number of payments made over a loan’s life, as long as the loan ‘survives’.<br>

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
- Model evaluation: ROC Curve
- Validate on test data
- Feature importance

### Tuned best parameters:
- colsample_bytree': 0.23031944035829088,
- gamma': 1.7060478383477278,
- max_depth': 7.938421102340797,
- min_child_weight': 2.672125154299445,
- subsample': 0.8746946441939958

### Model evaluation:
The predicted default rate on the test set is about 0.2055




## Data Source
LendingClub Statistics historical data 2014 <br>
https://www.lendingclub.com/info/download-data.action


### Reference
https://www.lendacademy.com/lending-club-review/  <br>
http://blog.lendingrobot.com/research/predicting-the-number-of-payments-in-peer-lending/ <br>
