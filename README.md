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
- sklearn: Sci-kit Learn for model buidling and evaluation
- xgboost: XGBoost algorithm.
- bayes_opt: Bayesian Optimization.

## 1. Data Preparation and EDA
- Numerical and Categorical Data explore
- Label Target

## 2. Faeture Engineer
- Datetime to numeric feature
- Work year length(string) to numeric feature: emplength
- % to numeric feature: intrate, revolutil
- Category feature encoding : grade, subgrade
-Frequency encoding: zip Code
- Frequency encoding: addr_state
- One hot encoding - dummy features

## 3. Train Model
- XGBoost - Preliminary manually parameter tuning based on stratified train-test split
- ROC Curve
- Prediction
- Feature importance

## 4. Model Tunning
- Hyperparameter Tuning
- Retrain model with tuned parameters
- Validate on test data

## Data Source
LendingClub Statistics historical data 2014 <br>
https://www.lendingclub.com/info/download-data.action


### Reference
https://www.lendacademy.com/lending-club-review/
http://blog.lendingrobot.com/research/predicting-the-number-of-payments-in-peer-lending/
