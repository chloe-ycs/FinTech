# FinTech Project Background


# 1. Data Preparation and EDA
- Numerical and Categorical Data explore
- Label Target

# 2. Faeture Engineer
- Datetime to numeric feature
- Work year length(string) to numeric feature:emplength
- % to numeric feature:intrate, revolutil
- Category feature encoding : grade, subgrade
- Zip Code - frequency encoding
- Addr_state - frequency encoding
- One hot encoding - dummy features

# 3. Train Model
- XGBoost - Preliminary manually parameter tuning based on stratified train-test split
- ROC Curve
- Prediction
- Feature importance

# 4. Model Tunning
- Hyperparameter Tuning
- Retrain model with tuned parameters
- Validate on test data

## Reference
Data Source:LendingClub Statistics historical data 2014 <br>
https://www.lendingclub.com/info/download-data.action
