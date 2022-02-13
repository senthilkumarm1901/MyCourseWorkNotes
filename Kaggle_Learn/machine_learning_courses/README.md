**Notes from code-first `Kaggle Learn` Courses on ML, Feature Engineering, Data Cleaning and Time Series Forecasting**

## Intro to ML
Key Learnings: <br>
- How to `train_test_split` the data
- Briefly discussed the concept of underfitting and overfitting (Loss vs model complexity curve)
- How to train a typical scikit-learn model like `DecisionTreeRegressor` or `RandomForestRegressor` 
    - both need no scaling of continuous or discrete data;
    - for sklearn might have to convert categorical data into encoded values
- After finding out the best parameters, one should train with the identified hyperparameters on the whole data 
    - (so that model will learn a bit more from held out data too) 
        
key scikit-learn modules used: <br>
- `from sklearn.tree import DecisionTreeRegressor` AND `from sklearn.ensemble import RandomForestRegressor`
- `from sklearn.model_selection import train_test_split`
- `from sklearn.metrics import mean_absolute_error`


## Intermediary ML
Key Learnings:<br>
- Missing Value Treatment:
    - Remove the Null Rows OR Columns (by column meaning the whole feature containing the missing value)
    - Impute (by some strategy like Mean, Median, some regression like KNN)
    - Impute + Add a boolean variable for every column imputed (so as to make the model hopefully treat the imputed row differently)
- Do removing missing values help or imputing missing values help more for the model accuracy?
- Opinion shared by the Author: SimpleImputer works as effectively as a complex imputing algorithm when used inside sophisticated ML models
- Categorical Column Treatment:

key scikit-learn modules used:
- `from sklearn.impute import SimpleImputer, KNNImputer`
- `from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, `

        - 
