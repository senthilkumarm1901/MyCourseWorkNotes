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
    - `Drop Categorical columns` (worst approach)
    - `OrdinalEncoder` 
    - `OneHotEncoder` (most cases, the best approach)
- Learnt the concept of "good_category_cols" and "bad_category_columns" <br> (if a particular class occurs new in the unseen dataset; `handle_unknown` argument in "OneHotEncoder" is possible)
- Think twice before applying onehot encoding "high cardinality columns"
- `Pipleine`: Bundles together `preprocessing` and `modeling` steps | makes codebase easier for productionalizing
- `ColumnTransformer`: Bundles together different preprocessing steps
- An example of Data Leakage: 
    - Doing Inputer before train_test_split. Validation data then would have "seen" training data  

- `Data Leakage Examples`:
Example 1 - `Nike`: <br>
    - Objective: How much shoelace material will be used?
    - Situation: With the feature `Leather used this month` , the prediction accuracy is 98%+. Without this featur, the accuracy is just ~80%
    - IsDataLeakage?: `Depends` ! 
        :x: `Leather used this month` is a bad feature if the number is populated during the month (which makes it not available to predict the amount of shoe lace material needed)
        ✔️ `Leather used this month` is a okay feature to use if the number determined during the beginning of the month (making it available during predition time on unseen data)

Example 2 - `Nike`: <br>
    - Objective: How much shoelace material will be used? 
    - Situation: Can we use the feature `Leather order this month`?
    - IsDataLeakage? `Most likely no, however ...`
       :x: If `Shoelaces ordered` (our Target Variable) is determined first and then only `Leather Ordered` is planned, <br>
       then we won't have `Leather Ordered` during the time of prediction of unseen data
       ✔️ If `Leather Ordered` is determined before `Shoelaces Ordered`, then it is a useful feature
        
 Example 3 - `Cryptocurrency`: <br>
    - Objective: Predicting tomo's crypto price with a error of <$1
    - Situation: Are the following features susceptible to leakage?
        - `Current price of Crypto`
        - `Change in the price of crypto from 1 hour ago`
        - `Avg Price of crypto in the largest 24 h0urs`
        - `Macro-economic Features`
        - `Tweets in the last 24 hours `
    - IsDataLeakage? `No`, none of the features seem to cause leakage.
    - However, more useful Target Variable `Change in Price (pos/neg) the next day`. If this can be consistently predicted higher, then it is a useful model


Example 4 - `Surgeon's Infection Rate Performance`: <br>
    - Objective: How to predict if a patient who has undergone a surgery will get infected post surgery?
    - Situation: How can information about each surgeon's infection rate performance be carefully utilized while training?
   
   
key scikit-learn modules used:
- `from sklearn.impute import SimpleImputer, KNNImputer`
- `from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
- `from sklearn.model_selection import cross_val_score`
     - cross_val_score(my_pipeline, X, y, scorung='neg_mean_absolute_error')
- `from xgboost import XGBRegressor`
     - `n_estimators`: Number of estimators is same as the number of cycles the data is processed by the model (`100-1000`)
     - `early_stopping_rounds`: Early stopping stops the iteration when the validation score stops improving 
     - `learning_rate`
           - xgboost_model = XGBRegressor(n_estimators=500)
           - xgboost_model.fit(X_train,y_train,early_stopping_rounds=5,eval_set=[(X_valid, y_valid)], verbose=False)

Noteworthy Pandas Lines:
- Exclude all categorical columns at once:
    - `df = df.select_dtypes(exclude=['object'])`
- Creating a new column just to show the model if which row of a particular column have null values
    - df[col + '__ismissing'] = df[col].isnull() 
- Isolate all categorical columns: 
    - `object_cols = [col for col in df.columns if df[col].dtype == "object"]`
- Segregate good and bad object columns (defined by the presence of "unknown" or new categories in validation or test dataset)
    - `good_object_cols = [col for col in object_cols if set(X_valid[col]).issubset(set(X_train[col])]`
    - `bad_object_cols = list(set(good_object_cols) - set(bad_object_cols))`
- Getting number of unique entries (`cardinality`) across `object` or categorical columns
    - `num_of_uniques_in_object_cols = list(map(lambda col: df[col].nunique(), object_cols))`  
    - `sorted(list(zip(object_cols, num_of_uniques_in_object_cols)), key=lambda x: x[1], reverse=True)`
