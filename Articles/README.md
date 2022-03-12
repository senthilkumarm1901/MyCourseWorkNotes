This directory will contain notes from some really good blog articles I have read (or saved for reading)

# [Analysis of 10 ML Case Studies](https://medium.com/@tzjy/10-wonderful-machine-learning-case-studies-from-tech-company-blogs-860ad7b8d1b0)

## Prediction of Value of Homes in Airbnb (article came in 2017)

**Airbnb Background**:<br>
- ML Infra team built `feature repository` that can be used by Data Scientists to develop better ML models
- ML Infra team provided a framework to convert Jupyter Notebooks to Airflow (similar to kubeflow) pipelines
- Data Scientists leveraged several AutoML tools

Type of Problem solved: **Customer Lifetime Value (LTV) modeling**: <br>
- LTV captures the `value` of a customer or group of customers over their `life-long` or `total` relationship with a company
- Useful in setting realistic budget requirements to marketing investments or customer acquisition programmes
- If LTV is high/low for a particular segment of customers, allocate marketing budgets appropriately to reach that segment.

In the context of **Airbnb**, we are going to predict the LTV of new listings using Machine Learning

Say, predicting LTV of a new listing in London for a fixed time period like 6 months, means calculating <br>
- the count of transactions (bookings) to be made in the next 6 months (N) and 
- the average revenue from each transaction aka booking (R)
<br>
Simply put,
> `LTV = N * R`

Note: Here, `lifetime` may just be 6 months. 


Four steps they followed to build a LTV model (a regression problem): <br>
- **0. Data Understanding & 1. Feature Engineering**:
    -  Using `Zipline` internal repository add suitable features at various granularity.
    -  Some key features for Airbnb to predict the LTV of a house
           - Location of the house
           - One Night Price
           - No. of bookings available/occupied in the next 6 months
           - Reviews (rating/talking about sub-topics like cleanliness, amenities, ambience, etc.,) 
    - Understand what are the numeric and categorical variables. 
    - Split the dataset wisely (so that there would be no data leakage)
    - Perform different transformations (imputation and categorical encoding, PCA, etc) on the numeric and categorical data  
    
- **2. Model Training**:
    - `Fit` various regression models (tune hyperparameters) on historical data consisting of the above `engineered or original features` as Input and 
    - their Value (revenue) generated in the last 6 months as Output


- **3. Model Selection** based on several metrics:
    - Measure against a common accuracy metric on the best-fit model in each type of model category


- **4. Deployment**
-  Using `ML Automator` that translates notesbooks to Airflow pipelines

![image](https://user-images.githubusercontent.com/24909551/158014451-aae9b4cc-7f4a-481c-bc11-cd3a38871cc2.png)

- In some cases, just using RMSE to select a model may not be enough. We need to consider "interpretability" too. 
 
[Source](https://medium.com/airbnb-engineering/using-machine-learning-to-predict-value-of-homes-on-airbnb-9272d3d4739d)
