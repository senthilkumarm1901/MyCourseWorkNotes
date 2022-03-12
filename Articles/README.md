This directory will contain notes from some really good blog articles I have read (or saved for reading)

# [Analysis of 10 ML Case Studies](https://medium.com/@tzjy/10-wonderful-machine-learning-case-studies-from-tech-company-blogs-860ad7b8d1b0)

## 1. Prediction of Value of Homes in Airbnb (article came in 2017)

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
 
[Source](https://medium.com/airbnb-engineering/using-machine-learning-to-predict-value-of-homes-on-airbnb-9272d3d4739d)\


## 2. Improve `streaming quality` using Machine Learning (Company: Netflix) 
[article written in 2018](https://netflixtechblog.com/using-machine-learning-to-improve-streaming-quality-at-netflix-9651263ef09f)

How `streaming quality` is measured:
- Video Quality
- Amount of wait time for the video during initial load or playback
- Amount of bufferring 

How `streaming quality` can be improved:
- Network quality characterization and prediction
    - Will network be more congested always? or during some time of the day?    
- `Predictive Caching`
- Adaptive Video Quality (for each chunk, what is the best quality that we can provide)
- Device Anamolies (may be changes to infra or UI is affecting performance?)

## 3. 6 lessons from training 150 ML models in booking.com [source](https://medium.com/@tzjy/10-wonderful-machine-learning-case-studies-from-tech-company-blogs-860ad7b8d1b0)

- 1. ML Models do deliver businss impact
    - All projects that incorporated ML had higher returns compared to projects that did not have ML
- 2. ML model performance improvement != business performance improvement
    - Market is saturated (no matter how well you predict, it won't translate into business value)
    - Over-optimization to `proxy metric` (e.g.: clicks) whereas `actual metric` (e.g.: conversion) does not happen
    - Uncanny valley effect - Predicting too much that users think it is invading into their privacy
- 3. Being clear on the objective of the problem
    - E.g.: Predicting user preference by click vs predicting user preference by NLP
- 4. Prediction serving latency (speed of model inferencing) matters
    - If latency (in the nlp world, amount of time to process a single sentence) increases by 30%, the conversion drops by 0.5% 
    - Reducing the time for model inferencing reduces latency.  
- 5. Get early feedback on model quality
    - Real-time early feedback is always difficult. 
    - `Response Distribution Analysis`: If the model is not confident between its highest predicted class and the second highest predicted class, then it is struggling to learn the distinction among the classes
- 6. Test the business impact of models through randomized controlled trials
![image](https://user-images.githubusercontent.com/24909551/158018419-6a781cc7-fee6-49bd-8957-cfa9f0eded24.png)
- How much % of the exposed/treated group showed an outcome to a new treatment viz-a-viz the % of controls/unexposed group showed an outcome with standard treatment?




