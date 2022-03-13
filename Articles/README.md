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


## 4. How `Chicismo` app grew its women user base ([source](https://medium.com/hackernoon/how-we-grew-from-0-to-4-million-women-on-our-fashion-app-with-a-vertical-machine-learning-approach-f8b7fc0a89d7))

- Used Collaborative Filtering Process to recommend right fashion preferrable to their taste

![image](https://user-images.githubusercontent.com/24909551/158019078-584d8cd9-6729-4870-98dd-e4de8b68de5e.png)

## 5. Search Ranking of Airbnb Experiences | [Source](https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789)

![image](https://user-images.githubusercontent.com/24909551/158026138-1f5210ca-7374-41ae-810b-d662d9974063.png)

- In Stage 1: Everyone had the same ranking of experiences
- In Stage 2: Started personalizing with user features
- In Stage 3: Used online scoring (query features available at the time of querying used in inferencing)

Example Experience Features: <br>
- Experience duration
- Price and Price/hour
- Category of Experience (music class, surfing, cooking)
- Review (rating, number of reviews, <s>textual comments </s>)
- Number of bookings for that experience
- Click-Through-Rate

Prediction Variable: Search Ranking based on probability of booking
Model Used: Gradient Boosted Decision Trees (GBDT)
- No need to scale features
- No need to treat missing values
- Changes in raw counts in a non-linear model like GBDT can influence predictions so much
    - Instead of `no. of bookings`, can use the ratio of `no. of bookings per 1000 viewers`


## 6. Fraud Detection in Lyft ([article writtent in 2018](https://eng.lyft.com/from-shallow-to-deep-learning-in-fraud-9dafcbcef743)

- About `Lyft`: It is a ride-hailing app | competitor to `Uber`
- Fraud detection is a classification problem
- Initially a logistic regression model was employed. Enabled in understanding `important features`
- As fraud techniques became more complex, complex models were used
    - Upgrading model from Logistic Regression to Gradient-boosted Decision Trees, improved precision by 60% for the operating recall and set of features 
    - Later moved from GBDT to DL models
  
![image](https://user-images.githubusercontent.com/24909551/158047351-73732bdb-e470-4bb7-96a7-c87c6fb91820.png)

- Discussion in the article: Complexity vs Interpretability | Performance vs Ease of Deployment
- Talks about roles of Model Development vs Model Deployment teams

![image](https://user-images.githubusercontent.com/24909551/158047494-32596aff-b662-463d-9790-fef8a7875a4c.png)

## 7. Path Optimization ([article written in 2017](https://tech.instacart.com/space-time-and-groceries-a315925acf3a))

- About `Instacart`: It is a groceries delivery company | Similar to Bigbasket
- Optimizing path for efficient and timely delivery of groceries to households in the US


## 8. OCR + Word Detector Model for Dropbox ([source](https://dropbox.tech/machine-learning/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning))

- Utilized in Dropbox product `mobile document scanner`

## 9. Uber Big Data Platform ([source](https://eng.uber.com/uber-big-data-platform/))

- Handling 100+ Petabytes of data with minute legacy

## 10. Uber Michalengelo: ([source](https://eng.uber.com/scaling-michelangelo/))

- Scaling ML models in production
