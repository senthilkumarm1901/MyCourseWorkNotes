This directory will contain notes from some really good blog articles I have read (or saved for reading)

<h1 align="center"> Analysis of 10 ML Case Studies </h1>
[Source](https://medium.com/@tzjy/10-wonderful-machine-learning-case-studies-from-tech-company-blogs-860ad7b8d1b0)

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
- Type of Problem: Stage 0 (over-simplified view): `Traveling Salesman Problem` 

![image](https://user-images.githubusercontent.com/24909551/158048447-a69c69ea-a7fc-4945-932e-ebfd0cc69e17.png)

- Stage 1 (simplified view) `Vehicle Routing Problem`

![image](https://user-images.githubusercontent.com/24909551/158048428-076b24e9-6b7b-44c9-81db-c4996bd52024.png)

- Stage 2 (decent but still inadequate view) `Vehicle Routing Problem with Time Windows`

![image](https://user-images.githubusercontent.com/24909551/158048594-1d419dec-aa19-487e-9403-9f258e6df1c3.png)

- Stage 3 (real-life view of the problem) `Capacitated Vehicle Routing Problem with Time Windows and Multiple Routes`
    - Rationale for `Capacitated`: Some have small vehicles and some large. Some cannot transport alcoholo and some can 
![image](https://user-images.githubusercontent.com/24909551/158048651-365cafa6-3beb-4151-b5f9-bc03c0e38f60.png)

Goal of the company: reduce average delivery time
- Shown below as % of max time taken

![image](https://user-images.githubusercontent.com/24909551/158048719-eb1c6337-e458-405c-87bd-4b39d9d1c496.png)

This goal of `logistics fulfillment` in the larger picture:

![image](https://user-images.githubusercontent.com/24909551/158048745-5d4c443b-06d4-4742-8c79-2d7eba25a87b.png)

## 8. OCR + Word Detector Model for Dropbox ([source](https://dropbox.tech/machine-learning/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning))

- The OCR + Word dector pipeline utilized in Dropbox product `mobile document scanner`
- OCR - detects what characters are mentioned in the image
- Word detector - detects where the words are occuring in the image

## 9. Uber Big Data Platform ([source](https://eng.uber.com/uber-big-data-platform/))

- Handling 100+ Petabytes of data with minute legacy
- A fundamental article on the evaluation of Data Engineering at Uber
- Phase 1: `Before 2014`:
    - Used Online Transaction Processing (OLTP) Databse (MySQL, PostgreSQL) 

![image](https://user-images.githubusercontent.com/24909551/158053026-f5b52368-b76a-4bc0-aecf-94bd71c22158.png)

- Phase 2: From OLTP to OLAP
    - All of Uber's Online Analytical Processing needs were saved in a warehouse software known as `Vertica` 
    > *developed multiple ad hoc ETL (Extract, Transform, and Load) jobs that copied data from different sources (i.e. AWS S3, OLTP databases, service logs, etc.) into Vertica*

![image](https://user-images.githubusercontent.com/24909551/158053171-b9ca8d2d-f1cc-41db-ab52-dee07f1df294.png)

- Global View (of all regions)
- All data in one place

- Phase 3: Adding `Hadoop` before `OLAP`

![image](https://user-images.githubusercontent.com/24909551/158053737-9cc9f1f8-8be7-4229-940c-df6e8496a58f.png)

    - In the pic below, before dumping data into `hadoop` database, there is no transformatio done. Only "EL" jobs. But schema enforced which improved reliability of data
    - The "ETL" jobs were performed on only **some** of the data in `hadoop` before loading into OLAP data warehouse `Vertica`. That data used by `city operators` were alone transformed and loaded into `Vertica`
    - How was data accessed in Hadoop - Using 3 different query engines
          - `Presto` - A distributed SQL query engine to enable interactive ad hoc user queries
          - `Apache Spark` for prgrammatic access to small to medium sized queries
          - `Apache Hive` for large queries

- Phase 4: Reduce Data Ingestion latency further   
    - Introduced `Spark Hudi` library 

![image](https://user-images.githubusercontent.com/24909551/158054072-ffe57c38-cfb4-4cbb-b148-f51cbf995395.png)


## 10. Uber Michalengelo: ([article written in 2018](https://eng.uber.com/scaling-michelangelo/))

- Scaling ML models in production using Michelangelo

![image](https://user-images.githubusercontent.com/24909551/158050161-34c344c3-9640-4c30-9459-5f9ad32be168.png)

- 1. ML models in `Uber Eats`:
    - Ranking of Restaurants and Menu items (based on historical data of user purchases and current search query and general features like Ratings)
    - ETA Prediction
- 2. Uber Marketplace Forecasting: 
    - `Spatiotemporal` forecasting models to predict 
          - where rider `demand` will be and 
          - where driver `supply` will be
    - Using the predicted supply-demand imbalances, send drivers to places well-ahead in time so that they get maximum opportunity for rides
- 3. Customer Support
    - Models built at `Michelangelo` used to `automate or speed up` variety of customer support domain issues   
    - Tree based models sped up the resolution of issues by 10% (compared to no models) and DL models added an additional 6% time efficiency. 
- 4. Ride Check
    - If the ride is halted in an unusual place/time for an extended time, it can alert for crash or other safety risk issues
- 5. Estimated Time of Arrivals (ETAs)
    - Uber's Map Services Team Estimate the `base `ETA values for each `segment`      
    - Accurate ETAs are critical to positive user experience and ETAs are critical for pricing and routing
