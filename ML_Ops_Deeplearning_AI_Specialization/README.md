This directory will hold my notes from MLOps specialization

## Course 1 - Intro to ML in Production

### ML Lifecycle and Deployment 

**Table of Contents**: <br> 
- Key components of ML Project Lifecycle
- `Data Drift` or `Concept Drift`
    - Key metrics to monitor Concept Drift
- ML Modeling Iterative Cycle vs ML Product Deployment Cycle
- 3 Different Types of Deployment (w.r.t degree of automation)
    - Shadow Deployment
    - Canary Deployment
    - Blue-Green Deployment

#### Key Components of ML Project Lifecycle
- An example ML Deployment 
> `New unseen photo` --> `ML_model inference` in a `Prediction server` (cloud) <--> `Edge Device` to do some decisioning

- ML Infrastructure Components [The Hidden Technical Debt in ML Systems](https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)
    - Configuration (defining the requirements?)
    - Data Collection, Data Verification
    - Feature Extraction
    - **ML Code**
    - Analysis Tools
    - Process Management Tools
    - Machine Resource Management  

![image](https://user-images.githubusercontent.com/24909551/157711630-6ba1b021-1957-40e9-b06b-33349499fbcc.png)


> (1) Scoping --> (2) Data --> (3) Modeling --> (4) Deployment
- The ML Project Lifecycle   
   - 1. Scoping: 
         - Define the Project Requirements
   - 2. Data:
         - Define Data
         - Establish baseline (e.g.: how much should the o/p accuracy)
         - Label Data
         - Feature Engineer Data
   - 3. Modeling:
         - Select the right model and hyperparameters to train the model
         - Perform error analysis
   - 4. Deployment:
         - Deploy in an API/UI in production
         - Monitor Model predictions
         - Maintain Infrastructure requirements

> If there is "concept drift" during deployment, get back to reflecting the change in the data
> Retrain model and deploy the newly trained model in production

**E.g.: Speech Recognition**: <br>
- 1. Scoping
    - `Why` Speech Recognition: For Voice Search 
    - `Metrics`: How are we measuring the accuracy - **word error rate (accuracy)**,**latency & throughput**
    - `Resources`(human, data, machines) & `Timeline`
- 2. Data:
    - `Data Reliability`: Consistency of Data labeling
    - `Data Normalization`: Here, how to perform volume normalization    
- 3. Modeling:
    - Selection of Model/Algorithm
    - Selection of Coding framework 
- 4. Deployment:
    - Mode of deployment: API (or) testing in a UI

![image](https://user-images.githubusercontent.com/24909551/157716379-a75a72d6-8f58-47ba-b661-a412163b0574.png)


- **A typical ML System** = **Code** + **Data**

**Practice of Research/cademica**: <br>
- Fix Data constant, 
    - keep changing code (algorithm/model) and hyperparameters 

**Practice of Product Teams**: <br>
- Fix Code(algoithm) (download the code to use from opensource github) and 
    - keep optimizing (improving or changing) data 
    - keep experimenting on different hyperparameters
(both Academia and Product teams tune hyperparameters)

![image](https://user-images.githubusercontent.com/24909551/157716001-b427dddc-8c07-4925-badd-9f9bca284085.png)


Concept Drift vs Data Drift: 
- Because of Inflation or other market reasons, the `same size house` (data constant) starts getting quoted a higher price (concept change)
- What if ppl start building smaller houses (a trend), then input distribution of the size of the houses changes over time


Software Engineering decisioning criteria:
- Streaming vs Batch Processing need
- Cloud vs Edge 
- CPU vs GPU
- Latency, Throughput (queries per second) 
![image](https://user-images.githubusercontent.com/24909551/157718684-06ea57ad-9e1b-4b8d-9787-891b47c35dfd.png)
- Logging
- Security and Privacy

**Types of Deployment**: <br>
1. Shadow Deployment: 
    - While the older version is still running the prediction, 
          - run the same traffic on the newer model verion in shadow (not known to the outisde world)
          - The newer version is rolled out, post successful traffic and load testing.  
2. Canary Deployment: 
> “Canary releases” get their name from an old coal mining tactic. Miners would release canaries into coal mines in an attempt to gauge the amount of toxic gases present. If the canary survived, well, things were safe
> [source](https://launchdarkly.com/blog/what-is-a-canary-release/#:~:text=%E2%80%9CCanary%20releases%E2%80%9D%20get%20their%20name,%2C%20well%2C%20things%20were%20safe.)
    - Roll out to a small fraction of the traffic
    - Monitor systems and ramp up traffic gradually

3. Blue-Green Deployment: 
    - Old version (blue)
    - New version (green)
> Have an older version as backup until the newer (Greener) version performs convincingly 

[Advantages and Disadvantages of above deployments](https://www.opsmx.com/blog/advanced-deployment-strategies-devops-methodology/)

Degrees of Automation: <br>
> Human-only --> Human used + AI shadowing --> AI Assistance --> Partial Automation --> Full Automation

Source: <br>
Additional Good Links: <br>
- https://www.datascience-pm.com/10-ways-to-manage-a-data-science-project/
