## **Notes from Deeplearning.ai 1st Course on NLP - `Classification and Vector Spaces`**

To Learn from the course: 
- Sentiment Analysis Classification using Logistic Regression and then Naive Bayes
- Understanding Word Vectors; Use PCA to reduce dimension and then visualize the relationship
- Translation using Word Vectors, locality sensitive hashing and Approximate Nearest Neighbours

### Logistic Regression basics: <br>
**Training**: 
- Step 1: Forward Propagation function == the core ML equation
    - `linear_model = np.dot(X,weights) + bias` # X is the input and (weights = w and bias = b) are the parameters that the model will learn
    - `y_predicted = sigmoid(linear_model)` # logistic regression equation (ML equation)
    - theta = fn(w,b) ==> parameters that the logistic regression model learns
- Step 2: Compute Cost Function = Cost(y_actual, y_predicted)
    - `cost_fn = log_loss = cross_entropy_loss` 
    - = -  1/m &sum; (i = 1 to N) {y(i) * log (h(x(i),theta) + (1 - y(i)) * log (1 - (h(x(i),theta) }
- Step 3: Do backward propagation
    - dtheta = dw and db = gradients = partial derivative of above cost fn w.r.t w and b respectively
    - `w = w - lr * dw` ; `b = b - lr * db` # update weights of the parameters
- Repeat Step 1 and 2 until n_iterations is over
<br>

**Inference**:
- Perform Step 1 of **Training** and compute an array of probability scores y_predicted
- `y_predicted_cls = [1 if predicted_probabilty > 0.5 else 0 for predicted_probability in y_predicted]`

### Naive Bayes basics: <br>

Navie Bayes is based on the Baye's Rule:
    - P(Y=y | X) = P(X | y) * <sup>P(y)</sup> / <sub>P(X)</sub>  # where y is one of the classes in Y and X represents all the features
    - Posterior_probability = Likelihood *  <sup>Prior Probability</sup> / <sub>Evidence</sub> 

**Case study Example: `Classifying text search queries`** <br>
- `Y = ['Zoology', 'ComputerScience', 'Entertainment']`
- Text Input Query: `x(i) = 'Python'`
    - `P(Y ='Zoology' | x_i)` > `P(Y='ComputerScience' | x_i)` > `P(Y='Entertainment' | x_i)` # as in 'snake Python' or 'python program' or 'Monty Python' respectively
- Suppose Text Input Query is `x(j) = 'Python download') 
    -  `P(Y='ComputerScience' | x_j)` > `P(Y='Entertainment' | x_j)` > `P(Y='Zoology' | x_j)` 
- `argmax P(Y | X) ~= argmax P(Y) * P(X | y)`
- For multiple features, argmax P(Y | x) ~= argmax P(y) * <sub>i=1</sub>&prod;<sup>n</sup> P(X | y);
    -  argmax P(Y=y | X='python download') ~= argmax P(y) * P('python' | y) * P('download' | y)

**Naive Bayes Training**: <br>
- Step 1A - Compute the learning parameters: 
    .- Learning Parameter 1 - compute `Prior Probability` for every class: 
           - Count the freq of each class and then compute P(y) for every y in Y
     - Learning Parameter 2 - compute `Likelihood` for every feature given every class: 
           - Count the frequency of every word (aka feature) given it belongs to a class Count(w_i | y) for every y
- Step 1B: Perform Laplacian smoothing (to avoid scenarios like what if P(x_i | y) == 0; P(x_i | y ) = <sup>count(w_i, y) + 1</sup> / <sub> count(y in X) + n_x </sub> where n_x is the number of "naively independent" features

**Naive Bayes Inferencing**: <br>
- Step 1: For unseen data `X_test_i`, "fetch" likelihood probabilities for every feature 
- Step 2: Compute log(prior) + &sum; of log(likelihood) for every `naively independent` feature for every `y` in Y
- Step 3: Assign that class `y` which has the highest `log prior + log likelihood` values 

**Key Questions**:
- Number of parameters in NB:
     - Number of classes = n_y
     - Number of features = n_x
     - Number of parameters = n_y - 1 + n_x * n_y  

Source: 
- Deeplearning.ai NLP Specialization
