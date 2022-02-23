## **Notes from Deeplearning.ai 1st Course on NLP - `Classification and Vector Spaces`**

To Learn from the course: 
- Sentiment Analysis Classification using Logistic Regression and then Naive Bayes
- Understanding Word Vectors; Use PCA to reduce dimension and then visualize the relationship
- Translation using Word Vectors, locality sensitive hashing and Approximate Nearest Neighbours

### Logistic Regression: <br>
**Training**: 
- Step 1: Forward Propagation function == the core ML equation
    - `linear_model = np.dot(X,weights) + bias` # X is the input and (weights = w and bias = b) are the parameters that the model will learn
    - `y_predicted = sigmoid(linear_model)` # logistic regression equation (ML equation)
    - theta = fn(w,b) ==> parameters that the logistic regression model learns
- Step 2: Compute Cost Function = Cost(y_actual, y_predicted)
    - `cost_fn = log_loss = cross_entropy_loss = -  1/m SUM of every training data point from i = 1 to N {y(i) * log (h(x(i),theta) + (1 - y(i)) * log (1 - (h(x(i),theta) }`
- Step 3: Do backward propagation
    - dtheta = dw and db = gradients = partial derivative of above cost fn w.r.t w and b respectively
    - `w = w - lr * dw` ; `b = b - lr * db` # update weights of the parameters
- Repeat Step 1 and 2 until n_iterations is over
<br>

**Inference**:
- Perform Step 1 of **Training** and compute an array of probability scores y_predicted
- `y_predicted_cls = [1 if predicted_probabilty > 0.5 else 0 for predicted_probability in y_predicted]`


Source: 
- Deeplearning.ai NLP Specialization
