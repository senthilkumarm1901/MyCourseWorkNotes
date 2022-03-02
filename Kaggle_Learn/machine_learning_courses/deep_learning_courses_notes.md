## Note This md file contains learning notes from Kaggle Learn DL courses

### Single Neuron
- A linear unit with 1 input

![image](https://user-images.githubusercontent.com/24909551/156300415-b206cfab-f925-4606-859f-e75208695ee1.png)

- A liniear unit with 3 inputs
> y = w<sub>0</sub>x<sub>0</sub> + w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub> + b

![image](https://user-images.githubusercontent.com/24909551/156300573-b83fb302-a2be-42c4-a90c-160387c46bb3.png)


- In Keras, the input_shape is a list 
> `model = keras.Sequential([layers.Dense(units=1, input_shape=[3]])`
> where `unit` represents the number of neurons in the `Dense` layer
> `input_shape` determines the size of input
- where for 
    - tabular data:
    > `input_shape = [num_columns]`
    - image_data:
    > `input_shape = [height, width, channels]`

### Deep Neural Network
- A dense layer consists of multiple neurons

![image](https://user-images.githubusercontent.com/24909551/156300748-7fab5237-4b16-4195-9b38-8963b61abecf.png)

- Empirical fact: Two dense layers with no activation function is not better than one dense layer
- Why Activation functions? <br>

![image](https://user-images.githubusercontent.com/24909551/156301224-bc2e01b1-95d3-4194-b35b-4c1dcda01c77.png)

![image](https://user-images.githubusercontent.com/24909551/156304054-df308632-3c04-4497-af93-55bddfb33ebd.png)

- Rectifier function "rectifies" the negative values to zero. ReLu puts a "bend" in the data and it is better than simple linear regression lines
- A single neuron with ReLu

![image](https://user-images.githubusercontent.com/24909551/156304468-fc4f589f-3b1a-4722-93d1-414a1d67d605.png)

- A Stack of Dense Layers with ReLu for non-linearity. An example of a Fully-Connected NN:

![image](https://user-images.githubusercontent.com/24909551/156304549-5e184743-523b-428a-8f2b-fc454d5789fa.png)

- the final layer is linear for a regression problem; can have softmax for a classification problem 

```python
from tensorflow import keras
from tensorflow.keras import layers

# defining a model
model = keras.Sequential([
    # the hidden ReLu layers
    layers.Dense(units=4, activation='relu', input_shape=[2]),
    layers.Dense(units=3, activation='relu'),
    layers.Dense(unit=1),
    ])
```

### Stochastic Gradient Descent
- Stochastic Gradient Descent is an optimization algorithm that tells the NN 
     - how to change its weight so that
     - the loss curve shows a descending trend

![image](https://user-images.githubusercontent.com/24909551/156314264-5de54383-e106-4d76-b700-f98a9870dd81.png)

Definition of terms: 

- `Gradient`: Tells us in what direction the NN needs to adjust its weights. It is computed as a partial derivative of a multivariable `cost func`
- `cost_func`: Simplest one: Mean_absolute_error: mean(abs(y_true-y_pred))
- `Gradient Descent`: You descend the loss curve to a minimum by reducing the weights `w = w - learning_rate * gradient`
- `stochastic` - occuring by random chance. The selection of samples in each mini_batch is by random chance
 
How SGD works:

- 1. Sample some training data (called `minibatch`) and predict the output by doing forward propagation on the NN architecture
- 2. Compute loss between predicted_values and target for those samples
- 3. Adjust weights so that the above loss is minimized in the next iteration
- Repeat steps 1, 2, and 3 for an entire round of data, then one `epoch` of training is over
- For every minibatch there is only a small shift in the weights. The size of the shifting of weights is determined by `learning_rate` parameter


```python
# define the optimizer
model.compile(optimizer="adam", loss="mae")
```

```python
# fitting the model
history = model.fit(X_train, y_train, 
    validation_data=(X_valid,y_valid),
    batch_size=256,
    epoch=10,
    )

# plotting the loss curve
history_df = pd.DataFrame(history.history)
history_df['loss'].plot()
```

### Underfitting and Overfitting

**Underfitting**
- Capacity Increase
    - If you increase the number of neurons in each layer (making it wider), it will learn the "linear" relationships in the features better
    - If you add more layers to the network (making it deeper), it will learn the "non-linear" relationships in the features better
    - Decision on `Wider` or `Deeper` networks depends on the dataset 

**Overfitting**
- Early Stopping: Interrupt the training process when the validation loss stops decreasing (stagnant)
- Early stopping ensures the model is not learning the noises and generalizes well

![image](https://user-images.githubusercontent.com/24909551/156336481-6c2ceb9b-97dc-494f-8e0d-07cc389664f3.png)

- Once we detect that the validation loss is starting to rise again, we can reset the weights back to where the minimum occured.

```python
from tensorflow.keras.callbacks import EarlyStopping
# a callback is just a function you want run every so often while the network trains

# defining the early_stopping class
early_stopping = EarlyStopping(min_delta = 0.001, # minimum about of change to qualify as improvement
                               restore_best_weights=True,
                               patience=20, # number of epochs to wait before stopping
                              )


history = model.fit(X_train, y_train, 
    validation_data=(X_valid,y_valid),
    batch_size=256,
    epoch=500,
    callbacks=[early_stopping],
    verbose=0 #turn off logging
    )
    
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot();
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))
```

![image](https://user-images.githubusercontent.com/24909551/156341007-74fa6d34-652d-49b4-a238-5d2a802b08bb.png)

### Batch Normalization

**Why `BatchNorm`?**
- Can prevent unstable training behaviour
    - the changes in weights are proportion to how large the activations of neurons produce
    - If some unscaled feature causes so much fluctuation in weights after gradient descend, it can cause unstable training behaviour
- Can cut short the path to reaching the minima in the loss curve (hasten training)
    - models with `BatchNorm` tend to need fewer epochs for training 


**What is `BatchNorm`?**
- On every batch of data subjected to training 
    - normalize the batch data with the batch's mean and standard deviation
    - multiply them with rescaling parameters that are learnt while training the model
 
**Three places where `BatchNorm` can be used
1. After a layer

```python
keras.Sequential([
    layers.Dense(16,activation='relu'),
    layers.BatchNormalization(),
    ])
```

2. in-between the linear dense and activation function

```python
keras.Sequential([
    layers.Dense(16),
    layers.BatchNormalization(),
    layers.Activation('relu')
    ])
```
3. As the first layer of a network (role would then be similar to similar to Sci-Kit Learn's preprocessor modules like `StandardScaler`)

```python
keras.Sequential([
    layers.BatchNormalization(),
    layers.Dense(16),
    layers.Activation('relu')
    ])
```

Source: <br>
- Kaggle.com/learn
