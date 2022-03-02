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

model = keras.Sequential([
    # the hidden ReLu layers
    layers.Dense(units=4, activation='relu', input_shape=[2]),
    layers.Dense(units=3, activation='relu'),
    layers.Dense(unit=1),
    ])
```

### Stochastic Gradient Descent


Source: <br>
- Kaggle.com/learn
