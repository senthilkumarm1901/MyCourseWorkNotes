## Note This md file contains learning notes from Kaggle Learn DL courses

### Single Neuron
- A liniear unit with 3 inputs
> y = w<sub>0</sub>x<sub>0</sub> + w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub> + b
- In Keras, the input_shape is a list 
> `model = keras.Sequential([layers.Dense(units=1, input_shape=[3]])`
> where `unit` represents the number of neurons in the `Dense` layer
> `input_shape` determines the size of input
- where for 
    - tabular data:
    > `input_shape = [num_columns]`
    - image_data:
    > `input_shape = [height, width, channels]`
