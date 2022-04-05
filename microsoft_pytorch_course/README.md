## Pytorch Fundamentals

- This is a [4 Course Specialization](https://docs.microsoft.com/en-us/learn/paths/pytorch-fundamentals/?source=learn) comprising: <br>
    - Introduction to PyTorch
    - Introduction to Computer Vision with PyTorch
    - Introduction to NLP with PyTorch
    - Introduction to Audio Classification with PyTorch


### Introduction to PyTorch

<details><summary><code> What is a Tensor?</code></summary>


- Tensors are **like** numerical arrays that encode the input, output and weights/parameters of a model in the form of arrays and matrices.
- Typical 1D and 2D arrays:

![image](https://user-images.githubusercontent.com/24909551/157380975-0402a8ec-7f49-49a3-aef8-5fadc19d4c9c.png)

- How to imagine a 3D array:

![image](https://user-images.githubusercontent.com/24909551/157381034-056897c1-acea-459b-b43a-1b56d55b2434.png)

- Tensors work better on GPUs. They are optimized for **automatic differentiation**
- Tensors and numpy often have the same memory address. For example, review the code below <br>

```python
import numpy as np
import torch

data = [[1,2],[3,4]]
np_array = np.array(data)
tensor_array = torch.from_numpy(np_array)

# doing multiplication opearation on `np_array`
np.multiply(np_array,2,out=np_array)

print(f"Numpy array:{np_array}")
print(f"Tensor array:{tensor_array}")
```

```python
Numpy array:[[2 4]
 [6 8]]
Tensor array:tensor([[2, 4],
        [6, 8]])
```

**How to initialize a tensor?**: <br>

```python    
# directly from a python datastructure element
data = [[1,2],[3,4]]
x_tensor_from_data = torch.tensor(data)

# from numpy_array
np_array = np.array(data)
x_tensor_from_numpy = torch.from_numpy(np_array)

# from other tensors
x_new_tensor = torch.rand_like(x_tensor_from_data, dtype=torch.float) # dtype overrides the dtype of z_tensor_from_data
    
# random or new tensor of given shape
shape = (2,3,) # or just (2,3)
x_new_tensor_2 = torch.ones(shape)
```
    
**What are the `attributes` of a tensor?**:<br>

```python
print(f"{x_new_tensor_2.shape}")
print(f"{x_new_tensor_2.dtype}")
print(f"{x_new_tensor_2.device}") # whether stored in CPU or GPU
```

**When to use CPU and and when to use GPU while `operating` tensors?**: <br>

- Some common tensor operations include: Any arithmetic operation, linear algebra, matrix manipulation (transposing, indexing, slicing)
- Typical GPUs have 1000s of cores. GPUs can handle parallel processing.

![image](https://user-images.githubusercontent.com/24909551/159158293-6faec4f4-e959-4fa6-a5cf-114ddb83810b.png)
    
- Typical CPUs have 4 cores. Modern CPUs can have upto 16 cores. Cores are units that do the actual computation. Each core processes tasks in **sequential** order

![image](https://user-images.githubusercontent.com/24909551/159158302-d75e6fea-eaaa-4c01-a930-0b41a5cfde7c.png)

- Caveat: Copying large tensors across devices can be expensive w.r.t `time` and `memory`

- `PyTorch` uses Nvidia `CUDA` library in the backend to operate on GPU cards

```python
if torch.cuda._is_available():
    gpu_tensor = original_tensor.to('cuda') 
```

**What are the common tensor operations?**: <br>
- `Joining` or `ConCATenate`
```python
new_tensor = torch.cat([tensor, tensor],dim=1) # join along column if dim=1
```    
- `Matrix Multiplication`   
```python
# you would have to do the transpose
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
assert y1 = y2 = y3
```

- `Element-wise Multiplication`    
```python
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
```  

- `Single element tensor` into python numerical value    
```python
sum_of_values = tensor.sum()
sum_of_values_python_variable = sum_of_values.item()
print(sum_of_values.dtype, type(sum_of_values_python_variable))
# >> torch.int64, <class 'int'>
```

- `In-place Operations`    
```python
# add in_place
tensor.add_(5)

# transpose  in place
tensor.t_()
```   
</details>



<details><summary><code> Dataset and DataLoader </code></summary>
   

Two data `primitives` to handle data efficiently: <br>
- `torch.utils.data.Dataset`
- `torch.utils.data.DataLoader` 

What does `Dataset` do?
- `Dataset`: Stores data samples and their corresponding labels
- `DataLoader`: Wraps an iterable around Dataset to enable easy access to the samples. `DataLoader` can also be used along with `torch.multiprocessing`
- `torchvision.datasets` and `torchtext.datasets` are both subclasses of `torch.utils.data.Dataset` (they have __getitem__ and __len__ methods implemented) and also they can be passed to a `torch.utils.data.DataLoader`

**Arguments of a pre-loaded dataset like `FashionMNIST`**:<br>

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

import matplotlib.pyplot as plt

# turn the integer y values into a `one_hot_encoded` vector 
# 1. create a zero tensor of size 10 torch.zeros(10, dtype=torch.float)
# 2. `scatter_` assigns a value =1
the_target_lambda_function = Lambda(lambda y: torch.zeros(10,
                                    dtype=dtype=troch.float).scatter_(dim=0,
                                                    index=torch.tensor(y), value=1))

# ToTensor() --> normalizes the features before feeding to model

training_data = datasets.FashionMNIST(
    root="data", # the path where the train/test data is stored
    train=True, # False if it is a test dataset 
    download=True, # downloads the data from Web if not available at root
    transform=ToTensor(), # transform the features; converts PIL image or numpy array into a FloatTensor and scaled the image's pixel intensity to the range [0,1]
    target_transform=the_target_lambda_function
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=torch.nn.functional.one_hot(y, num_classes=10) # alternate way
)
```    

**How should the data be preprocessed before training in DL?**: <br>
- Pass samples of data in `minibatches`
- reshuffle the data at every epoch to overfitting
- leverage Python's `multiprocessing` to speed up data retrieval
- `torch.utils.data.DataLoader` abstracts all the above steps

```python
train_dataloader = DataLoader(training_data, 
                              batch_size=64, 
                              shuffle=True)

test_dataloader = DataLoader(test_data, 
                             batch_size=64,
                             shuffle=True)
```

**How to iterate through DataLoader?**: <br>

```python
train_features, train_labels = next(iter(train_dataloader))
feature_data = img  = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
```   
**What does normalization do?**: <br>
- Changes the range of the data
- When one pixel value is 15 and another pixel is 190, the higher pixel value will deviate the learning 

**Why do we do normalization of data before training a DL**:
- Prediction accuracy is better for normalized data
- Model can learn faster if data is normalized

</details>

    
<details><summary>Building <code> Model Layers </code> in PyTorch</summary>
    
**Components of a Neural Network**:

- Typical Neural Network: <br>

![image](https://user-images.githubusercontent.com/24909551/160055546-f6150c41-acb0-44a4-942e-0d20c86e8972.png)

- Activation Function, Weight and Bias

![image](https://user-images.githubusercontent.com/24909551/160055714-0bfb081d-6c1b-4733-a226-d7db71e74fec.png)

- Linear weighted sum of inputs: x = &sum;(`weights` * `inputs`) + `bias`    
- f(x) = activation_func(x)

- Activation Functions add non-linearity to the model    
- Different Activation Functions: <br>
    - **Sigmoid**: <sup>1</sup>/<sub>(1 + exp(-x))</sub>
    - **Softmax**: <sup>exp(x)</sup> / <sub>(sum(exp(x)))</sub>
    - **ReLU**: max(0,x)
    - **Tanh**: <sup>(exp(x) - exp(-x))</sup>/<sub>(exp(x) + exp(-x))</sub>

**Building a neural network in PyTorch** 
- `torch.nn` class provides all the building block needed to build a NN
- Every module/layer in PyTorch subclases the `torch.nn.Module`
- A NN is a composite module consisting of other modules (layers)
    
```python 
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
```
    
- Initialize all layers in `__init__` module
- Build a 3-layer NN with 
    - flattened `28*28` image as input,
    - 2 hidden layers will have 512 neurons each and
    - the third layer will have 10 neurons each corresponding to the number of classes
    
```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
 
# create a instance of the class NeuralNetwork and move it to the device (CPU or GPU)
model = NeuralNetwork().to(device)
    
```

- How a forward pass would be like: 
    - Why `model(X)` instead of `model.forward(X)`? [Source](https://stackoverflow.com/questions/55338756/why-there-are-different-output-between-model-forwardinput-and-modelinput) 
    
```python
x = torch.rand(1, 28, 28, device=device)
logits = model(X) # runs the __init__ method
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```

```python
print("Weights stored in first layer: {model.linear_relu_stack[0].weight} \n")
print("Bias stored in first layer: {model.linear_relu_stack[0].bias} \n") 
```

- **Step 1**:Convert `28*28` into a contiguous array of 784 pixel values
    
```python
input_image = torch.rand(3, 28, 28)
print(input_image.size())
# step 1: Flatten the input image
flatten = nn.Flatten() # instantitate
flat_image = flatten(input_image)  # pass the prev layer (input) into the instance
print(flat_image.size())
```
- **Step 2**: Dense or linear layer in PyTorch `weight * input + bias`
    
```python    
# step 2: apply linear transformation `weight * input + bias`
layer1 = nn.Linear(in_features=28*28, out_features=20) # instantiate
hidden1 = layer1(flat_image) # pass the prev layer (flattened image) into the instance
print(hidden1.size())
```

- **Step 3**: Apply Relu activation on the linear transformation
    
```python
relu_activation = nn.ReLU() #instantiate
hidden1 = relu_activation(hidden1)
```    
- **Step 4**: Compute the logits
    

    
- **Step 5**: Apply `Softmax` function
    
- Full NN workflow: 
 
![image](https://user-images.githubusercontent.com/24909551/161696907-8672f820-3293-4390-b153-bf702731352d.png)

 
</details>
