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
layer1 = nn.Linear(in_features=28*28, out_features=512) # instantiate
hidden1 = layer1(flat_image) # pass the prev layer (flattened image) into the instance
print(hidden1.size())
```

- **Step 3**: Apply Relu activation on the linear transformation
    
```python
relu_activation = nn.ReLU() #instantiate
hidden1 = relu_activation(hidden1)
```    
Repeat Step 2 and 3 for `hidden2`: <br>

```python
layer2 = nn.Linear(in_features=512, out_features=512)
hidden2 = layer2(hidden1)
hidden2 = relu_activation(hidden2)
```    
    
- **Step 4**: Compute the logits
    
```python
# a simple 1 hidden layer NN with 20 neurons in the hidden layer
nn_seq_modules = nn.Sequential(
                    flatten,
                    layer1,
                    relu_activation,
                    layer2,
                    relu_activation,
                    nn.Linear(512, 10), # the output                )
input_image = torch.rand(3, 28, 28)
logits =  nn_seq_modules(input_image)   
```
    
- **Step 5**: Apply `Softmax` function
    
```python

softmax = nn.Softmax(dim=1)
predict_probab = softmax(logits)

```
    
- Full NN workflow: 
 
![image](https://user-images.githubusercontent.com/24909551/161696907-8672f820-3293-4390-b153-bf702731352d.png)


**How to see internal layers of a NN in PyTorch**:

```python
print("Weights stored in first layer: {model.linear_relu_stack[0].weight} \n")
print("Bias stored in first layer: {model.linear_relu_stack[0].bias} \n") 
    
from name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()}"
```
    
```bash
Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784])
Layer: linear_relu_stack.0.bias | Size: torch.Size([512])
Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512])
Layer: linear_relu_stack.2.bias | Size: torch.Size([512])
Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512])
Layer: linear_relu_stack.4.bias | Size: torch.Size([10])
```  
</details>
 
 <details> <summary> Automatic Differentiation <code>torch.autograd</code> </summary>
     
- `torch.autograd` is the engine that automatically computes gradients during model optimization     

- `Back Propagation`: An algorithm to adjust the `weights` in a neural network according to the `gradient` of the `loss function`. E.g. algorithm: `Stochastic Gradient Descent`    
- `Gradient`: Partial derivative of a multivariable loss/cost function
- `Loss function`: It is the difference between the expected output and actual output
- `Gradient Descent`: Adjust the weights according to the gradient such that loss curve keeps reducing (i.e. reduce loss to 0) `w = w - learning_rate * gradient_wrt_w`
- `Stochastic`: Occurring by random chance; Selection of each samples in mini_batch occurs by random chance
     
 ```python
 
 x = torch.ones(5) # input tensor
 y = torch.zeros(3) # expected output
 # requires_grad argument is set to True to `w` and `b`
 w = torch.randn(5, 3, requires_grad=True) 
 b = torch.randn(3, requires_grad=True)
 z = torch.matmul(x,w) + b
 loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)
 ```
     
 - The above code constructs the below **computational graph**
 
  ![image](https://user-images.githubusercontent.com/24909551/161719441-3569dd3f-a9e2-4af5-9835-b96da11f0dfa.png)
     
 - Only variables `w` and `b` are passed the `requires_grad` argument
 - `<variable>.grad_fn` stores the reference to the backward propagation function

- By default, we can perform gradient calculations `backward` only once (for performance reasons)
- If we need to do several `backward` calls on the same graph, we need to pass `retain_graph=True`

```python
 loss.backward()
 print(b.grad)
 print(w.grad) 
```
     
     
**Different ways of disabling gradient tracking**:
- Disabling gradient tracking is needed when doing inference (where only forward pass is needed)
- Disabling some parameters in your neural network as **forzen parameters**. This is a common scenario for fine-tuning a pre-trained network.
- 1. `with torch.no_grad():` context manager

```python
with torch.no_grad():
     z = torch.matmul(x,w) + b
# now z will have z.requires_grad == True
```     

- 2. disable gradient tracking using `detach()` method

```python
z = torch.matmul(x,w) + b
z_det = z.detach()   
# now z_det will have z_det.requires_grad == True
```    

**About Directed Acyclic Graph based backpropagation**:
- `autograd` keeps a record of all data and the executed operations in a directed acyclic graph (DAG) consisting of `torch.autograd.Function` objects
- While doing `forward pass` on a tensor with `requires_grad=True` argument, 
    - the forward pass operation is computed to obtain a resulting tensor
    - the backward `gradient` function is maintained (sort of like `instantiated`) in the DAG (aka computational graph)
     
- When the `.backward()` is called, the `autograd` then:
    - computes the gradients from each `.grad_fn`
    - accumulates the resulting gradient values in the respective tensor's `.grad` attribute
    - computes back propagation from root (output tensors) till leaves (the input tensors with `requires_grad` = True)
- *DAGs are dynamic in PyTorch*: The graph is recreated from scratch after each `.backward()` call
     
</details>

<details> <summary> Model Parameters Optimization </summary>    

- How do you optimize the model parameters? Using `optimizers` (e.g.: SGD, `adam`, etc.,) that take in arguments such as `type_of_optimizer`, `model.parameters()` and `learning_rate`
    
- A revisit of the codes from prev modules    
```python    
%matplotlib inline
import torch
from torch import nn
from torch.utils.data import DataLoader #for iterating through the dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
                root="data",
                train=True,
                download=True,
                transform=ToTensor()
                )

test_data = datasets.FashionMNIST(
                root="data",
                train=False,
                download=True,
                transform=ToTensor()
                )
                
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
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

# instantiate the class
model = NeuralNetwork()    
```    
    
**Hyper parameters**:
    
- `num_of_epochs`: The number of times the entire training dataset is pass through the network
- `batch_size`: The number of data samples seen by the model before updating its weights. (derived parameter `steps = total_training_data/batch_size` - the number of batches needed to complete an epoch)
- `learning_rate`: How much to change the weights in the `w = w - learning_rate * gradient`. Smaller value means the model will take a longer time to find best weights. Larger value of learning_rate might make the NN miss the optimal weights because we might step over the best values

**Common Loss Functions**:    
- nn.MSELoss # Mean Squared Error
- nn.NLLLoss #Negative Log Likelihood    
- nn.CrossEntropyLoss # = combine(`nn.LogSoftmax` and `nn.NLLLoss`)   
 
```python
# initialize the loss function
loss_fn = nn.CrossEntropyLoss()
```    

```python
# initialize the optimizer
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
# key optimizer steps
# by default, gradients add up in PyTorch
# we zero out in every iteration
optimizer.zero_grad() 
# performs the gradient computation steps (across the DAG)
loss.backward()
# adjust the weights
optimizer.step()
```    

- `loss_fn` and `optimizer` are passed to `train_loop` and just `loss_fn` to `test_loop`    
    
```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

epochs = 10
for i in range(epochs):
    print(f"Epoch {i+1}\n ----------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader,model, loss_fn)
print("Over!")    
```   
```python
def train_loop(traindataloader, model, loss_fn, optimizer):
    for X, y in traindataloader:
        # forward pass
        pred = model(X)
        
        # compute loss
        loss = loss_fn(pred, y)
        
        # backpropagation
        optimizer.zero_grad()
        optimizer.backward()
        optimizer.step()
        
        print(f"Loss: {loss.item()}")
 
 
 
def test_loop(testdataloader, model, loss_fn):
    size = len(testdataloader.dataset)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X,y in testdataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # find out where argmax index is same as that actual
            # then convert into float
            # sum across all of the records in the batch size
            # .item() - converts pytorch tensor into Python value
            correct += (pred.argmax(dim=1)==y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    
    print(f"Test Accuracy {correct*100}; Test Data Average Loss = {test_loss}")

```   
</details>    

<details> <summary> How to save, load and export PyTorch models</summary>   
    
**How to save and load the model for inference?**    
```python    
# pytorch models save the parameters in a internal state dictionary called `state_dict`
torch.save(model.state_dict(),"data/modelname.pth")
    
# infer from a saved model
# instantiate the model architecture class
model = NeuralNetwork()
model.load_state_dict(torch.load("data/modelname.pth"))
# the eval method is called before inferencing so that the batch normalization dropout layers are set to `evaluation` mod
# Failing to do this can yield inconsistent inference results
model.eval()
```    

**How to export a pytorch model to run in any Programming Language/Platform**: <br>
    
- **ONNX**: Open Neural Network Exchange 
- Converting `PyTorch` model to `onnx` format aids in running the model in Java, Javascript, C# and ML.NET
    
```python
# while explorting pytorch model to onnx, 
# we'd have to pass a sample input of the right shape
# this will help produce a `persisted` ONNX model    
import torch.onnx as onnx
input_image = torch.zeros((1,28,28))
onnx_model_location = 'data/model.onnx'
onnx.export(model, input_image, onnx_model)
```    
    
</details>
     
     
     
          
Source: docs.microsoft.com/en-US/learn    
