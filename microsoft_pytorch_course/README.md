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
    
</details>
