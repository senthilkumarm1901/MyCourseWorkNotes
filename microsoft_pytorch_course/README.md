## Pytorch Fundamentals

- This is a [4 Course Specialization](https://docs.microsoft.com/en-us/learn/paths/pytorch-fundamentals/?source=learn) comprising: <br>
    - Introduction to PyTorch
    - Introduction to Computer Vision with PyTorch
    - Introduction to NLP with PyTorch
    - Introduction to Audio Classification with PyTorch


### Introduction to PyTorch

<details><summary><code> What is a Tensor?</code></summary>


- Tensors are **like** numerical arrays that encode the input, output and weights/parameters of a model
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
   
 
</details>
