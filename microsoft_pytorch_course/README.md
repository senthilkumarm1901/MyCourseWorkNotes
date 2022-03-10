## Pytorch Fundamentals

- This is a [4 Course Specialization](https://docs.microsoft.com/en-us/learn/paths/pytorch-fundamentals/?source=learn) comprising: <br>
    - Introduction to PyTorch
    - Introduction to Computer Vision with PyTorch
    - Introduction to NLP with PyTorch
    - Introduction to Audio Classification with PyTorch


### Introduction to PyTorch

#### What is a Tensor?

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


