import torch
import numpy as np

'''
This file is to create tensor using different method.
'''

# #######################  TYPE 1  #######################
# Create directly using torch.tensor or torch.from_numpy

'''
TYPE 1: Create tensor directly

1. torch.tensor

    torch.tensor(
                data,
                dtype=None,
                device=None,
                requires_grad=False,
                pin_memory=False)
        
    Function: create tensor from data
    
    Parameters:
    
    data: could be list or numpy
    dtype: the type of data, same as the type of input data by default
    device: cuda or cpu
    requires_grad: whether require grad
    pin_memory: whether store in pin memory

2. torch.from_numpy

    Function: create tensor from data
    
    Note: the tensor created by touch.from_numpy share the memory from original ndarray.
    If we change one, the other one will also change.
    
'''

# #######################  example 1  #######################

# Create directly using torch.tensor

# flag = True
flag = False

if flag:
    arr = np.ones((3,3))
    print("the data type of ndarray: ", arr.dtype)

    # t = torch.tensor(arr, device='cuda')

    t = torch.tensor(arr)

    print(t)


# ###################  example 2  #####################

# Create from numpy using torch.from_numpy(ndarray)

# Note: the tensor created by touch.from_numpy share the memory from original ndarray.
# If we change one, the other one will also change.

# flag = True
flag = False

if flag:
    arr = np.array([[1,2,3], [4,5,6]])
    t = torch.from_numpy(arr)

    print("numpy array: ", arr)
    print("tensor: ", t)

    print("\nchange arr\n")
    arr[0, 0] = 0
    print("numpy array: ", arr)
    print("tensor: ", t)

    print("\nchange tensor\n")
    t[0, 0] = -1
    print("numpy array: ", arr)
    print("tensor: ", t)


# #######################  TYPE 2  #######################
# create from values

'''
TYPE 2: Create from values

1. torch.zeros()

    torch.zeros(*size,
                out=None,
                dtype=None,
                layout=torch.strided,
                device=None,
                requires_grad=False)
    
    Function: create a tensor of zeros based on the size which specifying the shape
    
    Parameters:
    
    size: the shape of the tensor, such as (3,3), (3,224,224)
    out: the output of the tensor
    layout: the layour in memory, such as strided, sparse_coo, etc.
    device: cuda or cpu
    requires_grad: whether require grad
    

2. torch.zeros_like()

    torch.zeros_like(input,
                     dtype=None,
                     layout=None,
                     device=None,
                     requires_grad=False)
                     
    Funcion: create a tensor of zeros based on the shape of input

    Parameter: 
    
    input: create a tensor of zeros with the same shape of input
    dtype: the type of the data
    layout: the layour in memory, such as strided, sparse_coo, etc.

3. torch.ones()
4. torch.ones_like()

5. torch.full()
6. torch.full_like()
    
    torch.full(size,
               fill_value,
               out=None,
               dtype=None,
               layout= torch.strided,
               device=None,
               requires_grad=False)
               
    Parameters:
    
    fill_value: the table of the tensor

7. torch.arange()

    torch.arange(start=0,
                 end,
                 step=1,
                 out=None,
                 dtype=None,
                 layout=torch.strided,
                 device=None,
                 requires_grad=False)

8. torch.linspace()

    torch.linspace(start,
                   end,
                   steps=100,
                   out=None,
                   dtype=None,
                   layout=sparse.strided,
                   device=None,
                   requires_grad=Flase)
                   
    Function: create a averaged ones tensor
    
    Parameters:
    
    step: the length of tensor, not the step length
    
    Note: the range of values is [start, end]
    
9. torch.logspace()

    torch.logspace(start,
                   end,
                   steps=100,
                   base=10,
                   out=None,
                   dtype=None,
                   layout=sparse.strided,
                   device=None,
                   requires_grad=Flase)
                   
    base: the base of log function, default by 10
    
10. torch.eye()

    torch.eye(n,
              m=None,
              dtype=None,
              layout=torch.strided,
              device=None,
              requires_grad=False)
              
    Function: create a diagonal metrix (2-dimention tensor). Default by square matrix.
    
    
'''

# #######################  example 3  #######################

# Create a tensor using torch.zeros

# flag = True
flag = False

if flag:
    out_t = torch.tensor([1])

    t = torch.zeros((3,3), out=out_t)

    print(t, '\n', out_t)

    # id(), shows the memory addresses of variables
    print(id(t), id(out_t), id(t) == id(out_t))


# #######################  example 4  #######################

# Create a tensor using torch.full

# flag = True
flag = False

if flag:
    out_t = torch.tensor([1])

    # have to specify the parameter of out
    t = torch.full((3,3), 1, out=out_t)
    print("tensor: ", t)


# #######################  example 5  #######################

# Create a tensor using torch.arange

# fflag = True
lag = False

if flag:
    out_t = torch.tensor([1])

    t = torch.arange(2, 10, 2)
    print("tensor: ", t)


# #######################  example 6  #######################

# Create a tensor using torch.linspace

# flag = True
flag = False

if flag:
    out_t = torch.tensor([1])

    t = torch.linspace(2, 10, 5)
    print("tensor: ", t)
    t2 = torch.linspace(2, 10, 6)
    print("tensor: ", t2)


# #######################  TYPE 3  #######################
# create tensor based on distribution

'''
TYPE 3: Create tensor based on distribution

1. torch.normal()

    torch.normal(mean,
                 std,
                 out=None)
    
    torch.normal(mean,
                 std,
                 size,
                 out-None)
       
2.1 torch.randn()

    torch.randn(*size,
                out=None,
                dtype=None,
                layout=torch.strided,
                device=None,
                requires_grad=False)
       
    Function: form a standard normal distribution   
             
2.2 torch.randn_like()
    
              
3.1 torch.rand()

    torch.rand(*size,
               out=None,
               dtype=None,
               layout=torch.strided,
               device=None,
               requires_grad=False)
               
    Function: form a uniform distribution in the interval of [0,1)

3.2 torch.rand_like()

    torch.randint(low=0,
                  high,
                  size,
                  out=None,
                  dtype=None,
                  layout=torch.strided,
                  device=None,
                  requires_grad=False)
                  
    Function: form a uniform distribution in the interval of [low, high)
    
    size: the shape of the tnesor
    
3.3 torch.randint()

3.4 torch.randint_like()

4.1 torch.randperm()

    torch.randperm(n,
                   out=None,
                   dtype=None,
                   layout=torch.strided,
                   device=None,
                   requires_grad=False)
                   
    Function: form a random list of 0 to n-1
    
4.2 torch.bernoulli()

    torch.bernoulli(input,
                    *,
                    generator=None,
                    out=None)
                    
    Function: Using input as probability, form a bernoulli distribution
'''

# #######################  example 7  #######################

# Create a tensor using torch.normal

# flag = True
flag = False

if flag:

    # mean, std are all tensor

    print("mean, std are all tensor")
    mean = torch.arange(1,5, dtype=torch.float)
    std = torch.arange(1,5, dtype=torch.float)
    t_normal = torch.normal(mean, std)
    print("mean:{} \n std:{}".format(mean, std))
    print(t_normal)

    # mean: value, std: value

    print("\n mean: value, std: value")
    t_normal = torch.normal(0., 1., size=(4,))
    print(t_normal)

    # mean: tensor, std: value

    print("\n mean: tensor, std: value")
    mean = torch.arange(1, 5, dtype=torch.float)
    t_normal = torch.normal(mean, 1)
    print(t_normal)

    # mean: tensor, std: value

    print("\n mean: tensor, std: value")
    std = torch.arange(1, 5, dtype=torch.float)
    t_normal = torch.normal(0., std)
    print(t_normal)


