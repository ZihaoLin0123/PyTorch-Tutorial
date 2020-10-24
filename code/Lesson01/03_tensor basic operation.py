import torch

'''
This file contains tensor basic operation.

'''

# #######################  PART 1  #######################
# Tensor Operation

'''
Part I: Tensor concatenate and cut

1.1 torch.cat()

    torch.cat(tensors,
              dim=0,
              out=None)
              
    Function: concatenate tensors by dimension
    
1.2 torch.stack()

    torch.stack(tensors,
                dim=0,
                out=None)
                
    Function: concatenate tensors by dimension which is created newly
   
1.3 torch.chunk()

    torch.chunk(input,
                chunks,
                dim=0)
                
    Function: partition the tensor evenly according to dimensions
    
    Note: if it's not divisible, the last part will be smaller than the other parts
    
    Parameters:
        a. input: the tensor need to be partitioned
        b. chunks: the number of shares to be divided
        c. dim: the dimension to be partitioned
        
1.4 torch.split()

    torch.split(tensor,
                split_size_or_sections,
                dim=0)
                
    Function: partition the tensor evenly according to dimensions
    
    Parameters:
        a. input: the tensor need to be partitioned
        b. split_size_or_sections: int represents the length of each part,
                                   list means the tensor is divided by elements
        c. dim: the dimension to be partitioned


Part II: the index of tensor

2.1 torch.index_selection()

    torch.index_selection(input,
                          dim,
                          index,
                          out=None)
                          
    Function: select data by value in the dimension dim
    
2.2 torch.masked_select()

    torch.masked_select(input,
                        mask,
                        out=None)
                        
    Function: select values according to the True value of mask
    
    Return values: one dimension tensor
    
    Parameters:
        a. mask: has the same shape as input
        
        
Part III: tensor transforming

3.1 torch.reshape()

    torch.reshape(input,
                  shape)
                  
    Function: change the shape of tensor
    
    Note: when tensor is stored in memory sequential, the new tensor and input share the memory
    
3.2 torch.transpose()

    torch.transpose(input,
                    dim0,
                    dim1)
                    
    Function: change the dimension of the tensor
     
3.3 torch.t()

    torch.t(input)
    
    Function: same as torch.transpose(input, 0, 1)
    
3.4 torch.squeeze()

    torch.squeeze(input,
                  dim=None,
                  out=None)
                  
    Function: squeeze the dimensions whose lengths are 1
    
    Parameters:
        a. dim: if None, remove all dimensions whose lengths are 1,
                if specify the dimension, only when this dimension's length is 1, the dimension is removed

3.5 torch.unsqueeze()

    torch.unsqueeze(input,
                    dim,
                    out=None)
                    
    Function: expand dimensions according to dimension dim
'''

# #######################  example 1  #######################
# torch.cat

# flag = True
flag = False

if flag:
    t = torch.ones((2, 3))

    t_0 = torch.cat([t, t], dim=0)
    t_1 = torch.cat([t, t], dim=1)

    print("t_0:{} \n t_1:{}".format(t_0, t_1))

# #######################  example 2  #######################
# torch.stack

# flag = True
flag = False

if flag:
    t = torch.ones((2, 3))

    t_stack_0 = torch.stack([t, t], dim=0)
    t_stack_2 = torch.stack([t, t], dim=2)
    print("t_stack_0:{} \n t_stack_1:{}".format(t_stack_0, t_stack_2))

# #######################  example 3  #######################
# torch.chunk

# flag = True
flag = False

if flag:
    a = torch.ones((2, 5))  # 7

    list_of_tensors = torch.chunk(a, dim=1, chunks=2)  # 3

    for idx, t in enumerate(list_of_tensors):
        print("the {}-th tensor is {} whose shape is {}".format(idx, t, t.shape))

# #######################  example 4  #######################
# torch.split

# flag = True
flag = False

if flag:
    a = torch.ones((2, 5))  # 7

    list_of_tensors = torch.split(a, 2, dim=1)  # [2,1,2], the sum of this list should be equal to the dimension

    for idx, t in enumerate(list_of_tensors):
        print("the {}-th tensor is {} whose shape is {}".format(idx, t, t.shape))


# #######################  example 5  #######################
# torch.index_select

# flag = True
flag = False

if flag:
    a = torch.randint(0, 9, size=(3,3))
    idx = torch.tensor([0, 2], dtype=torch.long)  # float
    t_select = torch.index_select(a, dim=0, index=idx)
    print("t: \n {} \n t_select: \n {}".format(a, t_select))


# #######################  example 6  #######################
# torch.masked_select

# flag = True
flag = False

if flag:
    a = torch.randint(0, 9, size=(3,3))
    mask = a.ge(5) # ge: greater than or equal to / gt: greater than / le: <= / lt: <
    t_select = torch.masked_select(a, mask)
    print("a: \n {} \n mask: \n {} \n t_select: \n {}".format(a, mask, t_select))


# #######################  example 7  #######################
# torch.reshape

# flag = True
flag = False

if flag:
    a = torch.randperm(8)
    t_reshape = torch.reshape(a, (2, 4))  # -1, means don't care this dimension
    print("t:{} \n t_reshape: \n {}".format(a, t_reshape))

    a[0] = 1024
    print("a:{} \n t_reshape: \n {}".format(a, t_reshape))
    print("a.data memory address:{}".format(id(a.data)))
    print("t_reshape.data memory address:{}".format(id(t_reshape.data)))

# #######################  example 7  #######################
# torch.transpose

# flag = True
flag = False

if flag:
    a = torch.rand(2,3,4)
    t_transpose = torch.transpose(a, dim0=1, dim1=2)
    print("a shape:{} \n t_transpose shape:{}".format(a.shape, t_transpose.shape))


# #######################  example 8  #######################
# torch.stack

flag = True
# flag = False

if flag:
    a = torch.rand((1, 2, 3, 1))
    a_sq = torch.squeeze(a)
    a_0 = torch.squeeze(a, dim=0)
    a_1 = torch.squeeze(a, dim=1)

    print(a)
    print(a.shape)
    print(a_sq.shape)
    print(a_0.shape)
    print(a_1.shape)
