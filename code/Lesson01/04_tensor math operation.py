import torch

'''
This file contains tensor math operation.

'''

# #######################  PART 1  #######################
# ADD, SUB, MULTIPLE, DIVIDE

'''
Part I: ADD, SUB, MULTIPLE, DIVIDE

1.1 torch.add()

    torch.add(input,
              alpha = 1,
              other,
              out=None)
              
    Function: return input + alpha * other
    
    Parameters:
        a. input: the first tensor
        b. alpha: the factor of multiple
        c. other: the second tensor
        
1.2 torch.addcdiv()

    Function: return input_i + value * (tensor1_i / tensor2_i)
    

1.3 torch.addcmul()

    Function: return input_i + value * tensor1_i * tensor2_i
    


'''

# #######################  example 1  #######################
# torch.add

flag = True
# flag = False

if flag:
    t_0 = torch.randn((3, 3))
    t_1 = torch.ones_like(t_0)

    t_add = torch.add(t_0, 10, t_1)

    print("t_0:{} \n t_1:{} \n t_add:{}".format(t_0, t_1, t_add))


# #######################  PART 2  #######################
# Log, Exp, Pow

'''
Part II: Log, Exp, Pow

1.1 torch.log(input, out=None)

1.2 torch.log10(input, out=None)

1.3 torch.log2(input, out=None)

1.4 torch.exp(input, out=None)

1.5 torch.pow()


'''


# #######################  PART 3  #######################
# Trigonometric Function

'''
Part II: Trigonometric Function

1.1 torch.abs(input, out=None)

1.2 torch.acos(input, out=None)

1.3 torch.cosh(input, out=None)

1.4 torch.cos(input, out=None)

1.5 torch.asin(input, out=None)

1.4 torch.atan(input, out=None)

1.4 torch.atan2(input, out=None)

'''