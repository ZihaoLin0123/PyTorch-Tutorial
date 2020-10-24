import torch
torch.manual_seed(10)

'''
This file contains autograd.

Automatic gradient system.
'''

'''
1. torch.autograd.backward

    torch.autograd.backward(tensors,
                            grad_tensors=None,
                            retain_graph=None,
                            create_graph=False)
                            
    Function: automatic calculate gradient
    
    Parameters:
        a. retain_graph: save the graph of calculation
        b. create_graph: create the graph of calculation, used for higher order gradient
        c. grad_tensors: the weight of multi gradient
        
'''

# #######################  retain_graph  #######################

# flag = True
flag = False

if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    y.backward(retain_graph=True)
    print(w.grad)
    y.backward()

    '''
    y.backward() could not run twice:
    
    error message:
    
    RuntimeError: Trying to backward through the graph a second time, 
    but the saved intermediate results have already been freed. 
    Specify retain_graph=True when calling backward the first time.

    '''

# #######################  retain_graph  #######################

# flag = True
flag = False

if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)

    y_0 = torch.mul(a, b)  # y_0 = (x + w) * (w + 1)
    y_1 = torch.add(a, b)  # y_1 = (x + w) + (w + 1)

    loss = torch.cat([y_0, y_1], dim=0)
    grad_tensors = torch.tensor([1., 1.])

    loss.backward(gradient=grad_tensors)  # import grad_tensors from torch.autograd.backward() to gradient

    print(w.grad)

'''
2. torch.autograd.grad

    torch.autograd.grad(outputs,
                        inputs,
                        grad_outputs=None,
                        retain_graph=None,
                        create_graph=False)
                        
    Function: calculate gradient
    
    Parameters:
        a. outputs: tensors for derivative, e.g. loss
        b. inputs: tensors for gradient, e.g. w, x
        
'''

# #######################  autograd.grad  #######################

# flag = True
flag = False

if flag:

    x = torch.tensor([3.], requires_grad=True)
    y = torch.pow(x, 2)  # y = x**2

    grad_1 = torch.autograd.grad(y, x, create_graph=True)  # grad_1 = dy/dx = 2x = 2 * 3 = 6
    print(grad_1)

    grad_2 = torch.autograd.grad(grad_1[0], x)
    print(grad_2)


'''
Tips:

1. gradient will not clear
2. requires_grad is True by default when the nodes depend on leaf variables
3. leaf variables could not run in-place
'''

# #######################  Tips: 1  #######################

# flag = True
flag = False

if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    for i in range(2):
        a = torch.add(w, x)
        b = torch.add(w, 1)
        y = torch.mul(a, b)

        y.backward()
        print(w.grad)

        # every time the gradient will add to itself.
        # therefore we should clear the previous result manually.

        w.grad.zero_()  # _ means in-place process

# #######################  Tips: 2  #######################

# flag = True
flag = False

if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    print(a.requires_grad, b.requires_grad, y.requires_grad)

# #######################  Tips: 3.1  #######################

# flag = True
flag = False

if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    # w.add_(1)
    # RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.

    # Because when calculate dy/da = w + 1, we will use w which requires the address of w in memory.
    # If we change the value in this address before backward, the final result will be wrong.

    y.backward()

# #######################  Tips: 3.2 in place  #######################

flag = True
# flag = False

if flag:

    a = torch.ones((1,))
    print(id(a), a)

    a = a + torch.ones((1, ))
    print(id(a), a)

    a += torch.ones((1, ))
    print(id(a), a)

    # a = a + 1, change the address in memory
    # a += 1, does not change the address in memory

