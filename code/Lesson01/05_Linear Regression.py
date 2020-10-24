import torch
import matplotlib.pyplot as plt

'''
This file contains linear regression.

'''

'''
Steps:

1.  Model

    y = wx + b
    
2. Loss Function

    MSE = 1/m * (sum of (yi - yi_)^2 )(from i=1 to m)
    
    where m is the number of data in the data set, yi_ is the
    estimated value of yi
    
3. Solve the gradient and update w & b

    w = w - LR * w.grad
    b = b - LR * w.grad
    
    where LR is learning rate, or called step length
'''

torch.manual_seed(10)

lr = 0.1  # learning rate

# create training data
x = torch.rand(20, 1) * 10  # x data (tensor), shape = (20,1)
y = 2 * x + (5 + torch.randn(20, 1)) # y data (tensor), shape = (20, 1)

# create parameters of linear regression
w = torch.randn((1), requires_grad=True)
b = torch.zeros((1), requires_grad=True)

for iteration in range(1000):

    # feed forward network

    wx = torch.mul(w, x)
    y_pred = torch.add(wx, b)

    # calculate MSE loss
    loss = (0.5 * (y - y_pred) ** 2).mean()

    # backward propagate
    loss.backward()

    # update parameters
    b.data.sub_(lr * b.grad)
    w.data.sub_(lr * w.grad)

    # plot
    if iteration % 20 == 0:

        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
        plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size':20, 'color': 'red'})
        plt.xlim(1.5, 10)
        plt.ylim(8, 28)
        plt.title("Iteration: {} \n w: {} \n b: {}".format(iteration, w.data.numpy(), b.data.numpy()))
        plt.pause(0.5)

        if loss.data.numpy() < 1:
            break




