import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(10)

'''
This file contains logistic regression.
'''

'''
Logistic regression is a dichotomy model.

y = f(Wx + b)

f(x) = 1/(1 + e^(-x))

f(x) is sigmoid function, also called logistic function.

'''

'''
Five steps of training machine learning model

1. Data:
    a. getting data
    b. data cleaning
    
2. Model:
    choose different models according to the question
    
3. Loss funcion
    a. choose function
    b. calculate gradient
    
4. Optimizer
    optimize parameters of models

5. Iteration training
'''

# #######################  step1: getting data  #######################
sample_nums = 100
mean_value = 1.7
bias = 1
n_data = torch.ones(sample_nums, 2)
x0 = torch.normal(mean_value * n_data, 1) + bias   # class 0：data shape = (100, 2)
y0 = torch.zeros(sample_nums)                      # class 0: label shape = (100, 1)
x1 = torch.normal(-mean_value * n_data, 1) + bias  # class 1: data shape = (100, 2)
y1 = torch.ones(sample_nums)                       # class 1: label shape=(100, 1)
train_x = torch.cat((x0, x1), 0)
train_y = torch.cat((y0, y1), 0)


# #######################  step2: model  #######################
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.features = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)
        return x

lr_net = LR()  # instanse logistic regression model


# #######################  step3: loss function  #######################

loss_fn = nn.BCELoss()
# binary cross entropy


# #######################  step4: optimizer  #######################
lr = 0.01  # learning rate
optimizer = torch.optim.SGD(lr_net.parameters(), lr=lr, momentum=0.9)


# #######################  step5: training model  #######################
for iteration in range(1000):

    # forward propagate
    y_pred = lr_net(train_x)

    # calculate loss function
    loss = loss_fn(y_pred.squeeze(), train_y)

    # backward propagate
    loss.backward()

    # update parameters
    optimizer.step()

    # plot
    if iteration % 20 == 0:

        mask = y_pred.ge(0.5).float().squeeze()  # classify by 0.5

        correct = (mask == train_y).sum()  # calculate the number of correct prediction
        # something wrong with the previous one, I have nor figured it out why


        acc = correct.item() / train_y.size(0)  # calculate the accurate rate of prediction

        plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c='r', label='class 0')
        plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c='b', label='class 0')

        w0, w1 = lr_net.features.weight[0]
        w0, w1 = float(w0.item()), float(w1.item())
        plot_b = float(lr_net.features.bias[0].item())
        plot_x = np.arange(-6, 6, 0.1)
        plot_y = (-w0 * plot_x - plot_b) / w1

        plt.xlim(-5, 7)
        plt.ylim(-7, 7)
        plt.plot(plot_x, plot_y)

        plt.text(-5, 5, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.title("Iteration: {}\nw0:{:.2f} w1:{:.2f} accuracy:{:.2%}".format(iteration, w0, w1, plot_b, acc))
        plt.legend()

        plt.show()
        plt.pause(0.5)

        if acc > 0.99:
            break



