import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import matplotlib.pylab as pylab
import sys

params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

np.random.seed(0)
torch.manual_seed(42)

train_Source = True

# Choose from 'entropy_min', 'pseudolabel'
algo = sys.argv[1]
if (algo=='entropy_min'):
    entropy_Min = True
else:
    entropy_Min = False

n_T = 10000
n_S = 10000
d1 = 2 # x1 feature dimension
d2 = 2 # x2 feature dimension
sigma1 = 1 # x1 standard deviation
sigma2 = 1 # x2 standard deviation
x1_type = '2Mixture'

c_len = 2 # length of mean of 2 mixtures for x1
thr = 0.1 # Pseudolabeling cutoff

# Create Source Domain
c = torch.empty(1, d1).normal_(mean=0,std=sigma1)
c = nn.functional.normalize(c, p=2, dim=1) * c_len
print('Cluster mean', c)

w_star = torch.Tensor([c[0, 0], c[0, 1], 0, 0])
print('w_star', w_star)

# Label is 0/1
Y_S = torch.Tensor(n_S, 1).random_(0, 2)
X_S_1 = torch.empty(n_S, d1).normal_(mean=0,std=sigma1)
X_S_1 = X_S_1 + (2 * Y_S - 1) * c
X_S_2 = torch.empty(n_S, d2).normal_(mean=0,std=sigma2)
X_S = torch.cat((X_S_1, X_S_2), 1)


cor_prob = 0.8 # 80% of x2 are correlated with labels
for i in range(n_S):
    for j in range(d2):
        coin_flip = np.random.binomial(n=1, p=cor_prob)
        if (coin_flip==1):
            X_S[i,j+d1] = np.abs(X_S[i,j+d1]) * (Y_S[i]*2-1)

# Create Target Domain Test Set
Y_T_test = torch.Tensor(n_T, 1).random_(0, 2)
X_T_1 = torch.empty(n_T, d1).normal_(mean=0,std=sigma1)
X_T_1 = X_T_1 + (2 * Y_T_test - 1) * c
X_T_2 = torch.empty(n_T, d2).normal_(mean=0,std=sigma2)
X_T_test = torch.cat((X_T_1, X_T_2), 1)

Y_T_test = Y_T_test.detach().numpy()

if train_Source:
    # Train on Source
    w_S_hat = torch.randn(d1+d2, requires_grad=True, dtype=torch.float, device='cpu')

    lr = 1e-3
    n_epochs = 20000

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD([w_S_hat], lr=lr, weight_decay=0)

    acc_plot = []
    spu_1_plot = []
    spu_2_plot = []
    vio_plot = []
    train_ent_plot = []
    test_ent_plot = []

    for epoch in range(n_epochs):

        optimizer.zero_grad()
        yhat = torch.matmul(X_S, w_S_hat).reshape(n_S, 1)
        loss = loss_fn(yhat, Y_S)

        # Train on source labels + target entropy
        # Create a new batch of target domain training data
        Y_T_train = torch.Tensor(n_T, 1).random_(0, 2)
        X_T_1 = torch.empty(n_T, d1).normal_(mean=0,std=sigma1)
        X_T_1 = X_T_1 + (2 * Y_T_train - 1) * c
        X_T_2 = torch.empty(n_T, d2).normal_(mean=0,std=sigma2)
        X_T_train = torch.cat((X_T_1, X_T_2), 1)

        exp_wx = torch.exp(torch.matmul(X_T_train, w_S_hat))
        exp_negwx = torch.exp(-torch.matmul(X_T_train, w_S_hat))
        # Calculate entropy objective
        T_loss = torch.mean(torch.log(1+exp_negwx) / (1+exp_negwx) + torch.log(1+exp_wx) / (1+exp_wx))
        #loss = loss + 100*T_loss

        loss.backward()
        optimizer.step()

        # Normalize w_S_hat to unit length
        with torch.no_grad():
            l2norm = w_S_hat.norm(p=2, dim=0).detach()
            w_S_hat.data = w_S_hat.data / l2norm

        if (epoch % 100 == 0):
            # Calculate Test Entropy
            exp_wx = torch.exp(torch.matmul(X_T_test, w_S_hat))
            exp_negwx = torch.exp(-torch.matmul(X_T_test, w_S_hat))
            test_ent = torch.log(1+exp_negwx) / (1+exp_negwx) + torch.log(1+exp_wx) / (1+exp_wx)
            test_ent = torch.mean(test_ent)

            # Target test accuracy
            Y_T_test_pred = ((torch.sign(torch.matmul(X_T_test, w_S_hat))+1)/2).reshape(n_T, 1).detach().numpy()
            acc_real = 1-np.float(np.count_nonzero(Y_T_test_pred != Y_T_test)) / n_T
            print(epoch, ': w_S_hat =', w_S_hat.detach().numpy(), 'Target test accuracy =', acc_real)
            print('Training loss', loss.item())
            acc_plot.append(acc_real)
            spu_1_plot.append(w_S_hat.detach().numpy()[d1])
            if (d2>1):
                spu_2_plot.append(w_S_hat.detach().numpy()[d1+1])
            train_ent_plot.append(T_loss.item())
            test_ent_plot.append(test_ent.item())
else:
    # Select specific start point for Pseudolabeling
    w_S_hat = torch.Tensor([0.8236328, 0.31852758, 0.32869998, 0.3348515])
    w_S_hat = nn.functional.normalize(w_S_hat, p=2, dim=0)

# Pseudolabel for target test data
Y_T_pdo = ((torch.sign(torch.matmul(X_T_test, w_S_hat))+1)/2).reshape(n_T, 1).detach().numpy()
acc_real = 1-np.float(np.count_nonzero(Y_T_pdo != Y_T_test)) / n_T
print('After training on source: w_S_hat =', w_S_hat.detach().numpy(), 'Target Accuracy =', acc_real)

ori_target_acc = acc_real
ori_spurious_1 = w_S_hat.detach().numpy()[d1]
if (d2>1):
    ori_spurious_2 = w_S_hat.detach().numpy()[d1+1]

# Bayes optimal accuracy
Y_T_star = ((torch.sign(torch.matmul(X_T_test, w_star))+1)/2).reshape(n_T, 1).detach().numpy()
acc_bayes= 1-np.float(np.count_nonzero(Y_T_star != Y_T_test)) / n_T
print('Bayes Optimal Target Accuracy =', acc_bayes)


# Gradient Descent on entropy objective
if entropy_Min:
    # Start with source classifier
    w_T = w_S_hat.detach().requires_grad_()

    acc_plot = []
    spu_1_plot = []
    spu_2_plot = []
    vio_plot = []
    train_ent_plot = []
    test_ent_plot = []

    lr = 1e-3 # learning rate
    n_epochs = 10000 # number of GD steps
    optimizer = optim.SGD([w_T], lr=lr, weight_decay=0)

    for iter in range(n_epochs):

        # Create a new batch of Target Domain Training Data
        Y_T_train = torch.Tensor(n_T, 1).random_(0, 2)
        X_T_1 = torch.empty(n_T, d1).normal_(mean=0,std=sigma1)
        X_T_1 = X_T_1 + (2 * Y_T_train - 1) * c
        X_T_2 = torch.empty(n_T, d2).normal_(mean=0,std=sigma2)
        X_T_train = torch.cat((X_T_1, X_T_2), 1)

        exp_wx = torch.exp(torch.matmul(X_T_train, w_T))
        exp_negwx = torch.exp(-torch.matmul(X_T_train, w_T))
        # Calculate entropy objective
        loss = torch.log(1+exp_negwx) / (1+exp_negwx) + torch.log(1+exp_wx) / (1+exp_wx)
        loss = torch.mean(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            l2norm = w_T.norm(p=2, dim=0).detach()
            w_T.data = w_T.data / l2norm

        if (iter % 100 == 0):

            # Calculate Test Entropy
            exp_wx = torch.exp(torch.matmul(X_T_test, w_T))
            exp_negwx = torch.exp(-torch.matmul(X_T_test, w_T))
            test_ent = torch.log(1+exp_negwx) / (1+exp_negwx) + torch.log(1+exp_wx) / (1+exp_wx)
            test_ent = torch.mean(test_ent)

            # Violation of Pseudolabel
            Y_T_test_pred = ((torch.sign(torch.matmul(X_T_test, w_T))+1)/2).reshape(n_T, 1).detach().numpy()
            vio_pdo = np.float(np.count_nonzero(Y_T_test_pred != Y_T_pdo)) / n_T

            # Target test accuracy
            acc_real = 1-np.float(np.count_nonzero(Y_T_test_pred != Y_T_test)) / n_T
            print(iter, ': w_T =', w_T.detach().numpy(), 'Target test accuracy =', acc_real)
            print('Training loss', loss.item())

            # Record for plotting
            vio_plot.append(vio_pdo)
            acc_plot.append(acc_real)
            spu_1_plot.append(w_T.detach().numpy()[d1])
            if (d2>1):
                spu_2_plot.append(w_T.detach().numpy()[d1+1])
            train_ent_plot.append(loss.item())
            test_ent_plot.append(test_ent.item())

    fig, axs = plt.subplots(2)
    axs[0].plot(acc_plot)
    axs[0].set(xlabel = 'Training epochs')
    axs[0].set(ylabel = 'Target Accuracy')
    ind_max = np.argmax(acc_plot)
    val_max = max(acc_plot)
    axs[0].hlines(val_max, ind_max-10, ind_max+10, colors='m', label='Max accuracy '+str(round(val_max,3)))
    axs[0].hlines(ori_target_acc, 0, n_epochs/100, colors='g', linestyles='dashed', label='Source classifier acc '+str(round(ori_target_acc,3)))
    axs[0].hlines(acc_bayes, 0, n_epochs/100, colors='r', linestyles='dashed', label='Bayes optimal acc '+str(round(acc_bayes,3)))
    axs[0].legend()
    axs[1].plot(spu_1_plot, label='Spurious coordinate 1 coef')
    axs[1].hlines(ori_spurious_1, 0, n_epochs/100, colors='g', linestyles='dashed', label='Source classifier coef-1 = '+str(round(ori_spurious_1,3)))
    if (d2>1):
        axs[1].plot(spu_2_plot, label='Spurious coordinate 2 coef')
        axs[1].hlines(ori_spurious_2, 0, n_epochs/100, colors='m', linestyles='dashed', label='Source classifier coef-2 = '+str(round(ori_spurious_2,3)))
    axs[1].set(xlabel = 'Training epochs')
    axs[1].set(ylabel = 'Spurious coefficients')
    axs[1].legend()
    plt.show()

else:
    # Pseudolabeling Algorithm
    # Number of rounds of Pseudolabeling
    num_rounds = 200
    # lr and num_epochs for each round
    lr = 1e-3
    n_epochs = 50
    loss_fn = nn.BCEWithLogitsLoss()

    # Teacher classifier for each round
    w_T_hat = w_S_hat.clone().detach()
    w_T = w_T_hat
    w_T.requires_grad = True

    acc_plot = []
    spu_1_plot = []
    spu_2_plot = []

    # Reinitialize student classifier
    optimizer = optim.SGD([w_T], lr=lr, weight_decay=0)

    for r in range(num_rounds):

        # Create a new batch of Target Domain Training Data
        Y_T_train = torch.Tensor(n_T, 1).random_(0, 2)
        X_T_1 = torch.empty(n_T, d1).normal_(mean=0,std=sigma1)
        X_T_1 = X_T_1 + (2 * Y_T_train - 1) * c
        X_T_2 = torch.empty(n_T, d2).normal_(mean=0,std=sigma2)
        X_T_train_can = torch.cat((X_T_1, X_T_2), 1).detach().numpy()

        # Pseudolabel
        Y_pdo_pred = np.matmul(X_T_train_can, w_T_hat.detach().numpy())
        # Take most confident examples (abs(w*X) > thr)
        X_T_train = []
        Y_T_pdo = []
        for i in range(n_T):
            if (np.abs(Y_pdo_pred[i]) > thr):
                X_T_train.append(X_T_train_can[i])
                Y_T_pdo.append((np.sign(Y_pdo_pred[i])+1)/2)
        X_T_train = torch.Tensor(X_T_train)
        Y_T_pdo = torch.Tensor(Y_T_pdo)

        for epoch in range(n_epochs):

            optimizer.zero_grad()
            yhat = torch.matmul(X_T_train, w_T)
            loss = loss_fn(yhat, Y_T_pdo)
            loss.backward()
            optimizer.step()

            # Normalize w_S_hat to unit length
            with torch.no_grad():
                l2norm = w_T.norm(p=2, dim=0).detach()
                w_T.data = w_T.data / l2norm

            if (epoch % 30000 == -1):

                # Target test accuracy
                Y_T_test_pred = ((torch.sign(torch.matmul(X_T_test, w_T))+1)/2).reshape(n_T, 1).detach().numpy()
                acc_real = 1-np.float(np.count_nonzero(Y_T_test_pred != Y_T_test)) / n_T
                print(epoch, ': w_T =', w_T.detach().numpy(), 'Target test accuracy =', acc_real)
                print('Pseudolabeling training loss', loss.item())

                acc_plot.append(acc_real)
                spu_1_plot.append(w_T.detach().numpy()[d1])
                if (d2>1):
                    spu_2_plot.append(w_T.detach().numpy()[d1+1])

        w_T_hat = w_T.clone().detach()

        # Target test accuracy
        Y_T_test_pred = ((torch.sign(torch.matmul(X_T_test, w_T))+1)/2).reshape(n_T, 1).detach().numpy()
        acc_real = 1-np.float(np.count_nonzero(Y_T_test_pred != Y_T_test)) / n_T
        print('Round: ', r, ': w_T =', w_T.detach().numpy(), 'Target test accuracy =', acc_real)
        acc_plot.append(acc_real)
        spu_1_plot.append(w_T.detach().numpy()[d1])
        if (d2>1):
            spu_2_plot.append(w_T.detach().numpy()[d1+1])

    fig, axs = plt.subplots(2)
    axs[0].plot(acc_plot)
    axs[0].set(xlabel = 'Training epochs')
    axs[0].set(ylabel = 'Target accuracy')
    ind_max = np.argmax(acc_plot)
    val_max = max(acc_plot)
    axs[0].hlines(val_max, ind_max-10, ind_max+10, colors='m', label='Max accuracy '+str(round(val_max,3)))
    axs[0].hlines(ori_target_acc, 0, num_rounds, colors='g', linestyles='dashed', label='Source classifier acc '+str(round(ori_target_acc,3)))
    axs[0].hlines(acc_bayes, 0, num_rounds, colors='r', linestyles='dashed', label='Bayes optimal acc '+str(round(acc_bayes,3)))
    axs[0].legend()

    axs[1].plot(spu_1_plot, label='Spurious coordinate 1 coef')
    axs[1].hlines(ori_spurious_1, 0, num_rounds, colors='g', linestyles='dashed', label='Source classifier coef-1 = '+str(round(ori_spurious_1,3)))
    if (d2>1):
        axs[1].plot(spu_2_plot, label='Spurious coordinate 2 coef')
        axs[1].hlines(ori_spurious_2, 0, num_rounds, colors='m', linestyles='dashed', label='Source classifier coef-2 = '+str(round(ori_spurious_2,3)))
    axs[1].set(xlabel = 'Training epochs')
    axs[1].set(ylabel = 'Spurious coefficients')
    axs[1].legend()
    plt.show()
