import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.utils as vutils
import torch.nn.functional as F
from torch import nn, optim
import pickle
import shutil, os
from util import save_ckp, load_ckp, vat_loss
import matplotlib.pylab as pylab

params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

torch.manual_seed(42)
np.random.seed(0)

def get_colors(num, noise_type):
    if noise_type == 'Uniform':
        colors = torch.rand(num, 1, 1).repeat(1, 28, 28)
    elif noise_type == 'Gaussian':
        colors = torch.empty(num, 1, 1).normal_(mean=0.5,std=0.5/3).repeat(1, 28, 28)
    else:
        colors = torch.ones(num, 1, 1).repeat(1, 28, 28) * 0.5
    return colors

def construct_MNIST_dataset(noise_type, cor_prob, n_way):
    num_S_tot = num_S_train + num_S_test
    transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,)),
                                  ])

    trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=False, train=True, transform=transform)
    S_images = trainset.data[:num_S_tot].float() / 255.0
    # 10-way Version:
    S_labels = trainset.targets[:num_S_tot]
    # Binary Version:
    if (n_way==2):
        S_labels = (S_labels < 5).long()

    # Binomial mask for whether use label or random bit for color
    col_mask = torch.Tensor(np.random.binomial(n=1, p=cor_prob, size=(num_S_tot, 1)))
    if (noise_type=='Uniform'):
        col_lab = S_labels.view(num_S_tot, 1).float() / n_way
        # Smoothing out one color to be uniform within a 0.1-interval
        col_lab = col_lab + torch.rand(num_S_tot, 1) / n_way
        col_rnd = torch.rand(num_S_tot, 1)
    else:
        # Only makes sense when n_way=2
        col_rnd = torch.empty(num_S_tot, 1).normal_(mean=0.5,std=0.5/3)
        # If label = 1, make col_lab upper half of the Gaussian
        sign_lab = torch.sign(S_labels.view(num_S_tot, 1).float()-0.5)
        col_lab = torch.abs(col_rnd-0.5)*sign_lab+0.5

    colors = col_lab * col_mask + col_rnd * (1-col_mask)

    # Expand colors to whole channel
    colors = colors.view(num_S_tot, 1, 1).repeat(1, 28, 28)
    S_images = torch.stack([S_images*colors, S_images*(1-colors)], dim=-1)
    S_train_images = S_images[:num_S_train]
    S_train_labels = S_labels[:num_S_train]
    S_test_images = S_images[num_S_train:num_S_tot]
    S_test_labels = S_labels[num_S_train:num_S_tot]

    # Comment out below
    # Display images only works for 3 channels
    """
    figure = plt.figure()
    num_of_images = 60
    for index in range(1, num_of_images + 1):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(images[index].numpy())
    plt.show()
    """

    testset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=False, train=False, transform=transform)
    T_train_images = trainset.data[-num_T_train:].float() / 255.0
    T_test_images = testset.data[:num_T_test].float() / 255.0
    T_test_labels = testset.targets[:num_T_test]
    # Binary Version:
    if (n_way==2):
        T_test_labels = (T_test_labels < 5).long()
    # Use random color for Target
    # Uniform random between 0 and 1 to avoid clustering around colors
    #colors = torch.randint(0, 10, (testset.data.shape[0], 1)).float() / 10.0

    colors_train = get_colors(num_T_train, noise_type)
    colors_test = get_colors(num_T_test, noise_type)

    #T_images = torch.stack([images*colors, images*(1-colors), torch.zeros_like(images)], dim=-1) # 3-channel for demo
    T_train_images = torch.stack([T_train_images*colors_train, T_train_images*(1-colors_train)], dim=-1) # 2-channel
    T_test_images = torch.stack([T_test_images*colors_test, T_test_images*(1-colors_test)], dim=-1) # 2-channel

    return S_train_images, S_train_labels, S_test_images, S_test_labels, T_train_images, T_test_images, T_test_labels

def get_model(model_type, input_size, output_size):
    hidden_sizes = [128, 64]

    if model_type == 'Linear':
        # Linear Model
        nmodel = nn.Sequential(nn.Linear(input_size, output_size))
    elif model_type == '3-layer':
        # 3-layer model
        nmodel = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size))

    return nmodel

# Test on Source
def acc_On_Source(model):
    criterion = nn.CrossEntropyLoss()
    img = S_test_images.view(num_S_test, -1)
    with torch.no_grad():
        output = model(img)
    pred_label = np.argmax(output.numpy(), axis=1)
    true_label = S_test_labels.numpy()
    loss = criterion(input=output, target=S_test_labels)

    accuracy = np.sum(pred_label == true_label) / float(num_S_test)
    print("Source Test Accuracy, Loss =", accuracy, loss.item())
    return accuracy, loss.item()

# Test on Target
def acc_On_Target(model):
    criterion = nn.CrossEntropyLoss()
    img = T_test_images.view(num_T_test, -1)
    with torch.no_grad():
        output = model(img)
    pred_label = np.argmax(output.numpy(), axis=1)
    true_label = T_test_labels.numpy()
    loss = criterion(input=output, target=T_test_labels)

    accuracy = np.sum(pred_label == true_label) / float(num_T_test)
    print("Target Test Accuracy, Loss =", accuracy, loss.item())
    return accuracy, loss.item()

# Train on Source
def train_Source(loss_fn_choice):
    criterion = nn.CrossEntropyLoss()
    softmax_layer = nn.Softmax(dim=1)
    # Default: 0.003, 0.9, 0.002
    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=0.002)
    time0 = time()
    if (loss_fn_choice=='Mixed'):
        epochs = 500
        batch_size = 30000
    else:
        epochs = 30
        batch_size = 64
    num_batches = (num_S_train + batch_size - 1) // batch_size
    S_rec = []
    T_rec = []
    for e in range(epochs):
        running_loss = 0
        # Shuffle data in each epoch - Todo
        for b in range(num_batches):
            end_idx = min((b+1)*batch_size, num_S_train)
            images = S_train_images[b*batch_size : end_idx]
            labels = S_train_labels[b*batch_size : end_idx]
            images = images.view(images.shape[0], -1)
            if (loss_fn_choice=='Mixed'):
                end_idx_T = min((b+1)*batch_size, num_T_train)
                images_T = T_train_images[b*batch_size : end_idx_T]
                images_T = images_T.view(images_T.shape[0], -1)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(input=logits, target=labels)

            if (loss_fn_choice=='Mixed'):
                # Compute output from a batch in target
                logits_T = model(images_T)
                output_T = softmax_layer(logits_T)
                loss_entropy = torch.mean(torch.sum(- torch.log(output_T) * output_T, 1))
                beta = 0.1
                loss = loss + beta * loss_entropy
                print('Epoch', e, 'Total loss=', loss.item(), 'loss_entropy=', loss_entropy.item())

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Epoch {} - Training loss: {}".format(e, running_loss/num_S_train))
        acc, los = acc_On_Source(model)
        S_rec.append(los)
        acc, los = acc_On_Target(model)
        T_rec.append(los)
    print("\nTraining Time (in minutes) =",(time()-time0)/60)

    plt.figure(1)
    plt.plot(S_rec, label = 'Source Test Loss')
    plt.plot(T_rec, label = 'Target Test Loss')
    plt.ylabel('Test Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


# Violation of Pseudolabel
def vio_Pdo_Label(model):
    img = T_train_images.view(num_T_train, -1)
    with torch.no_grad():
        output = model(img)
    pred_label = np.argmax(output.numpy(), axis=1)
    true_label = pdo_label

    print("\nNumber Of Target Training Images =", num_T_train)
    violation = (1-np.sum(pred_label == true_label) / num_T_train)
    print("Violation of Pseudolabel =", violation)
    return violation

def acc_On_Target_Train(model):
    img = T_train_images.view(num_T_train, -1)
    with torch.no_grad():
        output = model(img)
    pred_label = np.argmax(output.numpy(), axis=1)
    true_label = pdo_label

    print("\n*** Number Of Target Training Images =", num_T_train)
    accuracy = (np.sum(pred_label == true_label) / num_T_train)
    print("Model Accuracy =", accuracy)
    return accuracy

# Plot activation distribution for a single x_1 via sampling x_2
def plot_dist(model, img_idx, target_per, img_ori, plot=False):
    transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,)),
                                  ])
    testset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=False, train=False, transform=transform)
    num_samples = 1000
    test_images = torch.Tensor()
    images = torch.stack([testset.data[img_idx].float()/255.0]*num_samples, dim=0)
    true_label = (testset.targets[img_idx]<5).long().numpy()
    #print('Plot example image with true label', true_label)
    if (target_per=='Uniform'):
        colors = torch.rand(num_samples, 1, 1).repeat(1, 28, 28)
    else:
        colors = torch.empty(num_samples, 1, 1).normal_(mean=0.5,std=0.5/3).repeat(1, 28, 28)
    images = torch.stack([images*colors, images*(1-colors)], dim=-1) # 2-channel
    images = images.view(num_samples, -1)
    with torch.no_grad():
        output = model(images)
    # Mark where original image lies
    with torch.no_grad():
        output_ori = model(img_ori)

    # Calculate avg activations before softmax_layer
    act_rec = (output[:,true_label]-output[:,1-true_label]).numpy()
    mean_act = np.mean(act_rec)
    std_act = np.std(act_rec)
    ori_act = float((output_ori[0,true_label]-output_ori[0,1-true_label]).numpy())
    if plot:
        plt.figure(1)
        plt.hist(act_rec, normed=False, bins=100)
        plt.ylabel('Frequency')
        plt.vlines(mean_act, 0, 60, colors='m', linestyles='solid', label='Mean of Activations '+str(round(mean_act, 2)))
        plt.vlines(ori_act, 0, 60, colors='g', linestyles='dashed', label='Test Img Activation '+str(round(ori_act, 2)))
        plt.hlines(30, mean_act-std_act, mean_act+std_act, colors='r', linestyles='solid', label='Std of Activations '+str(round(std_act, 2)))
        plt.xlim(-7.5, 7.5)
        plt.ylim(0, 60)
        plt.xlabel('Activations')
        plt.legend()
        plt.show()

    return mean_act, std_act

# GD on Entropy Objective
def finetune_Target(loss_fn_choice):
    ori_target_acc, _ = acc_On_Target(model)
    softmax_layer = nn.Softmax(dim=1)
    # Default is 0.003, 0.9, 0.002
    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=0.002)
    time0 = time()
    # Default is 100/30000
    epochs = 300 # Use 300
    batch_size = num_T_train
    vio_threshold = 1
    num_batches = (num_T_train + batch_size - 1) // batch_size
    # Record Target Test Accuracy for Plotting
    target_acc = []
    vio_rec = []
    loss_exp_rec = []
    loss_sqrt_rec = []
    loss_entropy_rec = []
    best_acc = 0
    for e in range(epochs):
        running_loss = 0
        # Shuffle data in each epoch - Todo
        for b in range(num_batches):
            end_idx = min((b+1)*batch_size, num_T_train)
            images = T_train_images[b*batch_size : end_idx]
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad()
            logits = model(images)
            output = softmax_layer(logits)
            # Use customized loss
            loss_entropy = torch.mean(torch.sum(- torch.log(output) * output, 1))
            loss_sqrt = torch.mean(torch.exp(-torch.sqrt(torch.abs(logits[:, 0]-logits[:, 1]))))
            loss_exp = torch.mean(torch.exp(-torch.abs(logits[:, 0]-logits[:, 1])))
            #print('batch', b, 'loss_exp', loss_exp.detach().numpy(), 'loss_entropy', loss_entropy.detach().numpy())
            loss_entropy_rec.append(loss_entropy.item())
            loss_exp_rec.append(loss_exp.item())
            loss_sqrt_rec.append(loss_sqrt.item())

            if (loss_fn_choice == 'Entropy'):
                loss = loss_entropy
            elif (loss_fn_choice == 'Exp'):
                loss = loss_exp
            else:
                loss = loss_sqrt

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Epoch {} - Training loss: {}".format(e, running_loss/num_batches))

        violation = vio_Pdo_Label(model)
        vio_rec.append(violation)
        t_acc, t_los = acc_On_Target(model)
        target_acc.append(t_acc)
        if t_acc > best_acc:
            best_model = pickle.loads(pickle.dumps(model))

        if (violation > vio_threshold):
            break;
    print("\nTraining Time (in minutes) =",(time()-time0)/60)

    plt.figure(1)
    #plt.subplot(231)
    plt.plot(target_acc)
    plt.hlines(ori_target_acc, 0, 20, label='Source Classifier Accuracy '+str(round(ori_target_acc, 2)))
    plt.xlabel('Training Epochs')
    plt.ylabel('Target Accuracy')
    ind_max = np.argmax(target_acc)
    val_max = max(target_acc)
    #plt.hlines(val_max, ind_max-10, ind_max+10, label='Max Accuracy '+str(round(val_max,2)))
    plt.legend()
    plt.show()

    plt.figure(2)
    """
    plt.subplot(232)
    plt.plot(vio_rec)
    plt.xlabel('Epochs')
    plt.ylabel('% Violation of Pseudolabels')
    plt.subplot(233)
    """
    plt.plot(loss_exp_rec)
    plt.xlabel('Training Epochs')
    plt.ylabel('Train Exponential Loss')
    plt.show()

    plt.figure(3)
    plt.plot(loss_entropy_rec)
    plt.xlabel('Training Epochs')
    plt.ylabel('Train Entropy Loss')
    plt.show()

    return best_model

# Find wrong predictions
def wrong_idx(model):
    img = T_test_images.view(num_T_test, -1)
    with torch.no_grad():
        output = model(img)
    pred_label = np.argmax(output.numpy(), axis=1)
    true_label = T_test_labels.numpy()
    return list(np.where(1-np.equal(pred_label, true_label))[0])

# Plot distribution of mu - f(x1, 0.5) over target testset
def plot_mu(model):
    transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,)),
                                  ])
    testset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=False, train=False, transform=transform)
    T_plot_images = testset.data[:num_T_test].float() / 255.0
    colors_plot = get_colors(num_T_test, noise_type='None')
    T_plot_images = torch.stack([T_plot_images*colors_plot, T_plot_images*(1-colors_plot)], dim=-1)
    true_label = (testset.targets[:num_T_test]<5).long().reshape(-1, 1)

    img = T_plot_images.view(num_T_test, -1)
    with torch.no_grad():
        output_mu = model(img)

    # Make histogram
    # Calculate avg activations before softmax_layer
    mu_rec = (torch.gather(output_mu, 1, torch.ones_like(true_label)) - torch.gather(output_mu, 1, torch.zeros_like(true_label))).numpy()
    #mu_rec = (torch.gather(output_mu, 1, true_label) - torch.gather(output_mu, 1, 1-true_label)).numpy()
    mean_act = np.mean(mu_rec)
    std_act = np.std(mu_rec)

    plt.figure(1)
    plt.hist(mu_rec, normed=True, bins=100)
    plt.ylabel('Probability')
    plt.ylim(0, 0.3)
    #plt.vlines(mean_act, 0, 0.15, colors='m', linestyles='solid', label='Mean of Mean Activations '+str(mean_act))
    #plt.hlines(0.075, mean_act-std_act, mean_act+std_act, colors='r', linestyles='solid', label='Std of Activations '+str(std_act))
    plt.xlabel('Mean activations')
    plt.xlim(-15, 15)
    #plt.legend()
    plt.show()

# ============ Main algorithm below ============
dataset = 'MNIST' # Choose from 'MNIST', 'CelebA'

if dataset == 'MNIST':
    n_way = 10 # Choose from 2, 10
    input_size = 784 * 2 # Color is relative ratio between 2 channels
    num_S_train = 20000 # These come from traditional trainset
    num_S_test = 10000
    num_T_train = 30000
    num_T_test = 10000 # This comes from traditional testset
    noise_type = 'Uniform' # Choose from 'Uniform' for 10, 'Gaussian' for binary
    cor_prob = 0.95 # Spurious correlation with label in Source
    model_type = '3-layer' # Choose from '3-layer', 'Linear'
    plot_mu_dist = False
    count_statistics = False
    train_S = True
    train_T = False
    pseudolabeling_flag = False
    save_pathS = 'S_classifier10_95%.pt'
    save_pathT = 'T_classifier10_95%.pt'
    S_train_images, S_train_labels, S_test_images, S_test_labels, T_train_images, T_test_images, T_test_labels = \
        construct_MNIST_dataset(noise_type=noise_type, cor_prob=cor_prob, n_way=n_way)
    model = get_model(model_type=model_type, input_size=input_size, output_size=n_way)
else:
    print('Dataset Error')

# Choose from Source_only, Mixed
if train_S:
    train_Source('Source_only')
    torch.save(model, save_pathS)
else:
    model = torch.load(save_pathS)

# Pseudolabel Target
def produce_pdolabel(T_train_images, teacher):

    img = T_train_images.view(T_train_images.size()[0], -1)
    with torch.no_grad():
        output = teacher(img)
    pdo_label = torch.argmax(output, 1)

    # Use only most confident Pseudolabels
    use_ratio = 1
    pdo_con = torch.zeros(T_train_images.size()[0])
    for i in range(T_train_images.size()[0]):
        pdo_con[i] = output[i, pdo_label[i]]
    num_T_train = int(use_ratio*T_train_images.size()[0])
    _, idx_conf = torch.topk(pdo_con, num_T_train, dim=0, sorted=False)
    images = T_train_images[idx_conf]
    pdo_label = pdo_label[idx_conf]

    return images, pdo_label

print('Testing Source Classifier :')
acc_On_Source(model)
acc_On_Target(model)
"""
# Deep copy source Classifier
S_classifier = pickle.loads(pickle.dumps(model))
"""
if plot_mu_dist:
    plot_mu(S_classifier)

# Choose from Sqrt, Exp, Entropy
if train_T:
    best_model = finetune_Target('Entropy')
    torch.save(best_model, save_pathT)
else:
    best_model = torch.load(save_pathT)

if plot_mu_dist:
    plot_mu(best_model)

# Count number of corrected examples with good distribution of activations while varying x2
if count_statistics:
    beforeT_wrong_idx = wrong_idx(S_classifier)
    afterT_wrong_idx = wrong_idx(best_model)
    turn_goods = list(set(beforeT_wrong_idx) - set(afterT_wrong_idx))
    turn_bads = list(set(afterT_wrong_idx) - set(beforeT_wrong_idx))
    keep_goods = list(set(range(num_T_test)) - set(beforeT_wrong_idx) - set(afterT_wrong_idx))
    keep_bads = list(set(beforeT_wrong_idx) - set(turn_goods))

    count1 = 0
    for idx in turn_goods:
        mean1, std1 = plot_dist(S_classifier, idx, noise_type, T_test_images[idx].view(1, -1), True)
        mean2, std2 = plot_dist(best_model, idx, noise_type, T_test_images[idx].view(1, -1), True)
        if ((mean1 > 0) and (mean2 > mean1) and (std1 > std2)):
            count1+=1
            print('success turn good')
            break

    count2 = 0
    for idx in turn_bads:
        mean1, std1 = plot_dist(S_classifier, idx, noise_type, T_test_images[idx].view(1, -1), True)
        mean2, std2 = plot_dist(best_model, idx, noise_type, T_test_images[idx].view(1, -1), True)
        if ((mean1 < 0) and (mean2 < mean1) and (std1 > std2)):
            count2+=1
            print('success turn bad')
            break

    count3 = 0
    for idx in keep_goods:
        mean1, std1 = plot_dist(S_classifier, idx, noise_type, T_test_images[idx].view(1, -1), False)
        mean2, std2 = plot_dist(best_model, idx, noise_type, T_test_images[idx].view(1, -1), False)
        if ((mean1 > 0) and (mean2 > mean1) and (std1 > std2)):
            count3+=1

    count4 = 0
    for idx in keep_bads:
        mean1, std1 = plot_dist(S_classifier, idx, noise_type, T_test_images[idx].view(1, -1), False)
        mean2, std2 = plot_dist(best_model, idx, noise_type, T_test_images[idx].view(1, -1), False)
        if ((mean1 < 0) and (mean2 < mean1) and (std1 > std2)):
            count4+=1

    print('Total number of converts', len(turn_goods))
    print('Good example number', count1)
    print('Total number of turn_bads', len(turn_bads))
    print('Good example number', count2)
    print('Total number of converts', len(keep_goods))
    print('Good example number', count3)
    print('Total number of turn_bads', len(keep_bads))
    print('Good example number', count4)


# Alternative Algorithm: Retrain target using pseudo-labels
def pseudolabeling(model):
    ori_target_acc = acc_On_Target(model)[0]
    criterion = nn.CrossEntropyLoss()
    time0 = time()
    num_rounds = 30 # number of pseudolabeling rounds, update teacher after each round
    epochs = 10 # number of batches per round to train student
    batch_size = 30000
    target_acc = []
    # Reinitialize model
    #model1 = get_model(model_type=model_type, input_size=input_size, output_size=n_way)
    model1 = pickle.loads(pickle.dumps(model))
    optimizer1 = optim.SGD(model1.parameters(), lr=0.03, momentum=0.9, weight_decay=0.002)
    #optimizer1 = optim.Adam(model1.parameters(), lr=0.03)

    for r in range(num_rounds):
        new_images, new_labels = produce_pdolabel(T_train_images, model)
        num_batches = (new_images.size()[0] + batch_size - 1) // batch_size
        for e in range(epochs):
            running_loss = 0
            for b in range(num_batches):
                end_idx = min((b+1)*batch_size, new_images.size()[0])
                images = new_images[b*batch_size : end_idx]
                labels = new_labels[b*batch_size : end_idx]

                images = images.view(images.shape[0], -1)
                optimizer1.zero_grad()
                output = model1(images)
                loss = criterion(input=output, target=labels)
                loss.backward()
                optimizer1.step()
                running_loss += loss.item()

            #print("Round {} Epoch {} - Training loss: {}".format(r, e, running_loss/num_batches))
            acc_On_Target(model1)

            target_acc.append(acc_On_Target(model1)[0])
        model = pickle.loads(pickle.dumps(model1))

    print("\nTraining Time (in minutes) =",(time()-time0)/60)

    plt.figure(1)
    #plt.subplot(211)
    plt.plot(target_acc)
    plt.hlines(ori_target_acc, 0, 20, label='Source Classifier Accuracy '+str(round(ori_target_acc, 2)))
    plt.xlabel('Total Epochs')
    plt.ylabel('Target Accuracy')
    #ind_min = np.argmin(target_acc)
    #val_min = min(target_acc)
    ind_max = np.argmax(target_acc)
    val_max = max(target_acc)
    #plt.hlines(ori_target_acc, 0, epochs)
    #plt.hlines(val_min, ind_min-10, ind_min+10, label='Min Acc '+str(val_min))
    plt.hlines(val_max, ind_max-10, ind_max+10, label='Max Acc '+str(round(val_max,2)))
    plt.legend()
    #plt.subplot(212)
    #plt.plot(vio_rec, label = 'Violation of Pseudolabels')
    #plt.ylabel('% Violation of Pseudolabels')
    #plt.xlabel('Epochs')
    #plt.legend()
    plt.show()
    #plt.savefig('6rounds_50epochs.png')

if pseudolabeling_flag:
    pseudolabeling(model)
