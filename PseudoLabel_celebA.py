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
from util import save_ckp, load_ckp
from model import get_model, weights_init
from process_data import celebADataset
import argparse

torch.manual_seed(46)
np.random.seed(4)

# Test Loss
def test_celebA(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        running_loss = 0
        acc = 0
        for b, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(input=logits, target=labels)
            acc += torch.mean((torch.argmax(logits, 1)==labels).float())
            running_loss += loss.item()
    return running_loss / len(test_loader), acc.item() / len(test_loader)

# Train on Source
def train_Source_celebA(checkpoint_path, best_model_path, target_w, start_epochs, n_epochs, T_test_loss_min_input, S_train_loader, T_train_loader, S_test_loader, T_test_loader):
    criterion = nn.CrossEntropyLoss()
    time0 = time()
    print_iter = 1000
    target_batch_num = 4
    S_loss_rec = []
    T_loss_rec = []
    S_acc_rec = []
    T_acc_rec = []
    train_loss_rec = []
     # initialize tracker for minimum validation loss
    T_test_loss_min = T_test_loss_min_input
    for e in range(start_epochs, n_epochs+1):
        running_loss = 0
        running_T_ent = 0
        for b, (images, labels) in enumerate(S_train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(input=logits, target=labels)

            if target_w > 0:
                T_images = []
                for j in range(target_batch_num):
                    T_images_j, _, = next(iter(T_train_loader))
                    T_images.append(T_images_j)
                T_images = torch.cat(T_images, dim=0).to(device)

                logits_T = model(T_images)
                loss_T_ent = torch.mean(torch.sum(- F.log_softmax(logits_T, 1) * F.softmax(logits_T, 1), 1), 0)
                loss += loss_T_ent * target_w

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (target_w>0):
                running_T_ent += loss_T_ent.item()

            if b % print_iter == print_iter-1:
                print("Epoch {} Iter {} - tot Training loss: {:.4f}".format(e, b, loss.item()))
                if target_w > 0:
                    print("Target ent loss: {:.4f}".format(loss_T_ent.item() * target_w))

        train_loss = running_loss / len(S_train_loader)
        running_T_ent /= len(S_train_loader)
        train_loss_rec.append(train_loss)
        print("Epoch {} - Training loss: {:.4f}".format(e, train_loss))
        S_test_loss, S_test_acc = test_celebA(model, S_test_loader)
        print("S Test loss: {:.4f}".format(S_test_loss))
        print("S Test Acc: {:.4f}".format(S_test_acc))
        T_test_loss, T_test_acc = test_celebA(model, T_test_loader)
        print("T Test loss: {:.4f}".format(T_test_loss))
        print("T Test Acc: {:.4f}".format(T_test_acc))

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': e+1,
            'T_test_loss_min': T_test_loss_min,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        ## Save the model if validation loss has decreased
        if T_test_loss < T_test_loss_min:
            print('T_test_loss decreased ({:.4f} --> {:.4f}).  Saving model ...'.format(T_test_loss_min,T_test_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            T_test_loss_min = T_test_loss

        S_loss_rec.append(S_test_loss)
        T_loss_rec.append(T_test_loss)
        S_acc_rec.append(S_test_acc)
        T_acc_rec.append(T_test_acc)
    print("\nTraining Time (in minutes) =",(time()-time0)/60)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(S_loss_rec, label = 'Source Test')
    plt.plot(T_loss_rec, label = 'Target Test')
    plt.plot(train_loss_rec, label = 'Source Train')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.subplot(212)
    plt.plot(S_acc_rec, label = 'Source Test')
    plt.plot(T_acc_rec, label = 'Target Test')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
    plt.savefig('train_Source.png')

#======= Main Algorithm Below ======
parser = argparse.ArgumentParser()
parser.add_argument("--resume", dest='resume', action='store_true', help="flag to train from existing model")
parser.add_argument("--target_w", default=0, help="target entropy loss weight")
parser.add_argument("--n_epoch", default=300, help="target total training epochs")
args = parser.parse_args()

ngpu = 1
dataset = 'celebA'
new_train = not(args.resume)
print(new_train)
target_w = float(args.target_w)
n_epoch = int(args.n_epoch)

checkpoint_get_path = './hcb/checkpoints/Slabeled_checkpoint.pt'
if (target_w > 0):
    checkpoint_save_path = './hcb/checkpoints/Slabeled_Tent_checkpoint.pt'
else:
    checkpoint_save_path = './hcb/checkpoints/Slabeled_checkpoint.pt'
bestmodel_save_path = './hcb/checkpoints/best.pt'
model_type = 'svhnCNN'
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

n_way = 2
input_size = 3 * 32 * 32
save_path1 = './hcb/dataLoaders/S_train_loader.pt'
save_path2 = './hcb/dataLoaders/S_test_loader.pt'
save_path3 = './hcb/dataLoaders/T_train_loader.pt'
save_path4 = './hcb/dataLoaders/T_test_loader.pt'
time0 = time()
S_train_loader = torch.load(save_path1)
S_test_loader = torch.load(save_path2)
T_train_loader = torch.load(save_path3)
T_test_loader = torch.load(save_path4)
print("Data loading time (in minutes) =",(time()-time0)/60)

model = get_model(model_type=model_type, input_size=input_size, output_size=n_way).to(device)
model.apply(weights_init)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.002)

if new_train:
    start_epoch = 0
    T_test_loss_min = 999
else:
    # load the saved checkpoint
    print('load')
    model, optimizer, start_epoch, T_test_loss_min = load_ckp(checkpoint_get_path, model, optimizer)

train_Source_celebA(checkpoint_save_path, bestmodel_save_path, target_w, start_epoch, n_epoch, T_test_loss_min, S_train_loader, T_train_loader, S_test_loader, T_test_loader)
