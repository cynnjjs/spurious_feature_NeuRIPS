import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os

torch.manual_seed(42)
np.random.seed(0)
image_size = 32
transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                              ])

class celebADataset(Dataset):
    def __init__(self, Snum, Tnum, X2_attr, Y_attr, data_folder, source):

        self.data = []
        if source:
            self.target = torch.zeros(np.sum(Snum))
        else:
            self.target = torch.zeros(np.sum(Tnum))
        filepath = 'list_attr_celeba.txt'
        d = datasets.ImageFolder(root=data_folder, transform=transform)

        with open(filepath, 'r') as f:
            f.readline()
            attribute_names = f.readline().strip().split(' ')
            idx_y = attribute_names.index(Y_attr)
            idx_x2 = attribute_names.index(X2_attr)
            sc = [[0, 0], [0, 0]]
            tc = [[0, 0], [0, 0]]
            for i, line in enumerate(f):
                if (np.sum(sc)+np.sum(tc) >= np.sum(Snum) + np.sum(Tnum)):
                    break
                fields = line.strip().replace('  ', ' ').split(' ')
                attr_vec = np.array([int(x) for x in fields[1:]])
                x2 = int((attr_vec[idx_x2]+1)/2)
                y = int((attr_vec[idx_y]+1)/2)
                if sc[x2][y] < Snum[x2][y]:
                    if source:
                        self.data.append(d[i][0])
                        self.target[np.sum(sc)] = y
                    sc[x2][y] += 1
                elif tc[x2][y] < Tnum[x2][y]:
                    if source == False:
                        self.data.append(d[i][0])
                        self.target[np.sum(tc)] = y
                    tc[x2][y] += 1

            print('i, sc, tc:', i, sc, tc)

            self.data = torch.stack(self.data)
            self.target = self.target.long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

def split_dataset(dataset, batch_size, shuffle, validation_split):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    return train_loader, validation_loader

def construct_CelebA_dataset(num_S_tot, num_T_tot, X2_attr, Y_attr, data_path):
    split = .1
    batch_size = 64

    S_dataset = celebADataset(num_S_tot, num_T_tot, X2_attr, Y_attr, data_path, source=True)
    T_dataset = celebADataset(num_S_tot, num_T_tot, X2_attr, Y_attr, data_path, source=False)
    S_train_loader, S_test_loader = split_dataset(S_dataset, batch_size=batch_size, shuffle=True, validation_split=split)
    T_train_loader, T_test_loader = split_dataset(T_dataset, batch_size=batch_size, shuffle=True, validation_split=split)

    return S_train_loader, S_test_loader, T_train_loader, T_test_loader

if __name__ == "__main__":
    data_path = '../../../../scr-ssd/datasets/'

    # Parameters for hcb/
    # Dark Female 00, Dark Male 01, Blond Female 10, Blond Male 11
    num_S_tot = np.asarray([[1749, 0], [0, 1250]])
    num_T_tot = np.asarray([[25658, 23591], [8055, 499]])
    S_train_loader, S_test_loader, T_train_loader, T_test_loader = construct_CelebA_dataset(num_S_tot, num_T_tot, X2_attr='Blond_Hair', Y_attr='Male', data_path=data_path)
    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Save DataLoaders
    save_path1 = './hcb/dataLoaders/S_train_loader.pt'
    save_path2 = './hcb/dataLoaders/S_test_loader.pt'
    save_path3 = './hcb/dataLoaders/T_train_loader.pt'
    save_path4 = './hcb/dataLoaders/T_test_loader.pt'

    torch.save(S_train_loader, save_path1)
    torch.save(S_test_loader, save_path2)
    torch.save(T_train_loader, save_path3)
    torch.save(T_test_loader, save_path4)
    
    # Plot some training images
    T_train_loader = torch.load(save_path3)
    real_batch = next(iter(T_train_loader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Target Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig('T_train_loader.png')
    print(real_batch[1].to(device)[:64])
