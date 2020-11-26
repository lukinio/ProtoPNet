import torch
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torchvision.utils import save_image

def generate_stacked_imgs(*, not_nines, nines):
    data = []
    target = []
    for i in range(0, len(not_nines), 16):
        if i+15 >= len(not_nines):
            break
        # generating image without nines
        rows = [torch.hstack((not_nines[i], not_nines[i+1], not_nines[i+2], not_nines[i+3])),
                torch.hstack((not_nines[i+4], not_nines[i+5], not_nines[i+6], not_nines[i+7])),
                torch.hstack((not_nines[i+8], not_nines[i+9], not_nines[i+10], not_nines[i+11])),
                torch.hstack((not_nines[i+12], not_nines[i+13], not_nines[i+14], not_nines[i+15]))]
        
        # permutating rows to get more images
        for row_numbers in itertools.permutations([0,1,2,3]):
            img_1 = torch.vstack((rows[row_numbers[0]], rows[row_numbers[1]], rows[row_numbers[2]], rows[row_numbers[3]]))
            
        # randomly replacing 1-4 numbers from previous image with nines
            img_2 = img_1.detach().clone()
            n_of_nines = random.randint(1, 4) # n of numbers to replace with nines
            indices = random.sample(range(16), n_of_nines)
            for idx in indices:
                row = idx//4
                col = idx%4
                nine_idx = random.randint(0, len(nines)-1)
                img_2[row*28 : (row+1)*28, col*28 : (col+1)*28] = nines[nine_idx]
            
            data += [img_1.unsqueeze(0), img_2.unsqueeze(0)]
            target += [0, 1] # label 1 when 9 is present, 0 when 9 is absent
    return torch.cat(data), torch.as_tensor(target)


def create_train_set(train_set):
    targ = train_set.targets
    nine = np.where(targ==9)
    zero_to_eight = np.where((targ==0) | (targ==1) | (targ==2) | (targ==3) | (targ==4) | (targ==5) | (targ==6) | (targ==7) | (targ==8))
    nines = train_set.data[nine]
    not_nines = train_set.data[zero_to_eight] 

    data, target = generate_stacked_imgs(not_nines=not_nines, nines=nines)
    indices = random.sample(range(162144), 60000)  # ....
    data = data[indices]                           # jeśli chcemy miec więcej danych, to mozna zakomentowac
    target = target[indices]                       # te trzy linijki
    train_set.data = data
    train_set.targets = target
    return train_set


def create_test_set(test_set):

    targ = test_set.targets
    nine = np.where(targ==9)
    zero_to_eight = np.where((targ==0) | (targ==1) | (targ==2) | (targ==3) | (targ==4) | (targ==5) | (targ==6) | (targ==7) | (targ==8))
    nines = test_set.data[nine]
    not_nines = test_set.data[zero_to_eight] 

    data, target = generate_stacked_imgs(not_nines=not_nines, nines=nines)
    indices = random.sample(range(26928), 10000)  # ....
    data = data[indices]                          # jeśli chcemy miec więcej danych, to mozna zakomentowac
    target = target[indices]                      # te dwie linijki
    test_set.data = data
    test_set.targets = target
    return test_set

def bag_generator(train_set, test_set):
    trn = create_train_set(train_set)
    tst = create_test_set(test_set)
    return trn, tst

path = "/mnt/users/lpustelnik/local/ProtoPNet/data/bagged_mnist"
def create_images(test_set):
    test_set = create_test_set(test_set)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False,
                                              num_workers=4, pin_memory=False)
    idx=0
    for bag, label in test_loader:
        save_image(bag, f"{path}/test/{label.item()}_{idx}.jpg")
        idx+=1
    

if __name__ == "__main__":
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) )
    ])
    ds_test = MNIST('data', train=False, download=True, transform=transformation)
    create_images(ds_test)