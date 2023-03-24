import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import copy
import numpy as np

class TinyImagenet(Dataset):
    """Tiny Imagenet dataset.

       -Contains 200 classes
       -500 training, 50 validation, 50 test per class (100,000 total)
       -images downsized to 64x64
    """

    def __init__(self, root_dir,train=True,transform=None):
        """
        Args:
            root_dir (string): directory where tinyimagenet unzipped
            train (bool): if False then load validation set (used as test set)
            transform (callable, optional): transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform

        # get the class ids as a list in sorted order
        with open(os.path.join(self.root_dir,"wnids.txt")) as f:
            self.class_id_list = sorted(f.read().splitlines())

        # map from class id to clas idx, e.g. n0144... --> 0, the idx is the label
        self.class_idx_map = {self.class_id_list[i] : i for i in range(0, len(self.class_id_list))}
        
        # get the mapping from class ids to readable names, e.g. n0144... --> 'goldfish'
        df = pd.read_csv("mapping.txt",sep=":",header=None)
        df[1] = df[1].apply(lambda row: row.split(',')[0].replace("'",""))
        self.class_name_map = dict(zip(df[0], df[1]))

        if train:
            train_dir = os.path.join(root_dir,"train")

            self.labels = []
            self.img_paths = []

            # go through directory tree to get all img file paths
            i=0
            for class_dir in sorted(os.listdir(train_dir)):
                # if i == 10:
                #     break
                for f in os.listdir(os.path.join(train_dir,class_dir,"images")):
                    self.img_paths.append(os.path.join(train_dir,class_dir,"images",f))
                    self.labels.append(self.class_idx_map[class_dir])
                i+=1
        
        # use validation as test set
        if not train:
            val_dir = os.path.join(root_dir,"val")

            # get the validation img paths and corresponding labels
            df = pd.read_csv(os.path.join(val_dir,"val_annotations.txt"),sep="\t",header=None)
            df[0] = df[0].apply(lambda row: os.path.join(val_dir,"images",row))
            df[1] = df[1].apply(lambda row: self.class_idx_map[row])
            self.img_paths = df[0].to_list()
            self.labels = df[1].to_list()



    def __getitem__(self, idx):
        # read the image
        img = Image.open(self.img_paths[idx])
        img = img.convert("RGB") 

        # apply transform
        if self.transform:
            img = self.transform(img)

        # get the label
        label = self.labels[idx]
            
        # return the sample (img (tensor)), object class (int)
        return img, label

    def __len__(self):
        return len(self.img_paths)

    def visualize_batch(self):
        batch_size = 64
        data_loader = DataLoader(self,batch_size)

        # get the first batch
        (imgs, labels) = next(iter(data_loader))
        
        # display the batch in a grid with the img, label, idx
        rows = 8
        cols = 8
        
        fig,ax_array = plt.subplots(rows,cols,figsize=(20,20))
        fig.subplots_adjust(hspace=0.5)
        for i in range(rows):
            for j in range(cols):
                idx = i*rows+j
                text = self.class_name_map[self.class_id_list[labels[idx]]]

                ax_array[i,j].imshow((imgs[idx].permute(1, 2, 0)))

                ax_array[i,j].title.set_text(text)
                ax_array[i,j].set_xticks([])
                ax_array[i,j].set_yticks([])
        plt.show()


def load_tiny_imagenet(batch_size,rand_seed):
    
    root_dir = os.path.expanduser("~/Projects/data/tiny-imagenet-200")
    ts = transforms.Compose([
                transforms.PILToTensor(),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    # load the datasets
    train_set = TinyImagenet(root_dir,True,ts)
    test_set = TinyImagenet(root_dir,False,ts)

    # split training and validation)
    g_cpu = torch.Generator()
    g_cpu.manual_seed(rand_seed)
    train_split, val_split = torch.utils.data.random_split(train_set, [90000, 10000],generator=g_cpu)

    # create the data loaders
    train_loader = torch.utils.data.DataLoader(train_split, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=4,generator=g_cpu)
    val_loader = torch.utils.data.DataLoader(val_split, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=4,generator=g_cpu)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=4,generator=g_cpu)

    return (train_loader, val_loader, test_loader)





class Imagenet(Dataset):
    """Imagenet dataset.

    """

    def __init__(self, root_dir,transform=None,class_subset=None):
        """
        Args:
            root_dir (string): directory where the imgs are
            transform (callable, optional): transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform

        # get the class ids as a list in sorted order
        self.class_id_list = sorted(os.listdir(self.root_dir))

        # map from class id to clas idx, e.g. n0144... --> 0, the idx is the label
        self.class_idx_map = {self.class_id_list[i] : i for i in range(0, len(self.class_id_list))}
        
        # get the mapping from class ids to readable names, e.g. n0144... --> 'goldfish'
        df = pd.read_csv(os.path.join(root_dir,"../mapping.txt"),sep=":",header=None)
        df[1] = df[1].apply(lambda row: row.split(',')[0].replace("'",""))
        self.class_name_map = dict(zip(df[0], df[1]))


        self.labels = []
        self.img_paths = []

        # go through directory tree to get all img file paths
        for class_dir in sorted(os.listdir(root_dir)):
            for f in os.listdir(os.path.join(root_dir,class_dir)):
                self.img_paths.append(os.path.join(root_dir,class_dir,f))
                self.labels.append(self.class_idx_map[class_dir])

        if class_subset != None:
            label_tensor = torch.tensor(self.labels)
            self.dataset_idxs = []
            for c in class_subset:
                self.dataset_idxs.extend((label_tensor==c).nonzero().view(-1).tolist())
        else:
            self.dataset_idxs = list(range(len(self.labels)))


    def __getitem__(self, idx):
        # remap the idx
        idx = self.dataset_idxs[idx]

        # read the image
        img = Image.open(self.img_paths[idx])
        img = img.convert("RGB") 

        # apply transform
        if self.transform:
            img = self.transform(img)

        # get the label
        label = self.labels[idx]
            
        # return the sample (img (tensor)), object class (int)
        return img, label

    def __len__(self):
        return len(self.dataset_idxs)

    def visualize_batch(self):
        batch_size = 64
        data_loader = DataLoader(self,batch_size)

        # get the first batch
        (imgs, labels) = next(iter(data_loader))
        
        # display the batch in a grid with the img, label, idx
        rows = 8
        cols = 8
        
        fig,ax_array = plt.subplots(rows,cols,figsize=(20,20))
        fig.subplots_adjust(hspace=0.5)
        for i in range(rows):
            for j in range(cols):
                idx = i*rows+j
                text = self.class_name_map[self.class_id_list[labels[idx]]]

                ax_array[i,j].imshow((imgs[idx].permute(1, 2, 0)))

                ax_array[i,j].title.set_text(text)
                ax_array[i,j].set_xticks([])
                ax_array[i,j].set_yticks([])
        plt.show()


def load_imagenet(batch_size,rand_seed,train=True,class_subset=None,resize=224,norm=True):

    if train:
        root_dir = os.path.expanduser("~/Projects/data/imagenet/train")
        train_tf = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.Resize(resize),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

        # load the training dataset and make the validation split
        train_set = Imagenet(root_dir,train_tf,class_subset=class_subset)
        num_train = int(0.9*len(train_set))
        train_split, val_split = torch.utils.data.random_split(train_set, [num_train, len(train_set)-num_train],torch.Generator().manual_seed(42))

        train_loader = torch.utils.data.DataLoader(train_split, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_split, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=4)

        return train_loader, val_loader

    else:
        root_dir = os.path.expanduser("~/Projects/data/imagenet/val")
        if norm:
            test_tf = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.Resize(resize),
                        transforms.PILToTensor(),
                        transforms.ConvertImageDtype(torch.float),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
        else:
            print("no norm")
            test_tf = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.Resize(resize),
                        transforms.PILToTensor(),
                        transforms.ConvertImageDtype(torch.float)
                    ])

        # load the test set
        test_set = Imagenet(root_dir,test_tf,class_subset=class_subset)

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, pin_memory=True)

        return test_loader


def load_imagenet64(batch_size,rand_seed,train=True,class_subset=None):

    if train:
        root_dir = os.path.expanduser("~/Projects/data/imagenet64/train")
        train_tf = transforms.Compose([
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

        # load the training dataset and make the validation split
        train_set = Imagenet(root_dir,train_tf,class_subset=class_subset)
        num_train = int(0.9*len(train_set))
        train_split, val_split = torch.utils.data.random_split(train_set, [num_train, len(train_set)-num_train],torch.Generator().manual_seed(42))

        train_loader = torch.utils.data.DataLoader(train_split, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_split, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=4)

        return train_loader, val_loader

    else:
        root_dir = os.path.expanduser("~/Projects/data/imagenet64/val")
        test_tf = transforms.Compose([
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

        # load the test set
        test_set = Imagenet(root_dir,test_tf,class_subset=class_subset)

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, pin_memory=True)

        return test_loader


def load_imagenetc_val(batch_size,rand_seed,corruption="gaussian_noise",severity=1,class_subset=None,shuffle=False,resize=224,sampler=None):
    
    root_dir = os.path.join(os.path.expanduser("~/Projects/data/imagenetc_val"),corruption,str(severity))
    ts = transforms.Compose([
                transforms.Resize(resize),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    # load the dataset
    val_set = Imagenet(root_dir,ts,class_subset)

    # create the data loader
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, sampler=sampler,pin_memory=True,num_workers=4)

    return val_loader

def load_imagenetc64_val(batch_size,rand_seed,corruption="gaussian_noise",severity=1,class_subset=None,shuffle=False,sampler=None):
    
    root_dir = os.path.join(os.path.expanduser("~/Projects/data/ImageNet-64x64-C"),corruption,str(severity))
    ts = transforms.Compose([
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    # load the dataset
    val_set = Imagenet(root_dir,ts,class_subset)

    # create the data loader
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, sampler=sampler,pin_memory=True,num_workers=4)

    return val_loader

def visualize_batch(data_loader):

    # get the first batch (we only grab the first 64 images if batch > 64)
    (imgs, labels) = next(iter(data_loader))
    
    imgs,labels = imgs.to("cpu"), labels.to("cpu")

    input_tensor = copy.deepcopy(imgs).cpu()

    # undo imagenet normalization
    for it in input_tensor:
        it[0][:][:]*=(.229)
        it[0][:][:]+=(0.485)
        it[1][:][:]*=(.224)
        it[1][:][:]+=(0.455)
        it[2][:][:]*=(.225)
        it[2][:][:]+=(0.405)
        it = ((it-torch.min(it))/torch.max(it-torch.min(it)))

    # display the batch in a grid with the img, label, idx
    rows = 8
    cols = 8
    
    fig,ax_array = plt.subplots(rows,cols,figsize=(20,20))
    fig.subplots_adjust(hspace=0.5)
    num_imgs = 0
    for i in range(rows):
        for j in range(cols):
            if num_imgs == len(imgs):
                break

            idx = i*rows+j

            # create text labels
            try:
                text = data_loader.dataset.dataset.class_name_map[data_loader.dataset.dataset.class_id_list[labels[idx]]]
            except:
                text = data_loader.dataset.class_name_map[data_loader.dataset.class_id_list[labels[idx]]]

            ax_array[i,j].imshow((input_tensor[idx].permute(1, 2, 0).numpy()*255).astype(np.uint8))

            ax_array[i,j].set_title(text,color="black")
            ax_array[i,j].set_xticks([])
            ax_array[i,j].set_yticks([])
            num_imgs += 1
        if num_imgs == len(imgs):
                break
    plt.show()