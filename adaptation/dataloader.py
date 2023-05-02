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
import json

class ImagenetVidRobust(Dataset):
    """ImagenetVidRObust dataset.

    """

    def __init__(self, root_dir,transform=None,class_subset=None):
        """
        Args:
            root_dir (string): directory where the imgs are
            transform (callable, optional): transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform

        # load all the dataset metadata into dictionaries

        # get the anchor file dictionary
        with open(os.path.join(root_dir,"metadata","pmsets.json")) as pmsets:
            self.pmsets_dict = json.load(pmsets)

        # get the label file dictionary
        with open(os.path.join(root_dir,"metadata","labels.json")) as labels:
            self.labels_dict = json.load(labels)

        # get the class idx to wordnet mapping dictionary
        with open(os.path.join(root_dir,"misc","imagenet_vid_class_index.json")) as imagenet_vid_class_index:
            self.imagenet_vid_class_index_dict = json.load(imagenet_vid_class_index)  
            self.imagenet_vid_wnid_map_dict = {}
            for k,v in zip(self.imagenet_vid_class_index_dict.keys(),self.imagenet_vid_class_index_dict.values()):
                self.imagenet_vid_wnid_map_dict[v[0]] = [k,v[1]] 

        # get the child to parent wornet id mapping dictionary
        with open(os.path.join(root_dir,"misc","wnid_map.json")) as wnid_map:
            self.wnid_map_dict = json.load(wnid_map)

        # get the idx to imagenet wornet id mapping dictionary
        with open(os.path.join(root_dir,"misc","imagenet_class_index.json")) as imagenet_class_index:
            self.imagenet_class_index_dict = json.load(imagenet_class_index) 


        self.labels = []
        self.vid_paths = []

        # go through the anchor dictionary, only add videos with a single class
        for key in self.pmsets_dict.keys():
            if len(self.labels_dict[key]) == 1:
                self.vid_paths.append(key)
                self.labels.append(self.labels_dict[key][0])

        # if only want a subset of classes, get a sublist of relevant samples
        if class_subset != None:
            label_tensor = torch.tensor(self.labels)
            self.dataset_idxs = []
            for c in class_subset:
                self.dataset_idxs.extend((label_tensor==c).nonzero().view(-1).tolist())
        else:
            self.dataset_idxs = list(range(len(self.labels)))


    def __getitem__(self, idx):
        # remap the idx to relevant samples in the label subset
        idx = self.dataset_idxs[idx]
        anchor_key = self.vid_paths[idx]

        # read the image sequence
        img_sequence = []
        label_sequence = []

        for frame_path in self.pmsets_dict[anchor_key]:
            img = Image.open(os.path.join(self.root_dir,frame_path))
            img = img.convert("RGB") 

            # apply transform
            if self.transform:
                img = self.transform(img)

            # get the label
            label = self.labels[idx]

            img_sequence.append(img)
            label_sequence.append(label)
            
        # return the sample (img (tensor)), object class (int)
        try:
            return torch.stack(img_sequence), torch.tensor(label_sequence)
        except:
            print(f"idx:{idx}, anchor_key:{anchor_key}")
            exit()

    def __len__(self):
        return len(self.dataset_idxs)
    
    def pred_idx_to_label_idx(self,pred_idx):
        # map imagenet index to the wordnet id
        imgnet_wnid = self.imagenet_class_index_dict[str(pred_idx)][0]
        # get the parent wordnet id
        imgnetvid_wnid = self.wnid_map_dict[imgnet_wnid]
        # map the parent wordnet id to the label index
        return int(self.imagenet_vid_wnid_map_dict[imgnetvid_wnid][0])

    # show the first image in each sequence
    def visualize_batch(self):
        batch_size = 1
        data_loader = DataLoader(self,batch_size)

        # get the first batch
        (imgs, labels) = next(iter(data_loader))
        imgs = imgs[0]
        labels = labels[0]
        
        # display the batch in a grid with the img, label, idx
        rows = 8
        cols = 8
        
        fig,ax_array = plt.subplots(rows,cols,figsize=(20,20))
        fig.subplots_adjust(hspace=0.5)
        for i in range(rows):
            for j in range(cols):
                idx = i*rows+j
                if idx == len(labels):
                    continue
                text = self.imagenet_vid_class_index_dict[str(labels[idx].item())][1]

                ax_array[i,j].imshow((imgs[idx].permute(1, 2, 0)))

                ax_array[i,j].title.set_text(text)
                ax_array[i,j].set_xticks([])
                ax_array[i,j].set_yticks([])
        plt.show()


def load_imagenetvid_robust(batch_size,class_subset=None):

    root_dir = os.path.expanduser("~/Projects/data/imagenet_vid_ytbb_robust/imagenet-vid-robust")
    val_tf = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    # load the training dataset and make the validation split
    val_set = ImagenetVidRobust(root_dir,val_tf,class_subset=class_subset)
    # num_train = int(0.98*len(train_set))
    # train_split, val_split = torch.utils.data.random_split(train_set, [num_train, len(train_set)-num_train],torch.Generator().manual_seed(42))

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=4)

    return val_loader