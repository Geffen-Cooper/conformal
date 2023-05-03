import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import copy
import numpy as np
import json
import skimage as sk

class ImagenetVidRobust(Dataset):
    """ImagenetVidRObust dataset.

    """

    def __init__(self, root_dir,transform=None,pert_trans=None,class_subset=None):
        """
        Args:
            root_dir (string): directory where the imgs are
            transform (callable, optional): transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        self.pert_trans = pert_trans

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
            if len(self.labels_dict[key]) == 1 and len(self.pmsets_dict[key]) >= 5:
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

        for i,frame_path in enumerate(self.pmsets_dict[anchor_key]):
            img = Image.open(os.path.join(self.root_dir,frame_path))
            img = img.convert("RGB") 

            # apply transform
            if self.pert_trans and i >= len(self.pmsets_dict[anchor_key])//2:
                img = self.pert_trans(img)
            elif self.transform:
                img = self.transform(img)

            # get the label
            label = self.labels[idx]

            img_sequence.append(img)
            label_sequence.append(label)
            
        # return the sample (img (tensor)), object class (int)
        try:
            return torch.stack(img_sequence), torch.tensor(label_sequence),anchor_key
        except:
            print(f"idx:{idx}, anchor_key:{anchor_key}")
            exit()

    def __len__(self):
        return len(self.dataset_idxs)
    
    def pred_idx_to_label_idx(self,pred_idx):
        # map imagenet index to the wordnet id
        imgnet_wnid = self.imagenet_class_index_dict[str(pred_idx)][0]
        # get the parent wordnet id
        try:
            imgnetvid_wnid = self.wnid_map_dict[imgnet_wnid]
        except:
            # if no parent wnid try mapping to class label
            try:
                label = int(self.imagenet_vid_wnid_map_dict[imgnetvid_wnid][0])
            except:
                return pred_idx
            return label
        # map the parent wordnet id to the label index
        try:
            label = int(self.imagenet_vid_wnid_map_dict[imgnetvid_wnid][0])
        except:
            return pred_idx
        return label


    # show the first image in each sequence
    def visualize_vid(self,student,teacher,big_teacher,batch_idx=1):
        batch_size = 1
        data_loader = DataLoader(self,batch_size)

        # get the first batch
        for count,(imgs, labels,anchor_key) in enumerate(data_loader):
            if count == batch_idx:
                break
        imgs = imgs[0]
        labels = labels[0]
        
        preds = []
        track_dict_student = {}
        track_dict_teacher = {}
        track_dict_teacher_big = {}
        track_dicts = [track_dict_student, track_dict_teacher, track_dict_teacher_big]
        models = [student,teacher,big_teacher]

        stud_sms = torch.zeros((len(imgs),1000))
        teach_sms = torch.zeros((len(imgs),1000))
        big_teacher_sms = torch.zeros((len(imgs),1000))

        sm_scores = [stud_sms,teach_sms,big_teacher_sms]

        for id,track_dict in enumerate(track_dicts):
            # get a prediction for the current video
            with torch.no_grad():
                sm_scores[id][:,:] = torch.nn.functional.softmax(models[id](imgs).cpu(),dim=1)[:,:]


        # get the top five averaged over the whole video
        for id,track_dict in enumerate(track_dicts):
            # get the top5 on avg
            vals,idxs = torch.topk(sm_scores[id].mean(dim=0),5)
            # create the tracking stats
            for j,idx in enumerate(idxs):
                track_dict[idx.item()] = np.zeros(len(imgs))
            # fill in the conf scores
            for i, img in enumerate(imgs):
                for j,idx in enumerate(track_dict.keys()):
                    track_dict[idx][i] = sm_scores[id][i,idx]

        plt.close()
        fig,ax = plt.subplot_mosaic([['shufflenet_v2_x0_5','space','image'],
                                    ['efficientnet_b0','space','image'],
                                    ['resnet_50','space','image'],],figsize=(12,6),
                                    gridspec_kw={'width_ratios':[0.5,0.1,0.5],
                                                'height_ratios': [1,1,1]})
        fig.tight_layout(pad=2.0)
        ax['space'].axis('off')
        names = ["shufflenet_v2_x0_5","efficientnet_b0","resnet_50"]
        flops = ["40","400","4000"]

        def animate(i):
            ax['image'].clear()
            id = 0
            for name,track_dict in zip(names,track_dicts):
                ax[name].clear()
                
                for k in track_dict.keys():
                    imgnet_name = self.imagenet_class_index_dict[str(k)][1]
                    idx = self.pred_idx_to_label_idx(k)
                    try:
                        pred_label = imgnet_name + " ("+self.imagenet_vid_class_index_dict[str(idx)][1]+")"
                    except:
                        pred_label = imgnet_name
                    ax[name].plot(np.arange(i+1),track_dict[k][:i+1],label=pred_label)
                    ax[name].set_xticks(np.arange(len(track_dict[k])))
                ax[name].set_xlim([0,len(imgs)-1])
                ax[name].grid()
                # ax[name].set_ylim([0,1])
                ax[name].set_title(name+" ("+flops[id]+" MFLOPS)")
                ax[name].set_ylabel('confidence (%)')
                if id == len(names)-1:
                    ax[name].set_xlabel('frame #')
                ax[name].legend(loc="upper left",bbox_to_anchor=(1, 1))
                id += 1
                ax[name].set_yscale('log')
            if i >= len(imgs)//2:
                ax['image'].imshow(
                    self.pert_trans(Image.open(os.path.join(self.root_dir,self.pmsets_dict[anchor_key[0]][i])))
                    )
            else:
                ax['image'].imshow(Image.open(os.path.join(self.root_dir,self.pmsets_dict[anchor_key[0]][i])))
            

        # run the animation
        ani = FuncAnimation(fig, animate, frames=len(imgs), interval=300, repeat=False)
        ani.save('ani.gif')
        plt.show()


def load_imagenetvid_robust(batch_size,class_subset=None):

    root_dir = os.path.expanduser("~/Projects/data/imagenet_vid_ytbb_robust/imagenet-vid-robust")
    pert_tf = None
    # pert_tf = transforms.Compose([
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             Brightness(4),
    #             transforms.PILToTensor(),
    #             transforms.ConvertImageDtype(torch.float),
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_tf = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    # load the training dataset and make the validation split
    val_set = ImagenetVidRobust(root_dir,val_tf,pert_tf,class_subset=class_subset)
    # num_train = int(0.98*len(train_set))
    # train_split, val_split = torch.utils.data.random_split(train_set, [num_train, len(train_set)-num_train],torch.Generator().manual_seed(42))

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True,num_workers=4)

    return val_loader


# ================== dist shifts ================
class Brightness(object):

    def __init__(self, severity):
        self.severity = severity

    def __call__(self, x):
        c = [.1, .2, .3, .4, .5][self.severity - 1]

        x = np.array(x) / 255.
        x = sk.color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
        x = sk.color.hsv2rgb(x)

        return Image.fromarray(np.uint8(np.clip(x, 0, 1) * 255))