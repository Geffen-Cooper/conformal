import numpy as np
import torch
from torchvision import models,transforms
import cv2
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the class for single layer NN
class FC(torch.nn.Module):    
    # Constructor
    def __init__(self, input_size, output_size):
        super(FC, self).__init__()
        # hidden layer 
        self.fc = torch.nn.Linear(input_size, output_size)
    # prediction function
    def forward(self, x):
        return self.fc(x)

# get the label string from the label idx
df = pd.read_csv("mapping.txt",sep=":",header=None)
df[1] = df[1].apply(lambda row: row.split(',')[0].replace("'",""))
class_name_map = dict(zip(range(1000), df[1]))
def get_label(idx):
    return class_name_map[idx][1:]

# create the models
teacher = models.efficientnet_b0(weights='DEFAULT').to('cpu').eval()
student = models.shufflenet_v2_x0_5(weights='DEFAULT').to('cpu').eval()

student = models.shufflenet_v2_x0_5(weights='DEFAULT').to('cpu')
with torch.no_grad():
    # first reinitialize the layer before classification to match the teacher feature dimensions
    student.conv5[0] = torch.nn.Conv2d(student.conv5[0].in_channels,teacher.classifier[1].in_features,kernel_size=(1, 1), stride=(1, 1), bias=False)
    student.conv5[1] = torch.nn.BatchNorm2d(teacher.classifier[1].in_features,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)

    # next create a new fc layer to match the teacher dimension
    student.fc = torch.nn.Linear(teacher.classifier[1].in_features,teacher.classifier[1].out_features)

    # freeze the classification layer
    for param in student.fc.parameters():
        param.requires_grad = False
student.load_state_dict(torch.load("best_batch_i288264sn_frozen_head_long1682578354.1065493.pth",map_location=torch.device('cpu'))['model_state_dict'])
student.to('cpu')
student.eval()
models = [student,teacher]

# create the shared classification head
with torch.no_grad():
    shared_FC = FC(student.fc.in_features,student.fc.out_features)
    shared_FC.fc.weight[:,:] = student.fc.weight[:,:]
    shared_FC.fc.bias[:] = student.fc.bias[:]

# create the preprocessing transform
test_tf = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])


# create the stats structures
track_dict_student = {}
track_dict_teacher = {}
track_dicts = [track_dict_student, track_dict_teacher]


window_len = 20
stud_embds = torch.zeros((window_len,student.fc.in_features))
teach_embds = torch.zeros((window_len,teacher.classifier[1].in_features))
embds = [stud_embds,teach_embds]

stud_sms = torch.zeros((window_len,student.fc.out_features))
teach_sms = torch.zeros((window_len,teacher.classifier[1].out_features))

sm_scores = [stud_sms,teach_sms]

# setup embedding extraction
emb_d = {}
def get_emb(name):
    def hook(model, input, output):
        emb_d[name] = input[0].detach()
    return hook

student.fc.register_forward_hook(get_emb('emb'))
teacher.classifier[1].register_forward_hook(get_emb('emb'))



# ================================= Helper Functions =============================

# initialize the capture object
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # this is the magic!

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
FRAME_W = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
FRAME_H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# measure FPS
last_frame_time = 0
curr_frame_time = 0
frame_count = 0
fps = 0

# get the next camera frame
def read_camera():
    global frame_count
    global last_frame_time
    global curr_frame_time
    global fps
    ret,frame = cap.read()
    if not ret:
        raise RuntimeError("failed to read frame")
    
    # calculate the fps
    frame_count += 1
    curr_frame_time = time.time()
    diff = curr_frame_time - last_frame_time
    if diff > 1:
        fps = "FPS: " + str(round(frame_count/(diff),2))
        last_frame_time = curr_frame_time
        frame_count = 0
    
    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB), fps

# get the prediction and embedding
def process_frame(frame, model, offset=None, shared_FC=None, emb=None):
    # preprocess and forward pass through the model
    with torch.no_grad():
        frame = transforms.ToTensor()(frame)
        frame = test_tf(frame).unsqueeze(0)#+noise
        if offset is None:
            sm_score = torch.nn.functional.softmax(model(frame).cpu().view(-1),dim=0)
        else:
            sm_score = torch.nn.functional.softmax(model(frame).cpu().view(-1),dim=0)
            sm_score = torch.nn.functional.softmax(shared_FC(emb['emb']+offset).cpu().view(-1),dim=0)

    if emb is not None:
        return sm_score, emb['emb']




# ============================ Live Feed ==================================

# create the plot
fig,ax = plt.subplot_mosaic([['shufflenet_v2_x0_5','space','image'],
                              ['efficientnet_b0','space','image']],figsize=(12,6),
                              gridspec_kw={'width_ratios':[0.5,0.1,0.5],
                                           'height_ratios': [1,1]})
fig.tight_layout(pad=2.0)
ax['space'].axis('off')
names = ["shufflenet_v2_x0_5","efficientnet_b0"]
flops = ["40","400"]

for name in names:
    ax[name].tick_params(axis='both', which='major', labelsize=6)
    
    if names.index(name) == len(names)-1:
        ax[name].set_xlabel('frame #')
    
offset = torch.zeros(embds[0][0].shape)
def animate(i):
    global offset
    # clear the last image
    ax['image'].clear()
    
    # get the next frame
    frame, fps = read_camera()
    disp_frame = frame.copy()
    print(i,fps)

    # process the frame for each model
    id = 0
    for name,track_dict in zip(names,track_dicts):
        # if name == "shufflenet_v2_x0_5":
        #     continue
        # clear the plot
        ax[name].clear()
        
        # process the frame
        if i % 60 == 0 and i > 0: # every tenth frame, calculate the offset
            offset = embds[1][-1] - embds[0][-1]
        
        if id == 0:
            sm_score, embd = process_frame(frame,models[id],offset,shared_FC,emb_d)
        else:
            sm_score, embd = process_frame(frame,models[id],emb=emb_d)
        # sm_score, embd = process_frame(frame,models[id],emb=emb_d)

        # update the stats
        if i >= window_len:
            sm_scores[id] = torch.roll(sm_scores[id],-1,0)
            embds[id] = torch.roll(embds[id],-1,0)
            # sm_scores[id][:-1] = sm_scores[id][1:]
            sm_scores[id][-1] = sm_score
            embds[id][-1] = embd
        else:
            sm_scores[id][i] = sm_score
            embds[id][i] = embd

        # get the top5 on avg
        vals,idxs = torch.topk(sm_scores[id][-3:,:].mean(dim=0),5)

        # create the tracking stats
        for j,idx in enumerate(idxs):
            track_dict[idx.item()] = np.zeros(window_len)

        # fill in the conf scores (only for top 5)
        for j,idx in enumerate(idxs):
            track_dict[idx.item()][:] = sm_scores[id][:,idx]
        
        for k in idxs:
            k = k.item()
            if i < window_len:
                ax[name].plot(np.arange(i+1),track_dict[k][:i+1],label=get_label(k))
                # print(track_dict[k][:i+1])
                ax[name].set_xticks(np.arange(len(track_dict[k])))
            else:
                ax[name].plot(np.arange(i-20,i),track_dict[k][:window_len],label=get_label(k))
                ax[name].set_xticks(np.arange(i-20,i))
        
        if i < window_len:
            ax[name].set_xlim([0,window_len-1])
        else:
            ax[name].set_xlim([i-20,i])
        
        ax[names[0]].axvline(i-i%60,c='k')
        ax[name].set_title(name+" ("+flops[names.index(name)]+" MFLOPS)")
        ax[name].set_ylabel('confidence (%)')
        if id == 1:
            ax[name].set_xlabel('frame #')
        # ax[name].set_ylim([0,1])
        
        ax[name].legend(loc="upper left",bbox_to_anchor=(1, 1))
        ax[name].grid()
        # ax[name].set_yscale('log')
        id += 1
        
    ax['image'].imshow(disp_frame)
    

# run the animation
ani = FuncAnimation(fig, animate, interval=300)
plt.show()