# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 


# %%
df = pd.read_csv('./location_and_label_rmNan.csv',encoding='CP949')
df


# %%
val = df.loc[:,["위험정도","레이블영상"]].values
danger,labPaths = val[:,0],val[:,1]
danger[:10],labPaths[:10]
dir = "../targetData3"
rgbs = [os.path.join(dir,"R"+lab[3:]+'.png') for lab in labPaths]
thrs = [os.path.join(dir,"T"+lab[3:]+'.png') for lab in labPaths]
labs = [os.path.join(dir,lab+'.png') for lab in labPaths]
rgbs[:10],thrs[:10],labs[:10]
i = 0
while i < len(rgbs):
    if not os.path.exists(rgbs[i]) or not os.path.exists(thrs[i]) or not os.path.exists(labs[i]):
        print()
        rgbs.pop(i)
        thrs.pop(i)
        labs.pop(i)
        danger = np.delete(danger,[i])
    i+=1

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms
from PIL import Image
import cv2 

imgtrans = transforms.Compose([
    transforms.Resize(size=(128,256)),
    transforms.ToTensor()
])

class CompleteRgbThrDataset(torch.utils.data.Dataset):
    def __init__(self,rgbList,thrList,labList,label,imgtrans = imgtrans):
        self.rgbList = rgbList
        self.thrList = thrList
        self.labList = labList
        self.imgtrans = imgtrans
        self.label = label 
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self,idx):
        rgb = Image.open(self.rgbList[idx]).convert('RGB')
        thr = Image.open(self.thrList[idx]).convert('RGB')
        # lab = Image.fromarray((np.array(Image.open(self.labList[idx]).convert('L'))>=127.0).astype(np.uint8)*255)
        lab = Image.open(self.labList[idx]).convert('L')
        label = torch.LongTensor([self.label[idx]])
        if self.imgtrans:
            rgb = self.imgtrans(rgb)
            thr = self.imgtrans(thr)
            lab = self.imgtrans(lab)
        lab = (lab>=0.5).type(torch.float32)
        
        return rgb,thr,lab,label 



# %%
from models.MBConvNet import MBConvIntertwinedUNet
from models.FuseNet import FuseNet
from models.MFNet import MFNet
from models.UNet import UNet
from models.SegNet import SegNet
from models.RTFNet import RTFNet 
fuseNet = FuseNet(1)
mfNet = MFNet(1)
uNet = UNet(6,1)
segNet = SegNet(1)
rtfNet = RTFNet(1)
mbConvNet = MBConvIntertwinedUNet(1)

state_dict = torch.load(os.path.join("model_weight_history","MBConvNetNoFuse_gamma05.pt"))
mbConvNet.load_state_dict(state_dict)
mbConvNet.eval()
mbConvNet = mbConvNet.cpu()

state_dict = torch.load(os.path.join("model_weight_history","fusenet_step50.pt"))
fuseNet.load_state_dict(state_dict)
fuseNet.eval()
fuseNet = fuseNet.cpu()

state_dict = torch.load(os.path.join("model_weight_history","mfnet_step50.pt"))
mfNet.load_state_dict(state_dict)
mfNet.eval()
mfNet = mfNet.cpu()

state_dict = torch.load(os.path.join("model_weight_history","unet_step50.pt"))
uNet.load_state_dict(state_dict)
uNet.eval()
unet = uNet.cpu()

state_dict = torch.load(os.path.join("model_weight_history","segnet_step50.pt"))
segNet.load_state_dict(state_dict)
segNet.eval()
segNet = segNet.cpu()

state_dict = torch.load(os.path.join("model_weight_history","rtfnet_step50.pt"))
rtfNet.load_state_dict(state_dict)
segNet.eval()
rtfNet = rtfNet.cpu()


# %%
from sklearn.model_selection import train_test_split

rgbTrain,rgbTest,thrTrain,thrTest,labTrain,labTest,danTrain,danTest = train_test_split(rgbs,thrs,labs,danger,
        test_size = 0.2,random_state = 42)
rgbTrain,rgbValid,thrTrain,thrValid,labTrain,labValid,danTrain,danValid = train_test_split(rgbTrain,thrTrain,labTrain,danTrain,
        test_size = 0.2,random_state = 42)
len(rgbTrain),len(rgbValid),len(rgbTest),len(thrTrain),len(thrValid),len(thrTest),len(labTrain),len(labValid),len(labTest),len(danTrain),len(danValid),len(danTest)

trainDataset = CompleteRgbThrDataset(rgbTrain,thrTrain,labTrain,danTrain)
validDataset = CompleteRgbThrDataset(rgbValid,thrValid,labValid,danValid)
testDataset = CompleteRgbThrDataset(rgbTest,thrTest,labTest,danTest)

batch_size = 4


trainLoader = torch.utils.data.DataLoader(
trainDataset,batch_size=batch_size,shuffle=True,num_workers=8)

validLoader = torch.utils.data.DataLoader(
validDataset,batch_size=batch_size,shuffle=True,num_workers=8)

testLoader = torch.utils.data.DataLoader(
testDataset,batch_size=batch_size,shuffle=False,num_workers=8)
len(trainLoader),len(validLoader),len(testLoader)


# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# %%
import copy 
import time 
image_datasets = {
    'train': trainDataset, 'val': validDataset
}

dataloaders = {
    'train': trainLoader,
    'val': validLoader
}

dataset_sizes = {
    x: len(image_datasets[x]) for x in image_datasets.keys()
}

dataset_sizes


from collections import defaultdict
import torch.nn.functional as F
# from loss import dice_loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) /         (output.sum() + target.sum() + smooth)


def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=0.5):   
    bce = F.binary_cross_entropy_with_logits(pred, target)
        
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    
    return loss


def print_metrics(metrics, epoch_samples, phase = 'test'):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))    


def test_model(model,loader,return_prediction=False):
    model.eval()
    
    metrics = defaultdict(float)
    epoch_samples = 0
    iouMeter = AverageMeter()
    diceMeter = AverageMeter()
    predictions = [] 
    thermals = []
    rgbs = [] 
    _labels = []
    with torch.no_grad():
        for rgb,thr,labels,danger in loader:
            rgb = rgb.to(device)
            thr = thr.to(device)
            labels = labels.to(device)             

            outputs = model(rgb,thr)
            if return_prediction:
                for i in range(outputs.shape[0]):
                    prediction = torch.sigmoid(outputs[i].detach())
                    predictions.append(prediction.cpu().numpy())
                    thermals.append(thr[i].cpu().detach().numpy())
                    rgbs.append(rgb[i].cpu().detach().numpy())
                    _labels.append(labels[i].cpu().detach().numpy())

            loss = calc_loss(outputs, labels, metrics)
            epoch_samples += rgb.size(0)

            iou = iou_score(outputs,labels)
            dice = dice_coef(outputs,labels)
            iouMeter.update(iou, rgb.size(0))
            diceMeter.update(dice, rgb.size(0))

        print_metrics(metrics, epoch_samples,'test')
        print('test_model ##','iou:',iouMeter.avg,'dice:',diceMeter.avg)
        epoch_loss = metrics['loss'] / epoch_samples
    return predictions,thermals,rgbs,_labels,iouMeter.avg,diceMeter.avg

    
    
def train_model(model, optimizer, scheduler, num_epochs=25,early_stop=False):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    
    history = {}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        since = time.time()
        
        iouMeter = AverageMeter()
        diceMeter = AverageMeter()
        valid_iouMeter = AverageMeter()
        valid_diceMeter = AverageMeter()
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            
            for rgb,thr, labels,danger in dataloaders[phase]:
                
                rgb = rgb.to(device)
                thr = thr.to(device)
                labels = labels.to(device)             

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(rgb,thr)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += rgb.size(0)
                
                if phase == 'train':
                    iou = iou_score(outputs,labels)
                    dice = dice_coef(outputs,labels)
                    iouMeter.update(iou, rgb.size(0))
                    diceMeter.update(dice, rgb.size(0))
                    
                if phase == 'val':
                    iou = iou_score(outputs,labels)
                    dice = dice_coef(outputs,labels)
                    valid_iouMeter.update(iou, rgb.size(0))
                    valid_diceMeter.update(dice, rgb.size(0))

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples
            
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print('train iou:',iouMeter.avg,'dice:',diceMeter.avg)
        print('valid iou:',valid_iouMeter.avg,'dice:',valid_diceMeter.avg)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    if early_stop:
        model.load_state_dict(best_model_wts)
    return model


# %%
models = [fuseNet,mfNet,uNet,segNet,rtfNet,mbConvNet]
names = ["fuseNet","mfNet","uNet","segNet","rtfNet","mbConvNet"]
name2predictions = {}
name2thermals = {}
name2rgbs = {}
name2labels = {}
for name, model in zip(names,models):
    predictions,thermals,rgbs,labels,iou,dice = test_model(model.cuda(),testLoader,True)
    name2predictions[name] = predictions
    name2thermals[name] = thermals
    name2rgbs[name] = rgbs
    name2labels[name] = labels


# %%
import matplotlib.pyplot as plt 
def arr2img(arr):
    return ((arr*255).astype(np.uint8)).transpose((1,2,0))

def arr2seg(arr):
    return ((arr > 0.5).astype(np.uint8)*255).transpose((1,2,0))

def savefile(rgb,thr,gt,fuse_output,mf_output,u_output,seg_output,rtf_output,mb_output,filename):
    rgb = arr2img(rgb)
    thr = arr2img(thr)
    gt = arr2seg(gt)
    fuse_output = arr2seg(fuse_output)
    mf_output = arr2seg(mf_output)
    u_output = arr2seg(u_output)
    seg_output = arr2seg(seg_output)
    rtf_output = arr2seg(rtf_output)
    mb_output = arr2seg(mb_output)

    f, axarr = plt.subplots(nrows=1,ncols=9,figsize=(120,40))
    plt.sca(axarr[0]); 
    plt.axis('off');plt.imshow(rgb)

    plt.sca(axarr[1]); 
    plt.axis('off');plt.imshow(thr)

    plt.sca(axarr[2]); 
    plt.axis('off');plt.imshow(gt)

    plt.sca(axarr[3]); 
    plt.axis('off');plt.imshow(seg_output)

    plt.sca(axarr[4]); 
    plt.axis('off');plt.imshow(mf_output)

    plt.sca(axarr[5]); 
    plt.axis('off');plt.imshow(rtf_output)

    plt.sca(axarr[6]); 
    plt.axis('off');plt.imshow(u_output)

    plt.sca(axarr[7]); 
    plt.axis('off');plt.imshow(fuse_output)

    plt.sca(axarr[8]); 
    plt.axis('off');plt.imshow(mb_output)

    plt.savefig(filename)
    plt.close()


# %%
first_name = names[0]
for idx in range(len(name2rgbs[name]))[:]:
    print("%d/%d"%(idx,len(name2rgbs[name])),end='\r')
    
    rgb = name2rgbs[names[0]][idx]
    thr = name2thermals[names[0]][idx]
    gt = name2labels[names[0]][idx]
    
    fuse_output = name2predictions[names[0]][idx]
    mf_output = name2predictions[names[1]][idx]
    u_output = name2predictions[names[2]][idx]
    seg_output = name2predictions[names[3]][idx]
    rtf_output = name2predictions[names[4]][idx]
    mb_output = name2predictions[names[5]][idx]

    savefile(rgb=rgb,thr=thr,gt=gt,fuse_output=fuse_output,mf_output=mf_output,
    u_output=u_output,seg_output=seg_output,rtf_output=rtf_output,mb_output=mb_output,
    filename=os.path.join("result_images","%s.png"%(idx)))




