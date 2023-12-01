import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import os
import skimage.transform as skiTransf
from progressBar import printProgressBar
import scipy.io as sio
import pdb
import time
from os.path import isfile, join
import statistics
from PIL import Image
from medpy.metric.binary import dc, hd, asd, assd
import scipy.spatial
from torchmetrics import ConfusionMatrix

# from scipy.spatial.distance import directed_hausdorff



def evaluation(pred, labels, num_classes):
    if torch.cuda.is_available():
        confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    else:
        confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
    confmat = confmat(pred, labels).cpu().numpy()
    accuracy = np.array([confmat[0,0]/confmat[:,0].sum(),
                         confmat[1,1]/confmat[:,1].sum(),
                         confmat[2,2]/confmat[:,2].sum(),
                         confmat[3,3]/confmat[:,3].sum(),]).astype(float)
    res = np.array([confmat[0,0]/confmat[0,:].sum(),
                    confmat[1,1]/confmat[1,:].sum(),
                    confmat[2,2]/confmat[2,:].sum(),
                    confmat[3,3]/confmat[3,:].sum(),]).astype(float)
    accuracy[accuracy<0] = 0
    accuracy[accuracy>1] = 1
    accuracy[accuracy == float('nan')] = 0
    accuracy[np.isnan(accuracy)] = 0
    res[res<0] = 0
    res[res>1] = 1
    res[accuracy == float('nan')] = 0
    res[np.isnan(res)] = 0
    return accuracy#, res


def args_to_command(args):
    if args.loss_weights is None:
        str_loss_weights = ''
    else:
        str_loss_weights = '--loss-weights '+str(args.loss_weights[0])+\
            ' '+str(args.loss_weights[1])+' '+ \
            str(args.loss_weights[2])+' '+str(args.loss_weights[3])
    if args.augment is True:
        str_augment = '--augment'
    else:
        str_augment = ''
    if args.cuda is True:
        str_cuda = '--cuda'
    else:
        str_cuda = ''
    if args.non_label is True:
        str_non_label = '--non-label'
    else:
        str_non_label = ''
    if args.inference is True:
        str_inference = '--inference'
    else:
        str_inference = ''

    return 'python3 main.py --name={} --loss={} {} --model={} --num-workers={} --epochs={} --start-epoch={} --batch-size={} --val-batch-size={} --optimizer={} --lr={} --momentum={} --load-weights={} {} {} {} {}'.format(
                args.name,
                args.loss,
                str_loss_weights,
                args.model,
                args.num_workers,
                args.epochs,
                args.start_epoch,
                args.batch_size,
                args.val_batch_size,
                args.optimizer,
                args.lr,
                args.momentum,
                args.load_weights,
                str_augment,
                str_cuda,
                str_non_label,
                str_inference)


def save_args_to_sh(args):
    PATH_HISTORY = './training_history'
    if not os.path.exists(PATH_HISTORY):
        os.makedirs(PATH_HISTORY)
    i = 0
    while os.path.exists(PATH_HISTORY+'/'+args.model+'_'+args.name+'_training_'+str(i)+'.sh'):
        i += 1
    f = open(r''+PATH_HISTORY+'/'+args.model+'_'+args.name+'_training_'+str(i)+'.sh','w')
    f.write('#!/bin/sh\n')
    f.write(args_to_command(args))



labels = {0: 'Background', 1: 'Foreground'}

# def computeDSC(pred, gt):

#     dscAll= np.zeros((pred.shape[0],4))

#     for i_b in range(pred.shape[0]):
#         gt_id = (gt[i_b, 0, :]/0.005).round()
#         for i_c in range(pred.shape[1]-1):
#             pred_id = pred[i_b,i_c+1,:]
#             gt_class = np.zeros((gt_id.cpu().data.numpy().shape))
#             idx = np.where(gt_id.cpu().data.numpy()==(i_c+1))
#             gt_class[idx]=1
#             dscAll[i_b,i_c]=(dc(pred_id.cpu().data.numpy(),gt_class))

#     return dscAll.mean(axis=0)

def computeDSC(pred, gt):
    dscAll = []
    for i_b in range(pred.shape[0]):
        pred_id = pred[i_b, 1, :]
        gt_id = gt[i_b, 0, :]
        dscAll.append(dc(pred_id.cpu().data.numpy().astype(float), gt_id.cpu().data.numpy().astype(float)))
    DSC = np.asarray(dscAll)

    return DSC.mean()


def getImageImageList(imagesFolder):
    if os.path.exists(imagesFolder):
        imageNames = [f for f in os.listdir(imagesFolder) if isfile(join(imagesFolder, f))]

    imageNames.sort()

    return imageNames


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def DicesToDice(Dices):
    sums = Dices.sum(dim=0)
    return (2 * sums[0] + 1e-8) / (sums[1] + 1e-8)


def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    return (x == 1).float()


def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    # values are as follows : 0, 0.33333334, 0.6666667 and 0.94117647
    # output is 1 channel of discrete values : 0, 1, 2 and 3

    denom = 0.33333334  # for ACDC this value
    return (batch / denom).round().long().squeeze()

def getTargetSegmentationv2(batch):
    a = getTargetSegmentation(batch).float().numpy()
    x = np.zeros((batch.shape[0],4,batch.shape[2],batch.shape[3])).astype(float)
    x0 = np.copy(a)
    x0[x0 != 0] = -1
    x0 += 1
    x1 = np.copy(a)
    x1[x1 != 1] = 0
    x2 = np.copy(a)
    x2[x2 != 2] = 0
    x2 /= 2
    x3 = np.copy(a)
    x3[x3 != 3] = 0
    x3 /= 3
    
    x[:,0,:,:] = x0
    x[:,1,:,:] = x1
    x[:,2,:,:] = x2
    x[:,3,:,:] = x3
    return torch.from_numpy(x)


from scipy import ndimage


def inference(net, img_batch, modelName, epoch):
    total = len(img_batch)
    net.eval()

    softMax = nn.Softmax().cuda()
    CE_loss = nn.CrossEntropyLoss().cuda()

    losses = []
    for i, data in enumerate(img_batch):

        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        images, labels, img_names = data

        images = to_var(images)
        labels = to_var(labels)

        net_predictions = net(images)
        segmentation_classes = getTargetSegmentation(labels)
        CE_loss_value = CE_loss(net_predictions, segmentation_classes)
        losses.append(CE_loss_value.cpu().data.numpy())
        pred_y = softMax(net_predictions)
        masks = torch.argmax(pred_y, dim=1)

        path = os.path.join('./Results/Images/', modelName, str(epoch))

        if not os.path.exists(path):
            os.makedirs(path)

        torchvision.utils.save_image(
            torch.cat([images.data, labels.data, masks.view(labels.shape[0], 1, 256, 256).data / 3.0]),
            os.path.join(path, str(i) + '.png'), padding=0)

    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    losses = np.asarray(losses)

    return losses.mean()


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).float()

