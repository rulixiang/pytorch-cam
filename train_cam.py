
import argparse
import os
from datetime import datetime

TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now())
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default='0,1', type=str, help="gpu")
parser.add_argument("--config", default='configs/voc.yaml', type=str, help="config")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from dataset import voc
from net import resnet_cam
from utils import imutils


def makedirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)
    return True

def get_params(model, key):

    if key == '1x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d):
                if m[0]!='module.classifier':
                    yield m[1].weight
    if key == '2x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d):
                if m[0]!='module.classifier':
                    yield m[1].bias
    if key == '10x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d):
                if m[0]=='module.classifier':
                    yield m[1].weight
    if key == '20x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d):
                if m[0]=='module.classifier':
                    yield m[1].bias

def validate(model=None, criterion=None, data_loader=None, writer=None):

    print('Validating...')

    val_loss = 0.0
    preds, gts = [], []
    model.eval()

    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ascii=' 123456789>'):
            _, inputs, labels = data

            #inputs = inputs.to()
            #labels = labels.to(inputs.device)

            outputs = model(inputs)
            labels = labels.long().to(outputs.device)

            resized_outputs = F.interpolate(outputs, size=inputs.shape[2:], mode='bilinear', align_corners=True)

            loss = criterion(resized_outputs, labels)
            val_loss += loss

            preds += list(torch.argmax(resized_outputs, dim=1).cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

    model.train()

    score = pyutils.scores(gts, preds)

    return val_loss.cpu().numpy() / float(len(data_loader)), score

def train(config=None):
    # loop over the dataset multiple times

    num_workers = config.train.batch_size * 2 

    train_dataset = voc.VOClassificationDataset(root_dir=config.dataset.root_dir, txt_dir=config.dataset.txt_dir, n_classes=config.dataset.n_classes, split=config.train.split, crop_size=config.train.crop_size, scales=config.train.scales)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    val_dataset = voc.VOClassificationDataset(root_dir=config.dataset.root_dir, txt_dir=config.dataset.txt_dir, n_classes=config.dataset.n_classes, split=config.val.split, crop_size=config.train.crop_size, scales=config.train.scales)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

    # device
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available() is True:
        device = torch.device('cuda')
        print('%d GPUs are available:'%(torch.cuda.device_count()))
        for k in range(torch.cuda.device_count()):
            print('    %s: %s'%(args.gpu.split(',')[k], torch.cuda.get_device_name(k)))
    else:
        print('Using CPU:')
        device = torch.device('cpu')

    # build and initialize model
    model = resnet_cam.ResNet_CAM(n_classes=config.dataset.n_classes)

    # save model to tensorboard 
    writer_path = os.path.join(config.exp.path, config.exp.tensorboard_dir, TIMESTAMP)
    writer = SummaryWriter(writer_path)
    dummy_input = torch.rand(2, 3, 512, 512)
    #writer.add_graph(model, dummy_input)


    model = nn.DataParallel(model)
    model.to(device)
    
    optimizer = torch.optim.SGD(
        # 
        params=[
            {
                "params": get_params(model, key="1x"),
                "lr": config.train.opt.learning_rate,
                "weight_decay": config.train.opt.weight_decay,
            },
            {
                "params": get_params(model, key="10x"),
                "lr": 10 * config.train.opt.learning_rate,
                "weight_decay": config.train.opt.weight_decay,
            },
        ],
        momentum=config.train.opt.momentum,
    )

    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])
    
    makedirs(os.path.join(config.exp.path, config.exp.checkpoint_dir))
    makedirs(os.path.join(config.exp.path, config.exp.tensorboard_dir))
    
    iteration = 0

    for epoch in range(config.train.max_epoches):
        running_loss = 0.0
        print('Training epoch %d ...'%(epoch))

        for _, data in tqdm(enumerate(train_loader), total=len(train_loader), ascii=' 123456789>', dynamic_ncols=True):

            _, inputs, labels = data
            inputs =  inputs.to(device)
            outputs = model(inputs)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            loss = F.multilabel_soft_margin_loss(outputs, labels)

            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()

            iteration += 1
            ## poly scheduler
            
            for group in optimizer.param_groups:
                #g.setdefault('initial_lr', g['lr'])
                group['lr'] = group['initial_lr']*(1 - float(iteration) / config.train.max_epoches / len(train_loader)) ** config.train.opt.power

        # save to tensorboard
        '''
        temp_k = 4
        inputs_part = inputs[0:temp_k,:]
        resized_outputs = F.interpolate(outputs, size=inputs.shape[2:], mode='bilinear', align_corners=True)
        outputs_part = resized_outputs[0:temp_k,:]
        labels_part = labels[0:temp_k,:]

        grid_inputs, grid_outputs, grid_labels = imutils.tensorboard_image(inputs=inputs_part, outputs=outputs_part, labels=labels_part, bgr=config.dataset.mean_bgr)

        writer.add_image("train/images", grid_inputs, global_step=epoch)
        writer.add_image("train/preds", grid_outputs, global_step=epoch)
        writer.add_image("train/labels", grid_labels, global_step=epoch)
        '''
        train_loss = running_loss / len(train_loader)
        print(train_loss)
        #val_loss, score = validate(model=model, criterion=criterion, data_loader=val_loader, writer=None)
        #print('train loss: %f, val loss: %f, val pixel accuracy: %f, val mIoU: %f\n'%(train_loss, val_loss, score['Pixel Accuracy'], score['Mean IoU']))

        #writer.add_scalars("loss", {'train':train_loss, 'val':val_loss}, global_step=epoch)
        #writer.add_scalar("val/acc", scalar_value=score['Pixel Accuracy'], global_step=epoch)
        #writer.add_scalar("val/miou", scalar_value=score['Mean IoU'], global_step=epoch)

    dst_path = os.path.join(config.exp.path, config.exp.checkpoint_dir, config.exp.final_weights)
    torch.save(model.state_dict(), dst_path)
    torch.cuda.empty_cache()

    return True

if __name__=="__main__":

    config = OmegaConf.load(args.config)
    train(config)
