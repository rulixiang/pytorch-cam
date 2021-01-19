
import argparse
import os
from datetime import datetime

TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now())
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default='0,1', type=str, help="gpu")
parser.add_argument("--config", default='configs/voc.yaml', type=str, help="config")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import multiprocessing
from tqdm import tqdm
import numpy as np
from dataset import voc
from net import resnet_cam


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

def _infer_cam(pid, model=None, dataset=None, config=None):

    print('Validating...')

    data_loader = torch.utils.data.DataLoader(dataset[pid], batch_size=1, shuffle=False, num_workers=2, pin_memory=False)
    model.eval()

    with torch.no_grad(), torch.cuda.device(pid):
        model.cuda()
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ascii=' 123456789#'):
            _, inputs, labels = data

            #inputs = inputs.to()
            #labels = labels.to(inputs.device)
            inputs =  inputs.cuda()
            outputs = model.forward_cam(inputs)
            labels = labels.to(outputs.device)

            #loss = F.multilabel_soft_margin_loss(outputs, labels)



    return None

def main(config=None):

    infer_dataset = voc.VOClassificationDataset(root_dir=config.dataset.root_dir, txt_dir=config.dataset.txt_dir, augment=True, n_classes=config.dataset.n_classes, split=config.val.split, crop_size=config.train.crop_size, scales=config.train.scales)

    n_gpus = torch.cuda.device_count()

    if torch.cuda.is_available() is True:
        device = torch.device('cuda')
        print('%d GPUs are available:'%(n_gpus))
        for k in range(n_gpus):
            print('    %s: %s'%(args.gpu.split(',')[k], torch.cuda.get_device_name(k)))
    else:
        print('Using CPU:')
        device = torch.device('cpu')

    split_dataset = [torch.utils.data.Subset(infer_dataset, np.arange(i, len(infer_dataset), n_gpus)) for i in range (n_gpus)]

    # device
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build and initialize model
    model = resnet_cam.ResNet_CAM(n_classes=config.dataset.n_classes)
    model_path = os.path.join(config.exp.path, config.exp.checkpoint_dir, config.exp.final_weights)

    #model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    #_infer_cam(model=)
    multiprocessing.spawn(_infer_cam, nprocs=n_gpus, args=(model, split_dataset, config), join=True)

    torch.cuda.empty_cache()

    return True

if __name__=="__main__":

    config = OmegaConf.load(args.config)
    main(config)
