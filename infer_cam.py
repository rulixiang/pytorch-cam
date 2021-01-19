
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
from collections import OrderedDict

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

    data_loader = torch.utils.data.DataLoader(dataset[pid], batch_size=1, shuffle=False, num_workers=2, pin_memory=False)
    model.eval()

    with torch.no_grad(), torch.cuda.device(pid):
        model.cuda()
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ascii=' 123456789#'):
            _, input_list, labels = data

            #inputs = inputs.to()
            #labels = labels.to(inputs.device)
            for inputs in input_list:
                inputs =  inputs[0].cuda()
                _, outputs = model(inputs)
                labels = labels.to(outputs.device)

            #loss = F.multilabel_soft_margin_loss(outputs, labels)



    return None

def main(config=None):

    infer_dataset = voc.VOClassificationDatasetMultiScale(root_dir=config.dataset.root_dir, txt_dir=config.dataset.txt_dir, n_classes=config.dataset.n_classes, split=config.cam.split, scales=config.cam.scales)

    n_gpus = torch.cuda.device_count()

    split_dataset = [torch.utils.data.Subset(infer_dataset, np.arange(i, len(infer_dataset), n_gpus)) for i in range (n_gpus)]

    # device
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build and initialize model
    model = resnet_cam.ResNet(n_classes=config.dataset.n_classes, backbone=config.exp.backbone)
    model_path = os.path.join(config.exp.backbone, config.exp.checkpoint_dir, config.exp.final_weights)
    #model = nn.DataParallel(model)
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v
    model.load_state_dict(state_dict=new_state_dict, strict=True)
    model.eval()

    #_infer_cam(model=)
    print('Inferring...')
    multiprocessing.spawn(_infer_cam, nprocs=n_gpus, args=(model, split_dataset, config), join=True)

    torch.cuda.empty_cache()

    return True

if __name__=="__main__":

    config = OmegaConf.load(args.config)
    main(config)
