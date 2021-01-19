import os
import sys
import numpy as np
from tqdm import tqdm
from scipy import misc
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.getcwd())
from utils import imutils

def load_txt(txt_name):
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]
        return name_list

def get_label_from_mask(mask, n_classes=0):
    label = np.unique(mask)
    label = np.setdiff1d(label, [0, 255]).astype(np.uint8)
    oht_label = np.zeros((n_classes))
    oht_label[label-1] = 1
    return oht_label

class VOCDataset(Dataset):
    def __init__(self, root_dir=None, txt_dir=None, split='train', crop_size=None, scales=None, random_crop=False, random_fliplr=False, random_scaling=False):
        # super()
        # stage: train, val, test
        self.root_dir = root_dir
        self.txt_name = os.path.join(txt_dir, split) + '.txt'
        self.name_list = load_txt(self.txt_name)
        self.crop_size = crop_size
        self.scales = scales
        self.random_scaling = random_scaling
        self.random_crop = random_crop
        self.random_fliplr = random_fliplr
        #self.img_transforms = transforms.Compose([transforms.ToTensor(),])

    def __len__(self):
        return len(self.name_list)

    def _load_image(self, idx):

        image_path = os.path.join(self.root_dir, 'JPEGImages', self.name_list[idx]+'.jpg')
        mask_path = os.path.join(self.root_dir, 'SegmentationClassAug', self.name_list[idx]+'.png')

        image = misc.imread(image_path).astype(np.float32)
        mask = misc.imread(mask_path)

        return image, mask

    def _augmentation(self, image=None, mask=None):

        if self.random_scaling:
            image, mask = imutils.img_random_scaling(image, mask, scales=self.scales)

        image = imutils.img_normalize(image)

        if self.random_fliplr:
            image, mask = imutils.img_random_fliplr(image, mask, )

        if self.random_crop:
            image, mask = imutils.img_random_crop(image, mask, crop_size=self.crop_size)

        image = imutils.img_to_CHW(image)

        return image, mask

class VOClassificationDataset(VOCDataset):
    def __init__(self, root_dir, txt_dir=None, n_classes=20, split='train', crop_size=None, scales=None, random_crop=False, random_fliplr=False, random_scaling=False):
        super(VOClassificationDataset, self).__init__(root_dir, txt_dir, split, crop_size, scales, random_crop, random_fliplr, random_scaling)

        self.n_classes = n_classes

    def __getitem__(self, idx):

        image, mask = self._load_image(idx)
        image, mask = self._augmentation(image, mask)
        label = get_label_from_mask(mask, n_classes=self.n_classes)
        
        return self.name_list[idx], image, label

class VOClassificationDatasetMultiScale(VOCDataset):
    def __init__(self, root_dir, txt_dir=None, n_classes=20, split='train', scales=None,):
        super(VOClassificationDatasetMultiScale, self).__init__(root_dir, txt_dir, split, scales=scales)

        self.n_classes = n_classes

    def __getitem__(self, idx):

        image, mask = self._load_image(idx)

        img_list = []
        for scale in self.scales:
            if scale == 1:
                img = image
            else:
                img = imutils.img_rescaling(image, scale_factor=scale)

            img, mask = self._augmentation(img, mask)
            img_lr = np.flip(img, -1)

            img_expand = np.expand_dims(img, 0)
            img_lr_expand = np.expand_dims(img_lr, 0)
            img_ = np.concatenate((img_expand, img_lr_expand), axis=0)

            img_list.append(img_)
            
        #image, mask = self._augmentation(image, mask)
        label = get_label_from_mask(mask, n_classes=self.n_classes)
        
        return self.name_list[idx], img_list, label


if __name__ == "__main__":
    root_dir = '/home/rlx/VOCdevkit/VOC2012'
    voc12dataset = VOClassificationDatasetMultiScale(root_dir=root_dir, txt_dir = 'dataset/voc', split='train', scales=[0.5, 0.75, 1.0, 1.25, 1.5])
    loader = DataLoader(voc12dataset, batch_size=1, shuffle=True, num_workers=4)
    for i, batch in tqdm(enumerate(loader),total=len(loader)):
        print(i)
