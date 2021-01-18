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
    def __init__(self, root_dir=None, txt_dir=None, split='train', crop_size=None, scales=None,):
        # super()
        # stage: train, val, test
        self.root_dir = root_dir
        self.txt_name = os.path.join(txt_dir, split) + '.txt'
        self.name_list = load_txt(self.txt_name)
        self.crop_size = crop_size
        self.scales = scales
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

        image, mask = imutils.img_random_scaling(image, mask, scales=self.scales)
        image = imutils.img_normalize(image)
        image, mask = imutils.img_random_fliplr(image, mask, )
        image, mask = imutils.img_random_centralcrop(image, mask, crop_size=self.crop_size)
        image = imutils.img_to_CHW(image)

        return image, mask

class VOClassificationDataset(VOCDataset):
    def __init__(self, root_dir, txt_dir=None, n_classes=20, split='train', crop_size=None, scales=None, resize_long=None):
        super(VOClassificationDataset, self).__init__(root_dir, txt_dir, split, crop_size, scales)
        self.resize_long = resize_long
        self.n_classes = n_classes

    def __getitem__(self, idx):

        image, mask = self._load_image(idx)
        image, mask = self._augmentation(image, mask)
        label = get_label_from_mask(mask, n_classes=self.n_classes)
        
        return self.name_list[idx], image, label


if __name__ == "__main__":
    root_dir = '/data1/rlx/VOC2012/VOCdevkit/VOC2012'
    txt_file = '/data1/rlx/my_cam/dataset/voc'
    voc12dataset = VOClassificationDataset(root_dir=root_dir, txt_dir = 'dataset/voc', split='train', crop_size=321, scales=[0.5, 0.75, 1.0, 1.25, 1.5])
    loader = DataLoader(voc12dataset, batch_size=4, shuffle=True, num_workers=4)
    for i, batch in tqdm(enumerate(loader),total=len(loader)):
        print(i)
