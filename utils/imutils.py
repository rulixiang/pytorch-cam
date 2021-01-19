import random
import numpy as np
from PIL import Image

def img_to_CHW(image):
    return np.transpose(image, (2, 0, 1))

def img_normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

    image[:, :, 0] = (image[:, :, 0] / 255. - mean[0]) / std[0]
    image[:, :, 1] = (image[:, :, 1] / 255. - mean[1]) / std[1]
    image[:, :, 2] = (image[:, :, 2] / 255. - mean[2]) / std[2]

    return image

def img_random_fliplr(image, mask=None):
    if random.random() > 0.5:
        image = np.fliplr(image)
        if mask is not None:
            mask = np.fliplr(mask)

    return image, mask

def img_random_flipud(image, mask=None):

    if random.random() > 0.5:
        image = np.flipud(image)
        if mask is not None:
            mask = np.flipud(mask)
    
    return image, mask

def img_random_scaling(image, mask=None, scales=None):
    scale = random.choice(scales)
    h, w, _ = image.shape
    new_scale = [int(scale * w), int(scale * h)]
    new_image = Image.fromarray(image.astype(np.uint8)).resize(new_scale, resample=Image.BILINEAR)
    new_image = np.asarray(new_image).astype(np.float32)
    if mask is not None:
        mask = Image.fromarray(mask.astype(np.uint8)).resize(new_scale, resample=Image.NEAREST)
    
    return new_image, mask

def img_random_crop(image, mask=None, crop_size=None):
    '''
    after image normalization
    '''

    h, w, _ = image.shape
    H = max(crop_size, h)
    W = max(crop_size, w)
    pad_image = np.zeros((H,W,3), dtype=np.float32)
    
    
    H_pad = int(np.floor(H-h))
    W_pad = int(np.floor(W-w))
    pad_image[H_pad:(H_pad+h), W_pad:(W_pad+w), :] = image
    

    H_start = random.randrange(H - crop_size + 1)
    W_start = random.randrange(W - crop_size + 1)

    image = pad_image[H_start:(H_start+crop_size), W_start:(W_start+crop_size),:]

    if mask is not None:
        pad_mask = np.ones((H,W), dtype=np.float32)*255
        pad_mask[H_pad:(H_pad+h), W_pad:(W_pad+w)] = mask
        mask = pad_mask[H_start:(H_start+crop_size), W_start:(W_start+crop_size)]
    return image, mask
