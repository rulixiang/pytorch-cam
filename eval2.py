
import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion

def run():
    dataset = VOCSemanticSegmentationDataset(split='val', data_dir='/home/rlx/VOCdevkit/VOC2012')
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]

    preds = []
    for id in dataset.ids:
        cam_dict = np.load(os.path.join('resnet50/cam', id + '.npy'), allow_pickle=True).item()
        cams = cam_dict['high_res']
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0.14)
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels]
        preds.append(cls_labels.copy())

    confusion = calc_semantic_segmentation_confusion(preds, labels)

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator *100
    print(iou)
    import pandas as pd
    df = pd.DataFrame(list(iou), columns=['IoU'])
    df.to_csv('eval.csv')

    print({'iou': iou, 'miou': np.nanmean(iou)})

if __name__=="__main__":
    run()