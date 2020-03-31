from datetime import datetime
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn import config
import pycocotools as coco
from samples.deepfashion2_to_coco import get_dataset_loader, get_dataset_categories
from samples import deepfashion2_mrcnn_dataset
from samples.coco_eval import COCOeval
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import os
import io
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import tqdm
import itertools

from mlxtend.plotting import plot_confusion_matrix


# def plot_confusion_matrix(cm, class_names):
#     """
#         Returns a matplotlib figure containing the plotted confusion matrix.

#         Args:
#           cm (array, shape = [n, n]): a confusion matrix of integer classes
#           class_names (array, shape = [n]): String names of the integer classes
#         """
#     # Normalize the confusion matrix.
#     cm_ratio = np.around(cm.astype('float') / cm.sum(axis=1)
#                          [:, np.newaxis], decimals=2)

#     figure = plt.figure(figsize=(16, 8))
#     plt.subplot()
#     plt.imshow(cm, interpolation='none', cmap=plt.cm.Blues)
#     plt.title("Confusion matrix")
#     plt.colorbar()
#     tick_marks = np.arange(len(class_names)+1)
#     plt.xticks(tick_marks, class_names, rotation=90)
#     plt.yticks(tick_marks, class_names)

#     # Use white text if squares are dark; otherwise black.
#     threshold = cm.max()/2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         color = "white" if cm[i, j] > threshold else "black"
#         plt.text(j, i, cm_ratio[i, j],
#                  horizontalalignment="center", color=color)

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()
#     return figure


# Root directory of the project
ROOT_DIR = os.path.abspath("./")


# from samples import coco


def assign_weight_according_occlusion(dataset_loader):
    for image in tqdm.tqdm(dataset_loader['images']):
        max_occlusion = max([annotation['occlusion']
                             for annotation in image['annotations']])
        image['weight'] = 1/max_occlusion


# Import COCO config
# To find local version
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

if __name__ == '__main__':
    with open('samples/df2_val_loader.pkl', 'rb') as fp:
        df2_val_loader = pickle.load(fp)

    classes = get_dataset_categories()
    class_names = ['background']
    class_names.extend([x['name'] for x in classes])

    assign_weight_according_occlusion(df2_val_loader)

    df2_val_dataset = deepfashion2_mrcnn_dataset.DeepFashion2Dataset()
    df2_val_dataset.load(df2_val_loader)
    df2_val_dataset.prepare()

    df2_val_coco = df2_val_dataset.load_coco(
        '../../datasets/DeepFashion2', df2_val_loader, 'validation')

    with open('samples/df2_val_results_deepfashion220200313T1302.pkl', 'rb') as fp:
        df2_val_results = pickle.load(fp)

    # Load results. This modifies results with additional attributes.
    df2_val_coco_results = df2_val_coco.loadRes(df2_val_results)

    image_ids = df2_val_dataset.image_ids
    coco_image_ids = [df2_val_dataset.image_info[id]["id"] for id in image_ids]

    cocoEval = COCOeval(
        df2_val_coco, df2_val_coco_results, 'segm')
    # cocoEval.params.imgIds = coco_image_ids
    # cocoEval.params.catIds = [1]
    cm = cocoEval.get_cm()
    class_names = [x['name'] for x in classes]
    class_names.append('nothing')
    plot_confusion_matrix(conf_mat=cm,
                          figsize=(16, 16),
                          show_absolute=True,
                          show_normed=True,
                          colorbar=True,
                          class_names=class_names)
    plt.show()
    # cocoEval.evaluate()
    # cocoEval.accumulate()
    # cocoEval.summarize()
