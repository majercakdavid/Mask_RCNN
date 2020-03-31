# Mask R-CNN for Object Detection and Segmentation

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

![Instance Segmentation Sample](assets/street.png)

The repository includes:
* Source code of Mask R-CNN built on FPN and ResNet101.
* Training code for DeepFashion2


# Getting Started
* Download and extract DeepFashion2 dataset

* [MASK_RCNN_DeepFashion2.ipynb](MASK_RCNN_DeepFashion2.ipynb) Is the easiest way to start. It shows an example of training the model as well as using it afterwards
