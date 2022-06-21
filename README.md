# Color Invariant Skin Segmentation

This is the implementation of the [CVPR paper](https://openaccess.thecvf.com/content/CVPR2022W/FaDE-TCV/papers/Xu_Color_Invariant_Skin_Segmentation_CVPRW_2022_paper.pdf).

## Color Space Augmentation

Change the HSV values of images and enlarge the training set. We used the parameters shown in the image below.

[Here](https://github.com/HanXuMartin/Color-Invariant-Skin-Segmentation/blob/main/color%20augmentation/HSV_converter.py) is one example on how to change the HSV value.

![color augmentation](https://github.com/HanXuMartin/Color-Invariant-Skin-Segmentation/blob/main/color%20augmentation/color_augmentation.png)

## Trained Models

We trained [FCN](https://github.com/yunlongdong/FCN-pytorch) and [U-Net](https://github.com/zhixuhao/unet) using [ECU](https://ieeexplore.ieee.org/document/1359760) dataset.

Pretrained models can be found [here](https://drive.google.com/drive/folders/1QfoxabLN-UrsLwZjYXqmCYdHUkHxDJsf?usp=sharing)

## How to Use

U-Net: Open the main.ipynb. Change the file path to your own dataset. 

FCN: Change the file path in ECUdata.py (and ECUdata_val.py if you need validation). Run FCN.py for training and prediction.py for testing.
