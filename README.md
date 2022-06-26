# Color Invariant Skin Segmentation

This is the implementation of the [CVPR paper](https://openaccess.thecvf.com/content/CVPR2022W/FaDE-TCV/papers/Xu_Color_Invariant_Skin_Segmentation_CVPRW_2022_paper.pdf).

## Color Space Augmentation

Change the HSV values of images and enlarge the training set. We used the parameters shown in the image below.

[Here](https://github.com/HanXuMartin/Color-Invariant-Skin-Segmentation/blob/main/color%20augmentation/HSV_converter.py) is one example on how to change the HSV value.

![color augmentation](https://github.com/HanXuMartin/Color-Invariant-Skin-Segmentation/blob/main/color%20augmentation/color_augmentation.png)


# How to use
## Requirements
- Python 3.8.5
- PyTorch
- Tensorflow
- Keras 
## Test
1. Clone the repo: 
```
git clone https://github.com/HanXuMartin/Color-Invariant-Skin-Segmentation.git
```
2. cd to the local path
```
cd Color-Invariant-Skin-Segmentation
```
3. Download the models from [here](https://drive.google.com/drive/folders/1QfoxabLN-UrsLwZjYXqmCYdHUkHxDJsf?usp=sharing)

4. For U-Net: Change the model path, testing path and output path in the test.py and then run test.py.
```
cd U-Net
python test.py
```
5. For FCN: Change the model path, testing path and output path in the prediction.py and then run prediction.py.
```
cd FCN
python prediction.py
```


## Train model with your own dataset

We trained [FCN](https://github.com/yunlongdong/FCN-pytorch) and [U-Net](https://github.com/zhixuhao/unet) using [ECU](https://ieeexplore.ieee.org/document/1359760) dataset.

Pretrained models can be found [here](https://drive.google.com/drive/folders/1QfoxabLN-UrsLwZjYXqmCYdHUkHxDJsf?usp=sharing)


