# Color Invariant Skin Segmentation

This is the implementation of the paper [Color Invariant Skin Segmentation](https://openaccess.thecvf.com/content/CVPR2022W/FaDE-TCV/papers/Xu_Color_Invariant_Skin_Segmentation_CVPRW_2022_paper.pdf) using [FCN](https://github.com/yunlongdong/FCN-pytorch)<sup>1</sup> and [U-Net](https://github.com/zhixuhao/unet)<sup>1</sup>.

## Color Space Augmentation

Images will be augmentated in HSV color space. We change the HSV values of images and enlarge the training set. We used the parameters shown in the image below.


![color augmentation](https://github.com/HanXuMartin/Color-Invariant-Skin-Segmentation/blob/main/color%20augmentation/color_augmentation.png)

Here is the pipeline using color space augmentation for skin segmentation.

![pipeline](https://github.com/HanXuMartin/Color-Invariant-Skin-Segmentation/blob/main/examples/pipeline.png)

[Here](https://github.com/HanXuMartin/Color-Invariant-Skin-Segmentation/blob/main/color%20augmentation/HSV_converter.py) is one example on how to change the HSV value.

## Output
Color space augmentation can help skin segmentation models deal with complex illuminatlion conditions. Below is some examples. Label A means model was trained after (or with) color space augmentation while label B means before (or without) color space augmentation.
![examples](https://github.com/HanXuMartin/Color-Invariant-Skin-Segmentation/blob/main/examples/results%20examples%201.png)
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


## Train models with your own dataset

We trained our models using [ECU](https://ieeexplore.ieee.org/document/1359760) dataset. Following the steps if you want to train your own models. We suggest the groundtruth to be in the image format (like jpg, png)
## Dataset organization
Origanize your dataset as follows:
```
dataset
|-----train
        |-----image
        |-----mask
|-----validation
        |-----image
        |-----mask
```
## U-Net training
1. Open the U-Net/train.py and change the parameters in the "Training setup" section
2. Run the train.py file
```
python train.py
```
Checkpoints will be saved in the U-Net/checkpoints as default. 
## FCN training
1. Open the data_train.py/data_val.py and change the training/validation path. Remeber to change the format of the masks names in the line 
```
imgB = cv2.imread('your mask')
```
2. Open the FCN.py and change the training parameters and the saving path of checkpoints. Then run FCN.py.
```
python FCN.py
```
Checkpoints will be saved in the FCN/checkpoints as default.

## Reference
1. https://github.com/yunlongdong/FCN-pytorch
2. https://github.com/zhixuhao/unet
## Cite this repo
If you find this repo useful, please consider citing it as following:
```
@InProceedings{Xu_2022_CVPR,
    author    = {Xu, Han and Sarkar, Abhijit and Abbott, A. Lynn},
    title     = {Color Invariant Skin Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {2906-2915}
}
```





