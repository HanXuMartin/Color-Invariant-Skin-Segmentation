# Color Invariant Skin Segmentation
This is the implementation of the [CVPR paper](https://openaccess.thecvf.com/content/CVPR2022W/FaDE-TCV/papers/Xu_Color_Invariant_Skin_Segmentation_CVPRW_2022_paper.pdf).
## Color Space Augmentation
Change the HSV values of images and enlarge the training set. We used the parameters shown in the image below.
Here is one example ![color augmentation](https://github.com/HanXuMartin/Color-Invariant-Skin-Segmentation/blob/main/color%20augmentation/color_augmentation.png)
## Trained Models
We trained [FCN](https://github.com/yunlongdong/FCN-pytorch) and [U-Net](https://github.com/zhixuhao/unet) using [ECU](https://ieeexplore.ieee.org/document/1359760) dataset.
Pretrained models can be found [here](https://drive.google.com/drive/folders/1QfoxabLN-UrsLwZjYXqmCYdHUkHxDJsf?usp=sharing)
